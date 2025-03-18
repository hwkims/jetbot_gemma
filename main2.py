from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
from typing import Optional, AsyncGenerator, Dict, Any
from pydantic import BaseModel
import httpx
import websockets
import edge_tts
import base64
import io
from pathlib import Path
import os

# --- 로깅 ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"
JETBOT_WEBSOCKET_URL = "ws://192.168.137.233:8766"
VOICE = "ko-KR-HyunsuNeural"
TTS_RATE = "+30%"

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 ---
class OllamaRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    action: str = "navigate"
    direction_hint: Optional[str] = None

# --- Ollama API ---
async def query_ollama_stream(prompt: str, image_data: Optional[str] = None) -> AsyncGenerator[str, None]:  # Changed return type
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_data] if image_data else [],
        "stream": True,
        "format": "json",
        "options": {"temperature": 0.1, "top_p": 0.7},
    }
    logger.debug(f"Ollama 요청: {prompt[:50]}..., 이미지: {'있음' if image_data else '없음'}")

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/generate",
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=httpx.Timeout(60.0, connect=5.0, read=120.0, write=5.0),
            ) as response:
                response.raise_for_status()
                full_response = "" # 완전한 응답을 저장.
                async for chunk in response.aiter_text():
                    try:
                        for line in chunk.splitlines():
                            if line.strip():
                                # JSON 형식인지 확인
                                if line.strip().startswith('{') and line.strip().endswith('}'):
                                    json_part = json.loads(line)
                                    if "response" in json_part:
                                        full_response += json_part["response"] # "response"부분만

                                    if json_part.get("done"):  # done 필드가 True이면
                                        yield full_response # 최종 결과 yield
                                        full_response = ""  # Reset for next potential response
                                else:
                                  logger.warning(f"Received non-JSON data: {line.strip()}")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON Decode Error: {e}, chunk: {chunk}")


    except httpx.RequestError as e:
        logger.error(f"httpx RequestError: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except httpx.HTTPStatusError as e:
        logger.error(f"httpx HTTP Status Error: {e}, Response: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during Ollama streaming")
        raise HTTPException(status_code=500, detail=str(e))

# --- TTS ---
async def speak(text: str) -> str:
    try:
        communicate = edge_tts.Communicate(text, VOICE, rate=TTS_RATE)
        temp_file = "temp_audio.wav"
        await communicate.save(temp_file)
        with open(temp_file, "rb") as f:
            audio_data = f.read()
        os.remove(temp_file)
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Error: {e}")

# --- JetBot 명령 ---
async def send_command_to_jetbot(command: str, parameters: Optional[Dict[str, Any]] = None):
    try:
        async with websockets.connect(JETBOT_WEBSOCKET_URL) as websocket:
            await websocket.send(json.dumps({"command": command, "parameters": parameters}))
            logger.info(f"Sent command to JetBot: {command}, {parameters}")
    except Exception as e:
        logger.error(f"Failed to send command to JetBot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- FastAPI 엔드포인트 ---
@app.post("/api/generate")
async def generate(request_data: OllamaRequest):
    image_data = request_data.image.split(",")[1] if request_data.image and request_data.image.startswith("data:image") else None
    full_response_text = "" # 완전한 응답

    try:
        async for response_str in query_ollama_stream(request_data.prompt, image_data):
            full_response_text = response_str # 최종 응답 저장.
            logger.info(f"응답: {full_response_text}")

        if not full_response_text:
            return JSONResponse({"response": "Ollama로부터 받은 정보가 없습니다.", "jetbot_command": "none"})



        # JetBot 명령 결정
        jetbot_command = "none"
        parameters = {}
        response_lower = full_response_text.lower()

        if request_data.action == "navigate":
            if "obstacle" in response_lower or "object" in response_lower:
                jetbot_command = "avoid_obstacle"
                parameters = {"direction": "left"}  # LLM과 연동 필요
            elif request_data.direction_hint:
                if request_data.direction_hint == "left":
                    jetbot_command = "turn_left"
                elif request_data.direction_hint == "right":
                    jetbot_command = "turn_right"
                elif request_data.direction_hint == "forward":
                    jetbot_command = "move_forward"
                elif request_data.direction_hint == "backward":
                    jetbot_command = "move_backward"
                elif request_data.direction_hint == "stop": # 정지 추가.
                    jetbot_command = "stop"
        elif request_data.action == "describe":
            jetbot_command = "none"
        elif request_data.action == "custom":  # 사용자 지정
              jetbot_command = "custom"
              parameters = {"prompt": request_data.prompt}

        await send_command_to_jetbot(jetbot_command, parameters)

        # TTS
        encoded_audio = await speak(full_response_text)

        return JSONResponse({
            "response": full_response_text,
            "jetbot_command": jetbot_command,
            "audio": encoded_audio
        })

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise
    except Exception as e:
        logger.exception(f"Error during API request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 정적 파일 ---
static_dir = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(static_dir / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
