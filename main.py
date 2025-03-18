# main.py (PC - FastAPI + Ollama + JetBot Command + Static Files + TTS)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
from typing import Optional, AsyncGenerator, Dict, Any
from pydantic import BaseModel
import httpx  # Ollama API
import websockets  # JetBot 명령
import edge_tts
import subprocess
import base64
import io
from pathlib import Path

# --- 로깅 ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
OLLAMA_HOST = "http://localhost:11434"  # Ollama (PC에서 실행)
MODEL_NAME = "gemma3:4b"
JETBOT_WEBSOCKET_URL = "ws://192.168.137.233:8766"  # JetBot 웹소켓 변경!
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
async def query_ollama_stream(prompt: str, image_data: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
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
                async for chunk in response.aiter_text():
                    try:
                        for line in chunk.splitlines():
                            if line.strip():
                                yield json.loads(line)
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
async def speak(text: str):
    try:
        communicate = edge_tts.Communicate(text, VOICE, rate=TTS_RATE)
        await communicate.save("output.mp3")
        if os.name == 'nt': # 윈도우
            os.system("start output.mp3")  # Windows
        else: # 리눅스/맥
            os.system("aplay output.mp3") # Linux
        # os.system("afplay output.mp3") # macOS
        await asyncio.sleep(1) # aplay/afplay가 끝날때까지 기다림
        os.remove("output.mp3")

    except Exception as e:
        logger.error(f"TTS Error: {e}")

# --- JetBot 명령 전송 ---
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
    full_response_text = ""

    try:
        async for response_part in query_ollama_stream(request_data.prompt, image_data):
            if "response" in response_part:
                full_response_text += response_part["response"]

        if not full_response_text.strip():
            return JSONResponse({"response": "No information from Ollama", "jetbot_command": "none"})

        # JetBot 명령 결정
        jetbot_command = "none"
        parameters = {}
        response_lower = full_response_text.lower()

        if request_data.action == "navigate":
            if "obstacle" in response_lower or "object" in response_lower:
                jetbot_command = "avoid_obstacle"
                parameters = {"direction": "left"}
            elif request_data.direction_hint:
                if request_data.direction_hint == "left":
                    jetbot_command = "turn_left"
                elif request_data.direction_hint == "right":
                    jetbot_command = "turn_right"
                elif request_data.direction_hint == "forward":
                    jetbot_command = "move_forward"
                elif request_data.direction_hint == "backward":
                    jetbot_command = "move_backward"
        elif request_data.action == "describe":
            jetbot_command = "none"

        # JetBot에 명령 전송
        await send_command_to_jetbot(jetbot_command, parameters)

        # PC에서 TTS 실행
        await speak(full_response_text)

        return JSONResponse({
            "response": full_response_text,
            "jetbot_command": jetbot_command,
        })

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise
    except Exception as e:
        logger.exception(f"Error during API request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 정적 파일 제공 (index.html) ---
static_dir = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(static_dir / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
