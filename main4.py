from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel
import httpx
import websockets
import edge_tts
import base64
import os
import unicodedata
import re
import time
from pathlib import Path

# --- 로깅 ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"
JETBOT_WEBSOCKET_URL = "ws://192.168.137.233:8766"
VOICE = "ko-KR-HyunsuNeural"
MEMORY_FILE = "memory.json"

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
    speed: Optional[float] = None
    duration: Optional[float] = None
    angle: Optional[float] = None

# --- JetBot 명령어 매핑 ---
JETBOT_COMMANDS = {
    "left_medium": {"command": "left", "parameters": {"speed": 0.3, "duration": 0.7}, "tts": "왼쪽으로 회전합니다."},
    "right_medium": {"command": "right", "parameters": {"speed": 0.3, "duration": 0.7}, "tts": "오른쪽으로 회전합니다."},
    "forward_medium": {"command": "forward", "parameters": {"speed": 0.4, "duration": 1.0}, "tts": "앞으로 이동합니다."},
    "backward_medium": {"command": "backward", "parameters": {"speed": 0.4, "duration": 1.0}, "tts": "뒤로 이동합니다."},
    "stop": {"command": "stop", "parameters": {}, "tts": "정지합니다."},
    "avoid_obstacle": {"command": "avoid_obstacle", "parameters": {"direction": "left"}, "tts": "장애물을 피해 왼쪽으로 이동합니다."},
    "right_fast": {"command": "right", "parameters": {"speed": 0.5, "duration": 1.4}, "tts": "시계 방향으로 회전합니다."},
    "left_fast": {"command": "left", "parameters": {"speed": 0.5, "duration": 1.4}, "tts": "반시계 방향으로 회전합니다."},
    "dance": {"command": "dance", "parameters": {}, "tts": "춤을 춥니다!"},
    "custom_command": {"command": "custom_command", "parameters": None, "tts": None},
    "none": {"command": "none", "parameters": {}, "tts": ""}
}

# --- Ollama API ---
async def query_ollama_stream(prompt: str, image_data: Optional[str] = None):
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_data] if image_data else [],
        "stream": True,
        "format": "json",
        "options": {"temperature": 0.1, "top_p": 0.7},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0, read=120.0, write=5.0)) as client:
        async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=data) as response:
            response.raise_for_status()
            full_response = ""
            async for chunk in response.aiter_bytes():
                decoded_chunk = chunk.decode('utf-8')
                for line in decoded_chunk.splitlines():
                    if line.strip() and line.strip().startswith('{') and line.strip().endswith('}'):
                        json_part = json.loads(line)
                        if "response" in json_part:
                            full_response += json_part["response"]
                        if json_part.get("done"):
                            full_response = unicodedata.normalize('NFC', full_response)
                            yield full_response
                            full_response = ""

# --- TTS ---
async def speak(text: str) -> str:
    try:
        communicate = edge_tts.Communicate(text=text, voice=VOICE)
        temp_file = f"temp_audio_{int(time.time())}.wav"
        await communicate.save(temp_file)
        with open(temp_file, "rb") as f:
            audio_data = f.read()
        os.remove(temp_file)
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Error: {e}")

# --- JetBot 명령 및 이미지 요청 ---
async def send_command_to_jetbot(command: str, parameters: Optional[Dict[str, Any]] = None):
    try:
        async with websockets.connect(JETBOT_WEBSOCKET_URL) as websocket:
            # 명령 전송
            msg = {"command": command, "parameters": parameters or {}, "get_image": True}
            await websocket.send(json.dumps(msg))
            logger.info(f"Sent command to JetBot: {msg}")

            # 이미지 수신 대기
            response = await websocket.recv()
            data = json.loads(response)
            if "image" in data:
                logger.debug("Received image from JetBot")
                return data["image"]
            return None
    except Exception as e:
        logger.error(f"JetBot command error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 메모리 관련 함수 ---
def load_memory(filename: str = MEMORY_FILE, max_length: int = 50) -> list:
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)[-max_length:]
        return []
    except Exception as e:
        logger.warning(f"Failed to load memory: {e}")
        return []

def save_memory(memory_entry: dict, filename: str = MEMORY_FILE):
    memory = load_memory(filename)
    memory.append(memory_entry)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)

# --- FastAPI 엔드포인트 ---
@app.post("/api/generate")
async def generate(request_data: OllamaRequest):
    image_data = request_data.image.split(",")[1] if request_data.image and "," in request_data.image else None
    full_response_text = ""

    try:
        async for response_str in query_ollama_stream(request_data.prompt, image_data):
            full_response_text = response_str

        match = re.search(r'"description":\s*"(.*?)"', full_response_text, re.DOTALL)
        description = match.group(1).strip() if match else full_response_text

        jetbot_command = "none"
        parameters = {}
        tts_text = ""

        if request_data.action == "navigate":
            if request_data.direction_hint in JETBOT_COMMANDS:
                cmd_info = JETBOT_COMMANDS[request_data.direction_hint]
                jetbot_command = cmd_info["command"]
                parameters = cmd_info["parameters"].copy()
                tts_text = cmd_info["tts"]
            elif "obstacle" in description.lower() or "object" in description.lower():
                cmd_info = JETBOT_COMMANDS["avoid_obstacle"]
                jetbot_command = cmd_info["command"]
                parameters = cmd_info["parameters"].copy()
                tts_text = cmd_info["tts"]
        elif request_data.action == "describe":
            tts_text = description
        elif request_data.action == "custom":
            jetbot_command = "custom_command"
            parameters = {"prompt": request_data.prompt}
            tts_text = f"사용자 지정 명령: {request_data.prompt}"

        if request_data.speed:
            parameters["speed"] = request_data.speed
        if request_data.duration:
            parameters["duration"] = request_data.duration
        if request_data.angle:
            parameters["angle"] = request_data.angle

        # JetBot에 명령 전송 및 이미지 수신
        image_base64 = await send_command_to_jetbot(jetbot_command, parameters)
        encoded_audio = await speak(tts_text) if tts_text else ""

        memory_entry = {
            "timestamp": time.time(),
            "prompt": request_data.prompt,
            "image": image_data,
            "action": request_data.action,
            "direction_hint": request_data.direction_hint,
            "response": full_response_text,
            "description": description,
            "jetbot_command": jetbot_command,
            "tts_text": tts_text,
            "audio": encoded_audio,
        }
        save_memory(memory_entry)

        return JSONResponse({
            "response": description,
            "jetbot_command": jetbot_command,
            "image": f"data:image/jpeg;base64,{image_base64}" if image_base64 else None,
            "audio": encoded_audio
        })
    except Exception as e:
        logger.error(f"API generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 정적 파일 ---
static_dir = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
