from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
import base64
import os
import time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import httpx
import websockets
from pathlib import Path
import edge_tts

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"  # Correct model name if necessary
JETBOT_WEBSOCKET_URL = "ws://192.168.137.233:8766"  # Correct IP if necessary
STATIC_DIR = Path(__file__).parent / "static"
MEMORY_FILE = "memory.json"
TTS_VOICE = "ko-KR-HyunsuNeural"

# --- FastAPI Setup ---
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 루트 경로
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# --- Data Models ---
class OllamaRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    action: str = "navigate"
    direction_hint: Optional[str] = None
    speed: Optional[float] = None
    duration: Optional[float] = None
    angle: Optional[float] = None

# --- JetBot Commands ---
JETBOT_COMMANDS = {
    "left_medium": {"command": "left", "parameters": {"speed": 0.3, "duration": 0.7}, "tts": "젯봇이 왼쪽으로 살짝 몸을 틀어요!"},
    "right_medium": {"command": "right", "parameters": {"speed": 0.3, "duration": 0.7}, "tts": "젯봇이 오른쪽으로 멋지게 회전합니다!"},
    "forward_medium": {"command": "forward", "parameters": {"speed": 0.4, "duration": 1.0}, "tts": "젯봇, 앞으로 돌진!"},
    "backward_medium": {"command": "backward", "parameters": {"speed": 0.4, "duration": 1.0}, "tts": "젯봇이 살짝 뒤로 물러섭니다!"},
    "stop": {"command": "stop", "parameters": {}, "tts": "젯봇, 잠시 멈춰!"},
    "dance": {"command": "dance", "parameters": {}, "tts": "젯봇이 신나게 춤을 춰요!"},
    "none": {"command": "none", "parameters": {}, "tts": "젯봇이 가만히 있어요."}
}

# --- TTS Function (edge-tts) ---
async def generate_tts(text: str) -> str:
    try:
        if not text or text.isspace():
            text = "젯봇이 할 말을 잃었어요!"  # Better default message
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        temp_file = "temp_tts_" + str(int(time.time())) + ".mp3"
        temp_filepath = STATIC_DIR / temp_file
        await communicate.save(temp_filepath)
        with open(temp_filepath, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        os.remove(temp_filepath)
        logger.info(f"TTS generated successfully: {text}")
        return audio_data
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return await generate_tts("젯봇이 살짝 당황했어요!")  # Consistent fallback


# --- Gemma 3 Interaction ---
async def query_gemma3(prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
    base_prompt = (
        "너의 이름은 젯봇이야 살아있는 생명체처럼 재치 있고 생동감 있게 반응해. "
        f"사용자가 '{prompt}'라고 말했어. "
    )
    if image_data:
        base_prompt += (
            "이미지를 보고 구체적으로 설명하고, 상황에 맞는 행동을 제안해줘. "
            "예를 들어, 앞에 장애물이 있으면 '앗, 앞에 벽이 있어! 돌아가야겠어!' 같은 식으로."
        )
    instruction = (
        "어떤 명령이든 반드시 행동을 제안해줘. '할 수 없다'는 답변은 절대 하지 말고, "
        "최대한 창의적이고 자연스럽게 반응해. "
        "내가 할 다음 행동을 JSON 형식으로 제안해줘. "
        "반드시 다음 형식을 엄격히 준수해: {'commands': [{'command': '명령어', 'parameters': {'speed': 값, 'duration': 값}, 'tts': '음성 메시지'}]}\n"
        "명령어는 'forward', 'backward', 'left', 'right', 'stop', 'dance', 'describe' 중 하나\n"
        "'forward', 'backward', 'left', 'right'일 때 speed는 0.3~0.7, duration은 0.5~3.0\n"
    )
    data = {
        "model": MODEL_NAME,
        "prompt": base_prompt + instruction,
        "images": [image_data] if image_data else [],
        "stream": False,
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # Set a reasonable timeout
            response = await client.post(OLLAMA_HOST + "/api/generate", json=data)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            parsed_response = json.loads(result.get("response", "{}")).get("commands", [])
            logger.info(f"Gemma3 response: {parsed_response}")
            return {"commands": parsed_response}
    except httpx.HTTPError as e:
        logger.error(f"Gemma3 HTTP error: {e}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "젯봇 통신 오류! 잠시 멈춤!"}]} # More specific error message
    except Exception as e:
        logger.error(f"Gemma3 error: {e}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "젯봇이 살짝 당황했어요! 그래도 멈추고 생각해볼게요!"}]}

# --- JetBot Communication ---
async def send_command_to_jetbot(command: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[str]:
    try:
        async with websockets.connect(JETBOT_WEBSOCKET_URL) as websocket:
            msg = {"command": command, "parameters": parameters or {}, "get_image": True}
            await websocket.send(json.dumps(msg))
            logger.info(f"Sent command to JetBot: {msg}")
            response = await websocket.recv()
            data = json.loads(response)
            image = data.get("image")
            if not image:
                logger.warning("No image received from JetBot")
            return image
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"JetBot connection closed unexpectedly: {e}")
        return None  # Handle connection loss
    except Exception as e:
        logger.error(f"JetBot command error: {e}")
        return None

# --- Memory ---
def load_memory(filename: str = MEMORY_FILE) -> List[Dict[str, Any]]:
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)[-50:]  # Keep only the last 50 entries
        return []
    except (FileNotFoundError, json.JSONDecodeError) as e: # Handle file errors
        logger.warning(f"Error loading memory: {e}")
        return []

def save_memory(memory_entry: Dict[str, Any], filename: str = MEMORY_FILE):
    memory = load_memory(filename)
    memory.append(memory_entry)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=4)
    except OSError as e:  # Handle file write errors
        logger.error(f"Error saving memory: {e}")

# --- API Endpoint ---
@app.post("/api/generate")
async def generate(request_data: OllamaRequest):
    image_base64 = await send_command_to_jetbot("none", {})  # Get current image
    image_data = image_base64 if image_base64 else None

    gemma_response = await query_gemma3(request_data.prompt, image_data)
    commands = gemma_response.get("commands", []) or [{"command": "none", "parameters": {}, "tts": "젯봇이 뭘 할지 고민 중이에요!"}]
    cmd = commands[0]

    jetbot_command = cmd["command"]
    parameters = cmd["parameters"]
    tts_text = cmd["tts"]

    if request_data.action == "navigate" and request_data.direction_hint in JETBOT_COMMANDS:
        cmd_info = JETBOT_COMMANDS[request_data.direction_hint]
        jetbot_command = cmd_info["command"]
        parameters = cmd_info["parameters"].copy()
        tts_text = cmd_info["tts"]

    # Optional parameters override
    if request_data.speed is not None:
        parameters["speed"] = request_data.speed
    if request_data.duration is not None:
        parameters["duration"] = request_data.duration
    if request_data.angle is not None:
        parameters["angle"] = request_data.angle

    new_image_base64 = await send_command_to_jetbot(jetbot_command, parameters) # Execute command and get new image
    encoded_audio = await generate_tts(tts_text)

    response = {
        "response": tts_text,
        "jetbot_command": jetbot_command,
        "image": "data:image/jpeg;base64," + (new_image_base64 or image_base64) if new_image_base64 or image_base64 else "",  # Use new or current image
        "audio": "data:audio/mp3;base64," + encoded_audio,
    }

    save_memory({
        "timestamp": time.time(),
        "prompt": request_data.prompt,
        "action": request_data.action,
        "direction_hint": request_data.direction_hint,
        "jetbot_command": jetbot_command,
        "tts_text": tts_text,
    })

    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
