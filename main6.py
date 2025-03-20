from fastapi import FastAPI, HTTPException
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
# !! IMPORTANT !!  Update these with your actual values:
OLLAMA_HOST = "http://localhost:11434"  #  Ollama server address
MODEL_NAME = "gemma3:4b"  #  Ollama model name.  Make sure this is correct!
JETBOT_WEBSOCKET_URL = "ws://192.168.137.233:8766"   #  JetBot's IP address and port
STATIC_DIR = Path(__file__).parent / "static"  # Path to your static files
MEMORY_FILE = "memory.json"  # File to store conversation history
TTS_VOICE = "ko-KR-HyunsuNeural" # TTS voice

# --- FastAPI Setup ---
app = FastAPI()

# --- CORS (Cross-Origin Resource Sharing) ---
# Allows requests from your browser (running on a different origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Static Files ---
# Serves static files (HTML, CSS, JavaScript, images) from the 'static' directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Root Endpoint (HTML Page) ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main HTML page (index.html).
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# --- Data Models (Pydantic) ---
class OllamaRequest(BaseModel):
    """
    Defines the structure of the incoming request body for the /api/generate endpoint.
    """
    prompt: str  # User's text input
    image: Optional[str] = None  # Base64 encoded image (optional)
    action: str = "navigate"  # Action to perform (e.g., "navigate", "describe")
    direction_hint: Optional[str] = None  #  (e.g., "left_medium", "forward_medium")
    speed: Optional[float] = None  # Movement speed
    duration: Optional[float] = None  # Movement duration
    angle: Optional[float] = None  # Rotation angle

# --- JetBot Command Definitions ---
JETBOT_COMMANDS = {
    # Maps command names to JetBot actions, parameters, and TTS responses.
    "left_medium": {"command": "left", "parameters": {"speed": 0.3, "duration": 0.7}, "tts": "왼쪽으로 살짝 이동!"},
    "right_medium": {"command": "right", "parameters": {"speed": 0.3, "duration": 0.7}, "tts": "오른쪽으로 살짝 이동!"},
    "forward_medium": {"command": "forward", "parameters": {"speed": 0.4, "duration": 1.0}, "tts": "앞으로 이동!"},
    "backward_medium": {"command": "backward", "parameters": {"speed": 0.4, "duration": 1.0}, "tts": "뒤로 이동!"},
    "stop": {"command": "stop", "parameters": {}, "tts": "정지!"},
    "dance": {"command": "dance", "parameters": {}, "tts": "춤을 춥니다!"},
    "none": {"command": "none", "parameters": {}, "tts": "대기 중."}
}

# --- Text-to-Speech (TTS) Function ---
async def generate_tts(text: str) -> str:
    """
    Generates speech from text using edge-tts and returns the audio as a base64 string.
    """
    try:
        if not text or text.isspace():
            text = "처리할 내용이 없습니다."  # Default text if input is empty
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        temp_file = "temp_tts_" + str(int(time.time())) + ".mp3"  # Unique filename
        temp_filepath = STATIC_DIR / temp_file
        await communicate.save(temp_filepath)
        with open(temp_filepath, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        os.remove(temp_filepath)  # Delete the temporary file
        logger.info(f"TTS generated successfully: {text}")
        return audio_data
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return await generate_tts("음성 생성 중 오류가 발생했습니다.")  # Fallback TTS

# --- Gemma 3 Interaction Function ---
async def query_gemma3(prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
    """
    Sends a prompt and (optionally) an image to the Ollama model and returns the response.
    """
    if image_data:
        # Image description prompt (Object-focused, no quality comments, examples)
        base_prompt = (
            f"다음은 base64로 인코딩된 이미지입니다: {image_data} "
            "이미지에 있는 모든 물체를 최대한 자세하고 정확하게 설명해주세요. "
            "다음 사항들을 반드시 포함해야 합니다:\n"
            "- 각 물체의 종류 (예: 키보드, 책, 컵, 사람 등).\n"
            "- 각 물체의 색상.\n"
            "- 각 물체의 형태 (예: 직사각형, 원형, 둥근 모서리 등).\n"
            "- 각 물체의 상대적인 위치 (예: 이미지 중앙, 왼쪽 상단, ~옆에, ~위에 등).\n"
            "- 이미지에 텍스트가 있다면, 텍스트의 내용, 글꼴 스타일 (가능하다면), 색상, 위치.\n"
            "다음은 설명 예시입니다:\n"
            "예시 1: '이미지 중앙에 흰색 키보드가 있습니다. 키보드 위에는 검은색 글자가 있습니다.'\n"
            "예시 2: '이미지 왼쪽에 갈색 책상이 있고, 책상 위에 흰색 종이가 놓여 있습니다. 종이 옆에는 검은색 펜이 있습니다.'\n"
            "예시 3: '이미지에 \"Hello, World!\"라는 텍스트가 흰색 글꼴로 중앙에 있습니다.'\n"
            "최대한 객관적이고 상세하게 묘사해주세요.  분위기나 느낌에 대한 설명은 하지 마세요."
        )
        instruction = (
            "이제, 위에서 생성한 이미지 설명을 'tts' 필드에 넣고, 'command' 필드에는 'describe'를 넣어 JSON 형식으로 응답해주세요. "
            "다른 행동을 제안하지 마세요. 오직 이미지 설명만 'tts'에 넣어야 합니다. "
            "다음 형식을 엄격히 준수해야 합니다: {'commands': [{'command': 'describe', 'parameters': {}, 'tts': '이미지 설명'}]}"
        )

    else:
        # No-image prompt (for user text commands)
        base_prompt = f"사용자 요청: '{prompt}'"
        instruction = (
            "사용자 요청에 따라, 젯봇이 수행할 수 있는 적절한 행동을 '하나만' JSON 형식으로 제안해주세요. "
            "다음 형식을 엄격히 준수해야 합니다: {'commands': [{'command': '명령어', 'parameters': {'speed': 값, 'duration': 값}, 'tts': '음성 메시지'}]}\n"
            "'명령어'는 'forward', 'backward', 'left', 'right', 'stop', 'dance', 'describe' 중 하나여야 합니다.\n"
            "'forward', 'backward', 'left', 'right' 명령어의 경우, 'speed'는 0.3에서 0.7 사이, 'duration'은 0.5에서 3.0 사이의 값이어야 합니다.\n"
            "tts 음성메시지는 간결하고, 젯봇의 행동을 설명하는 내용이어야 합니다."
        )

    data = {
        "model": MODEL_NAME,
        "prompt": base_prompt + instruction,
        "images": [],  # Image is always in the prompt
        "stream": False,
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_HOST + "/api/generate", json=data)
            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()
            parsed_response = json.loads(result.get("response", "{}")).get("commands", [])
            logger.info(f"Gemma3 response: {parsed_response}")
            return {"commands": parsed_response}

    except httpx.HTTPError as e:
        logger.error(f"Gemma3 HTTP error: {e}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "통신 오류! 잠시 멈춤."}]}
    except Exception as e:
        logger.error(f"Gemma3 error: {e}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "오류 발생! 잠시 멈춤."}]}


# --- JetBot Communication Function ---
async def send_command_to_jetbot(command: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Sends a command to the JetBot over a WebSocket connection and returns the captured image.
    """
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

# --- Memory Management Functions ---
def load_memory(filename: str = MEMORY_FILE) -> List[Dict[str, Any]]:
    """
    Loads conversation history from a JSON file.
    """
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)[-50:]  # Keep only the last 50 entries
        return []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading memory: {e}")
        return []

def save_memory(memory_entry: Dict[str, Any], filename: str = MEMORY_FILE):
    """
    Saves a new memory entry to the conversation history file.
    """
    memory = load_memory(filename)
    memory.append(memory_entry)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=4)
    except OSError as e:
        logger.error(f"Error saving memory: {e}")

# --- Main API Endpoint (/api/generate) ---
@app.post("/api/generate")
async def generate(request_data: OllamaRequest):
    """
    Handles incoming requests, interacts with the LLM and JetBot, and returns a response.
    """
    image_base64 = await send_command_to_jetbot("none", {}) # Get image *before* anything else
    image_data = image_base64 if image_base64 else None

     # Image Verification (Logging) - Keep this!
    if image_data:
        logger.info(f"Image data (first 100 chars): {image_data[:100]}...")

    # Describe action takes precedence
    if request_data.action == 'describe':
        user_prompt = ""  # Empty prompt for describe
        gemma_response = await query_gemma3(user_prompt, image_data)
        # We expect ONLY a describe command, so we use it directly.
        if gemma_response and gemma_response.get("commands"):
            cmd = gemma_response["commands"][0]  # Directly access the first command
            jetbot_command = cmd["command"]
            parameters = cmd["parameters"]
            tts_text = cmd["tts"]
        else:
            jetbot_command = "none"
            parameters = {}
            tts_text = "이미지 설명 생성에 실패했습니다."
    else:
        # Handle other actions (non-describe)
        user_prompt = request_data.prompt
        gemma_response = await query_gemma3(user_prompt, image_data)
        commands = gemma_response.get("commands", []) or [{"command": "none", "parameters": {}, "tts": "명령을 기다리는 중."}]
        cmd = commands[0]

        jetbot_command = cmd["command"]
        parameters = cmd["parameters"]
        tts_text = cmd["tts"]


        if request_data.action == "navigate" and request_data.direction_hint in JETBOT_COMMANDS:
            # Only override if gemma did *not* return 'describe'
            if jetbot_command != "describe":
                cmd_info = JETBOT_COMMANDS[request_data.direction_hint]
                jetbot_command = cmd_info["command"]
                parameters = cmd_info["parameters"].copy()
                tts_text = cmd_info["tts"]

    # Optional parameter overrides (speed, duration, angle)
    if request_data.speed is not None:
        parameters["speed"] = request_data.speed
    if request_data.duration is not None:
        parameters["duration"] = request_data.duration
    if request_data.angle is not None:
        parameters["angle"] = request_data.angle

    # Execute the command on the JetBot and get the new image
    new_image_base64 = await send_command_to_jetbot(jetbot_command, parameters)
    encoded_audio = await generate_tts(tts_text)  # Generate TTS from the response

    # --- Prepare the API Response ---
    # Include Cache-Control headers to prevent browser caching of the image
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    response = {
        "response": tts_text,  # The text response (for display)
        "jetbot_command": jetbot_command,  # The command sent to the JetBot
        "image": "data:image/jpeg;base64," + (new_image_base64 or image_base64) if new_image_base64 or image_base64 else "",  #  image data
        "audio": "data:audio/mp3;base64," + encoded_audio,  # The TTS audio data
    }

    # --- Save to Memory ---
    save_memory({
        "timestamp": time.time(),
        "prompt": request_data.prompt,
        "action": request_data.action,
        "direction_hint": request_data.direction_hint,
        "jetbot_command": jetbot_command,
        "tts_text": tts_text,
    })

    return JSONResponse(content=response, headers=headers)  # Return the JSON response


# --- Run the Application (using Uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)  # reload=False for debugging
