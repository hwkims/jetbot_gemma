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
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llava:7b"  # Or your specific downloaded model
JETBOT_WEBSOCKET_URL = "ws://192.168.137.233:8766"  # Replace with JetBot's IP
STATIC_DIR = Path(__file__).parent / "static"
TTS_VOICE = "ko-KR-HyunsuNeural"
ITERATIONS = 50      # Increased iterations for longer autonomous run
DELAY_SECONDS = 1.5  # Delay between actions (adjust as needed)

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
    iterations: Optional[int] = ITERATIONS
    delay: Optional[float] = DELAY_SECONDS

# --- JetBot Commands (Expanded) ---
JETBOT_COMMANDS = {
    "left": {"speed": 0.3, "duration": 0.5},
    "right": {"speed": 0.3, "duration": 0.5},
    "forward": {"speed": 0.4, "duration": 0.8},
    "backward": {"speed": 0.4, "duration": 0.7},
    "stop": {},
    "avoid_obstacle_left": {"speed": 0.3, "duration": 1.2, "angle": 45},  # Example parameters
    "avoid_obstacle_right": {"speed": 0.3, "duration": 1.2, "angle": -45},
    "follow_lane": {"speed": 0.4, "duration": 0.5},  # Adjust speed/duration
    "search_for_lane": {"speed": 0.2, "duration": 0.5},  # Slow, short movements
    "none": {}
}

# --- TTS Function ---
async def generate_tts(text: str) -> str:
    try:
        if not text or text.isspace():
            text = "처리할 내용이 없습니다."
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        temp_file = "temp_tts_" + str(int(time.time())) + ".mp3"
        temp_filepath = STATIC_DIR / temp_file
        await communicate.save(temp_filepath)
        with open(temp_filepath, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        os.remove(temp_filepath)
        logger.info(f"TTS generated successfully: {text[:50]}...")
        return audio_data
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return await generate_tts("음성 생성 중 오류가 발생했습니다.")

# --- Ollama Interaction ---
async def query_ollama(prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_data] if image_data else [],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 512,  # Increased prediction length
        }
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(OLLAMA_HOST + "/api/generate", json=data)
            response.raise_for_status()
            result = response.json()

            try:
                parsed_response = json.loads(result.get("response", "{}")).get("commands", [])
                if not parsed_response:
                    raise json.JSONDecodeError("No 'commands' key", result.get("response", ""), 0)
                logger.info(f"Ollama response: {parsed_response}")
                return {"commands": parsed_response}

            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning(f"Initial JSON parsing failed. Trying alternative parsing. Response: {result.get('response')}")
                try:
                    parsed_response = json.loads(result.get("response", "{}"))
                    if isinstance(parsed_response, dict) and "commands" in parsed_response:
                        return {"commands": parsed_response["commands"]}
                    elif isinstance(parsed_response, list):
                        return {"commands": parsed_response}
                    else:
                        raise json.JSONDecodeError("Could not parse as dict or list", result.get("response",""), 0)
                except (json.JSONDecodeError, KeyError, TypeError):
                    logger.error(f"Alternative JSON parsing failed. Response: {result.get('response')}")
                    return {"commands": [{"command": "stop", "parameters": {}, "tts": "Response parsing error."}]}

    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama HTTP error ({e.response.status_code}): {e}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": f"HTTP error: {e.response.status_code}"}]}
    except httpx.RequestError as e:
        logger.error(f"Ollama Request error: {e}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "Request error."}]}
    except Exception as e:
        logger.error(f"General Ollama error: {e}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "An unexpected error occurred."}]}

# --- JetBot Communication ---
async def send_command_to_jetbot(command: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[str]:
    try:
        async with websockets.connect(
            JETBOT_WEBSOCKET_URL, ping_interval=20, ping_timeout=60
        ) as websocket:
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
        return None
    except Exception as e:
        logger.error(f"JetBot command error: {e}")
        return None

# --- API Endpoint (Autonomous Loop) ---
@app.post("/api/autonomous")
async def autonomous_control(request_data: OllamaRequest):
    all_responses = []
    current_image = None

    for i in range(request_data.iterations):
        logger.info(f"--- Iteration {i+1} of {request_data.iterations} ---")

        if i > 0:
            current_image = await send_command_to_jetbot("none", {})
            if not current_image:
                logger.error("Failed to get image from Jetbot. Stopping.")
                break
        else:
            current_image = await send_command_to_jetbot("none", {})
            if not current_image:
                logger.error("Failed to get initial image. Stopping.")
                break

        # --- Perception and Planning (Ollama) ---
        if current_image:
            prompt = (
                "Analyze the image for the Jetbot.  Provide information and a command.\n"
                "1. **Obstacles:**  Are there any obstacles?  If so, where are they (left, right, center)?\n"
                "2. **Path:** Is there a clear path (road, lane, corridor) visible?  If so, describe its direction.\n"
                "3. **Command:**  Based on the above, suggest ONE command for the Jetbot.\n"
                "\n"
                "Output MUST be JSON, in this format:\n"
                "```json\n"
                "{\n"
                "  \"commands\": [\n"
                "    {\n"
                "      \"command\": \"COMMAND_NAME\",\n"
                "      \"parameters\": { \"speed\": VALUE, \"duration\": VALUE, \"angle\": VALUE },\n"
                "      \"tts\": \"TTS_MESSAGE\",\n"
                "      \"obstacle_info\": { \"present\": true/false, \"position\": \"left/right/center/none\" },\n"
                "      \"path_info\": { \"visible\": true/false, \"direction\": \"straight/left/right/unknown\" }\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "```\n"
                "* **COMMAND_NAME:**  `forward`, `backward`, `left`, `right`, `stop`, `avoid_obstacle_left`, `avoid_obstacle_right`, `follow_lane`, `search_for_lane`.\n"
                "* **parameters:**  `speed`, `duration`, and `angle` as needed by the command.\n"
                "* **tts:**  A short description for text-to-speech.\n"
                "* **obstacle_info:**  Information about obstacles.\n"
                "* **path_info:** Information about the visible path.\n"
                "\n"
                "Prioritize obstacle avoidance. If no clear path, use `search_for_lane`. If a path is visible, use `follow_lane`."
            )
        else:
            prompt = (
                "Determine the best action for the Jetbot.  Output JSON.  "
                "Commands: `forward`, `backward`, `left`, `right`, `stop`, `avoid_obstacle_left`, `avoid_obstacle_right`, `follow_lane`, `search_for_lane`."
            )

        ollama_response = await query_ollama(prompt, current_image)
        commands = ollama_response.get("commands", [])

        # --- Action Selection and Execution ---
        if commands:
            cmd = commands[0]
            jetbot_command = cmd.get("command", "none")
            parameters = cmd.get("parameters", {})
            tts_text = cmd.get("tts", "No description.")
            obstacle_info = cmd.get("obstacle_info", {"present": False, "position": "none"})
            path_info = cmd.get("path_info", {"visible": False, "direction": "unknown"})

            # --- Decision-Making Logic (Prioritize Obstacle Avoidance) ---

            if obstacle_info["present"]:
                if obstacle_info["position"] == "left":
                    jetbot_command = "avoid_obstacle_right"  # Turn right
                elif obstacle_info["position"] == "right":
                    jetbot_command = "avoid_obstacle_left"   # Turn left
                else:  # Center or unknown
                    jetbot_command = "stop"  # Stop for safety
                tts_text = f"Obstacle detected.  Avoiding {obstacle_info['position']}."

            elif path_info["visible"]:
                jetbot_command = "follow_lane"  # Follow the detected path
                tts_text = f"Following the path. Direction: {path_info['direction']}."

            elif jetbot_command not in JETBOT_COMMANDS: # If Ollama gives bad command
                 jetbot_command = "search_for_lane"
                 tts_text = "Searching for a clear path."

            # --- Command Execution ---
            if jetbot_command != "none":
                await send_command_to_jetbot(jetbot_command, parameters)
        else:
            jetbot_command = "stop"
            parameters = {}
            tts_text = "No valid command. Stopping."
            logger.warning("No valid command from Ollama.")

        # --- Generate TTS and prepare response ---
        encoded_audio = await generate_tts(tts_text)
        response_data = {
            "iteration": i + 1,
            "response": tts_text,
            "jetbot_command": jetbot_command,
            "image": "data:image/jpeg;base64," + (current_image or ""),
            "audio": "data:audio/mp3;base64," + encoded_audio,
        }
        all_responses.append(response_data)

        # --- Delay ---
        await asyncio.sleep(request_data.delay)

    return JSONResponse(content={"results": all_responses})

# --- Run the FastAPI application ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
