from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
import base64
import time
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import httpx
from pathlib import Path
import edge_tts
import io

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llava:7b"
JETBOT_WEBSOCKET_URL = "ws://192.168.137.233:8766"  # Replace with your Jetbot's IP
STATIC_DIR = Path(__file__).parent / "static"
TTS_VOICE = "ko-KR-HyunsuNeural"  # Or "en-US-JennyNeural"
ITERATIONS = 10
DELAY_SECONDS = 0.1

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "WEBSOCKET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Pydantic Models ---
class OllamaRequest(BaseModel):
    prompt: str = Field(..., description="The user's text prompt.")
    iterations: int = Field(ITERATIONS, description="Number of iterations for autonomous control.")
    delay: float = Field(DELAY_SECONDS, description="Delay between actions in seconds.")

# --- HTML Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# --- TTS Function ---
async def generate_tts(text: str) -> str:
    try:
        if not text or text.isspace():
            text = "Processing..."
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        temp_file = f"temp_tts_{int(time.time())}.mp3"
        temp_filepath = STATIC_DIR / temp_file
        await communicate.save(temp_filepath)
        with open(temp_filepath, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        os.remove(temp_filepath)
        return audio_data
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return await generate_tts("TTS failed.")

# --- Ollama Interaction ---
async def query_ollama(prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_data] if image_data else [],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 512,
        },
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
                logger.warning(f"Initial JSON parsing failed.  Trying alternative. Response: {result.get('response')}")
                try:
                    parsed_response = json.loads(result.get("response","{}"))
                    if isinstance(parsed_response, dict) and "commands" in parsed_response:
                        return {"commands": parsed_response["commands"]}
                    elif isinstance(parsed_response, list):
                        return {"commands": parsed_response}
                    else:
                        raise json.JSONDecodeError("Could not parse as dict or list", result.get("response",""), 0)

                except (json.JSONDecodeError, KeyError, TypeError):
                    logger.error(f"Alternative JSON parsing failed. Response: {result.get('response')}")
                    return {
                        "commands": [
                            {
                                "command": "stop",
                                "parameters": {},
                                "tts": "Could not understand response.",
                            }
                        ]
                    }

    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama HTTP error ({e.response.status_code}): {e}")
        return {
            "commands": [
                {"command": "stop", "parameters": {}, "tts": f"HTTP error: {e.response.status_code}"}
            ]
        }
    except httpx.RequestError as e:
        logger.error(f"Ollama Request error: {e}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "Request error."}]}
    except Exception as e:
        logger.error(f"General Ollama error: {e}")
        return {
            "commands": [{"command": "stop", "parameters": {}, "tts": "An unexpected error occurred."}]
        }

# --- WebSocket Connections ---
client_websocket: Optional[WebSocket] = None
jetbot_websocket: Optional[WebSocket] = None

@app.websocket("/ws/jetbot")
async def jetbot_websocket_endpoint(websocket: WebSocket):
    global jetbot_websocket
    await websocket.accept()
    logger.info("Jetbot WebSocket connected")
    jetbot_websocket = websocket
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received from Jetbot: {data}")  # Log *all* messages
    except WebSocketDisconnect:
        logger.info("Jetbot WebSocket disconnected")
        jetbot_websocket = None
    except Exception as e:
        logger.error(f"Jetbot WebSocket error: {e}")
        jetbot_websocket = None

@app.websocket("/ws/client")
async def client_websocket_endpoint(websocket: WebSocket):
    global client_websocket
    await websocket.accept()
    logger.info("Client WebSocket connected")
    client_websocket = websocket
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received from client: {data}")
            try:
                message = json.loads(data)
                if "command" in message:
                    if jetbot_websocket:
                        await jetbot_websocket.send_text(data)  # Forward directly
                    else:
                        logger.warning("Jetbot not connected.")
                        await websocket.send_text(json.dumps({"error": "Jetbot not connected"}))

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client: {data}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))

    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")
        client_websocket = None
    except Exception as e:
        logger.error(f"Client WebSocket error: {e}")
        client_websocket = None

# --- Image Stream Receiver ---
async def receive_image_stream():
    global current_image_base64
    if not jetbot_websocket:
        logger.warning("Jetbot WebSocket not connected.")
        return

    try:
        while True:
            message = await jetbot_websocket.receive_text()
            data = json.loads(message)
            if "image" in data:
                current_image_base64 = data["image"]
                if client_websocket:
                    try:
                        await client_websocket.send_text(json.dumps({"image": current_image_base64}))
                    except Exception as e:
                        logger.error(f"Failed to send image to client: {e}")
            else:
                logger.debug(f"Received from Jetbot: {message}") # Log other

    except WebSocketDisconnect:
        logger.info("Jetbot WebSocket disconnected in receive_image_stream")
    except Exception as e:
        logger.error(f"Image stream receive error: {e}")

# --- Autonomous Control Loop ---
@app.post("/api/autonomous")
async def autonomous_control(request_data: OllamaRequest):
    global current_image_base64
    all_responses = []

    for i in range(request_data.iterations):
        logger.info(f"--- Iteration {i+1} of {request_data.iterations} ---")

        if not current_image_base64:
            logger.warning("No image available from Jetbot. Waiting...")
            await asyncio.sleep(0.5)
            if not current_image_base64:
                logger.error("Still no image. Stopping.")
                break

        prompt = (
            "Analyze the image and determine the best action for the Jetbot.\n"
            "Provide a JSON response with 'command', 'parameters' (speed, duration), 'tts', 'objects', and 'path'.\n"
            "```json\n"
            "{\n"
            "  \"commands\": [\n"
            "    {\n"
            "      \"command\": \"COMMAND_NAME\",\n"
            "      \"parameters\": { \"speed\": VALUE, \"duration\": VALUE },\n"
            "      \"tts\": \"TTS_MESSAGE\",\n"
            "      \"objects\": [\n"
            "        { \"name\": \"OBJECT_TYPE\", \"position\": \"POSITION\", \"distance_cm\": DISTANCE },\n"
            "        ...\n"
            "      ],\n"
            "      \"path\": { \"visible\": true/false, \"direction\": \"DIRECTION\", \"width_cm\": WIDTH }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "Where:\n"
            "- `COMMAND_NAME`: `forward`, `backward`, `left`, `right`, `stop`, `greet`.\n"
            "- `parameters`: `speed` (0.0 to 1.0) and `duration` (seconds).\n"
            "- `tts`: Short description.\n"
            "- `objects`: List of objects with `name`, `position`, `distance_cm`.\n"
            "- `path`: `visible` (boolean), `direction`, `width_cm`.\n"
            "\n"
            "Instructions:\n"
            "- Prioritize safety: Avoid obstacles.\n"
            "- Greet people/animals if close.\n"
            "- Follow a clear path if visible.\n"
            "- Be specific with `speed` and `duration`.\n"
            "- Explain your reasoning in `tts`.\n"
            "- If uncertain, stop.\n"
            "- You may issue a sequence of commands for a 'dance'."
          )

        ollama_response = await query_ollama(prompt, current_image_base64)
        commands = ollama_response.get("commands", [])

        if commands:
            for cmd in commands:
                jetbot_command = cmd.get("command", "none")
                parameters = cmd.get("parameters", {})
                tts_text = cmd.get("tts", "No description.")
                objects = cmd.get("objects", [])  # Not used in decision-making
                path = cmd.get("path", {"visible": False, "direction": "unknown"}) # Not used

                if jetbot_command not in ["forward", "backward", "left", "right", "stop", "greet"]:
                    jetbot_command = "stop"
                    tts_text = "Invalid command received. Stopping."
                    logger.warning(f"Invalid command received: {jetbot_command}")

                if jetbot_command != "none":
                    if jetbot_websocket:
                        command_message = json.dumps({
                            "command": jetbot_command,
                            "parameters": parameters
                        })
                        await jetbot_websocket.send_text(command_message)
                        await asyncio.sleep(parameters.get("duration", 1.0) + 0.1) # Wait
                    else:
                        logger.warning("Jetbot WebSocket not connected.")
                        tts_text = "Jetbot not connected."
                        jetbot_command = "stop"
        else:
            jetbot_command = "stop"
            tts_text = "No valid command. Stopping."
            logger.warning("No valid command from Ollama.")

        encoded_audio = await generate_tts(tts_text)
        response_data = {
            "iteration": i + 1,
            "response": tts_text,
            "jetbot_command": jetbot_command,
            "audio": "data:audio/mp3;base64," + encoded_audio,
        }
        all_responses.append(response_data)

        if client_websocket:
            try:
                await client_websocket.send_text(json.dumps(response_data))
            except Exception as e:
                logger.error(f"Failed to send response to client: {e}")

        await asyncio.sleep(request_data.delay)

    return JSONResponse(content={"results": all_responses})

@app.post("/api/stt")
async def speech_to_text(request: Request):
    body = await request.body()
    try:
        audio_bytes_io = io.BytesIO(body)
        text = await transcribe_audio(audio_bytes_io)
        return JSONResponse(content={"text": text})
    except Exception as e:
        logger.error(f"STT endpoint error: {e}")
        return JSONResponse(content={"error": "STT processing failed"}, status_code=500)

async def transcribe_audio(audio_data: io.BytesIO) -> str:
    import speech_recognition as sr
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_data) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language="ko-KR")
        logger.info(f"Transcribed text: {text}")
        return text
    except sr.UnknownValueError:
        logger.error("Could not understand audio")
        return "Could not understand audio"
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition; {e}")
        return "Could not request results from Google Speech Recognition"
    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        return "Transcription error"

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(receive_image_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
