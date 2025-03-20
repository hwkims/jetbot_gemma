import asyncio
import websockets
import json
import logging
from jetbot import Robot, Camera, bgr8_to_jpeg
import base64
import time

# --- OpenCV Handling ---
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
WEBSOCKET_PORT = 8766
CAMERA_WIDTH = 300
CAMERA_HEIGHT = 300
RECONNECT_DELAY = 5
FPS = 20

# --- JetBot Initialization ---
try:
    robot = Robot()
    camera = Camera.instance(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    logger.info("JetBot initialized successfully.")
except Exception as e:
    logger.error(f"JetBot initialization failed: {e}")
    robot = None
    camera = None

# --- Command Handling ---
async def handle_command(command, parameters=None):
    parameters = parameters or {}
    if not robot:
        logger.error("Robot not initialized.")
        return
    try:
        duration = parameters.get("duration", 1.0)
        speed = parameters.get("speed", 0.4)
        if command == "forward":
            robot.forward(speed)
            await asyncio.sleep(duration)
        elif command == "backward":
            robot.backward(speed)
            await asyncio.sleep(duration)
        elif command == "left":
            robot.left(speed)
            await asyncio.sleep(duration)
        elif command == "right":
            robot.right(speed)
            await asyncio.sleep(duration)
        elif command == "stop":
            pass  # Don't stop automatically
        else:
            logger.warning(f"Unknown command: {command}")
        robot.stop()  # Stop after *every* action

    except Exception as e:
        logger.error(f"Command execution error: {e}")
        robot.stop()

# --- WebSocket Handler ---
async def websocket_handler(websocket, path):
    logger.info("WebSocket connection established")

    async def send_image_stream():
        if not camera:
            logger.error("Camera not initialized.")
            return
        try:
            while True:
                frame = camera.value
                if OPENCV_AVAILABLE:
                    _, encoded_image = cv2.imencode('.jpg', frame)
                    image_base64 = base64.b64encode(encoded_image).decode('utf-8')
                else:
                    image_base64 = base64.b64encode(bgr8_to_jpeg(frame)).decode('utf-8')
                await websocket.send(json.dumps({"image": image_base64}))
                await asyncio.sleep(1/FPS)
        except websockets.exceptions.ConnectionClosedError:
            logger.warning("Image stream connection closed.")
        except Exception as e:
            logger.error(f"Image stream error: {e}")

    if hasattr(asyncio, 'create_task'):
        image_stream_task = asyncio.create_task(send_image_stream())
    else:
        image_stream_task = asyncio.ensure_future(send_image_stream())

    try:
        async for message in websocket:
            data = json.loads(message)
            command = data.get("command", "none")
            parameters = data.get("parameters", {})
            await handle_command(command, parameters)
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"WebSocket connection closed: {e}.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        image_stream_task.cancel()
        if robot:
            robot.stop()

# --- Main Function ---
async def main():
    while True:
        try:
            async with websockets.serve(websocket_handler, "0.0.0.0", WEBSOCKET_PORT, ping_interval=20, ping_timeout=60):
                logger.info(f"WebSocket server running on port {WEBSOCKET_PORT}")
                await asyncio.Future()
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"Address already in use. Retrying in {RECONNECT_DELAY} seconds...")
                await asyncio.sleep(RECONNECT_DELAY)
            else:
                logger.error(f"Server encountered an OSError: {e}")
                break
        except Exception as e:
            logger.error(f"Server encountered an error: {e}")
            break

if __name__ == "__main__":
    if robot:
        robot.stop()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if robot:
            robot.stop()
    finally:
        loop.close()
