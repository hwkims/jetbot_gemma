import asyncio
import websockets
import json
import logging
from jetbot import Robot, Camera, bgr8_to_jpeg
import base64
import time

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Improved logging format
logger = logging.getLogger(__name__)

# --- Configuration ---
WEBSOCKET_PORT = 8766
CAMERA_WIDTH = 300
CAMERA_HEIGHT = 300
RECONNECT_DELAY = 5  # Seconds to wait before attempting to reconnect

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
        return False

    try:
        if command == "forward":
            robot.forward(parameters.get("speed", 0.4))
            await asyncio.sleep(parameters.get("duration", 1.0))
            robot.stop()
        elif command == "backward":
            robot.backward(parameters.get("speed", 0.4))
            await asyncio.sleep(parameters.get("duration", 1.0))
            robot.stop()
        elif command == "left":
            robot.left(parameters.get("speed", 0.3))
            await asyncio.sleep(parameters.get("duration", 0.7))
            robot.stop()
        elif command == "right":
            robot.right(parameters.get("speed", 0.3))
            await asyncio.sleep(parameters.get("duration", 0.7))
            robot.stop()
        elif command == "stop":
            robot.stop()
        elif command == "dance":
            for _ in range(2):
                robot.left(0.5)
                await asyncio.sleep(0.5)
                robot.right(0.5)
                await asyncio.sleep(0.5)
            robot.stop()
        elif command == "describe" or command == "none":
            pass  # No action needed, just capture the image
        else:
            logger.warning(f"Unknown command: {command}")
            robot.stop()  # Always stop on an unknown command
        logger.info(f"Executed command: {command}")
        return True
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        robot.stop()  # Ensure the robot stops on *any* error during command execution
        return False

# --- Image Capture ---
async def capture_image():
    if not camera:
        logger.error("Camera not initialized.")
        return None
    try:
        image = camera.value
        if image is not None:
            return base64.b64encode(bgr8_to_jpeg(image)).decode('utf-8')
        logger.warning("Camera returned no image.")
        return None
    except Exception as e:
        logger.error(f"Image capture error: {e}")
        return None

# --- WebSocket Handler ---
async def websocket_handler(websocket, path):
    logger.info("WebSocket connection established")
    while True:  # Continuous loop to handle reconnections
        try:
            async for message in websocket:
                data = json.loads(message)
                command = data.get("command", "none")
                parameters = data.get("parameters", {})
                await handle_command(command, parameters)
                image_base64 = await capture_image()
                if image_base64:
                    await websocket.send(json.dumps({"image": image_base64}))
                else:
                    logger.error("Image capture failed, sending empty response")
                    await websocket.send(json.dumps({"image": ""}))

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"WebSocket connection closed: {e}.  Attempting to reconnect...")
            if robot:
                robot.stop()
            await asyncio.sleep(RECONNECT_DELAY)  # Wait before reconnecting
        except websockets.exceptions.ConnectionClosedOK:
            logger.info("WebSocket connection closed normally.")
            if robot:
                robot.stop()
            break  # Exit loop on normal close
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if robot:
                robot.stop() # Stop the robot
            await asyncio.sleep(RECONNECT_DELAY) # Wait before reconnect
        #except Exception as e: #Removed this block as all exception are handled
        #    logger.error(f"WebSocket error: {e}")
        #    break  # Break on other errors

# --- Main Function ---
async def main():
    while True: # Continuous loop for the server
        try:
            async with websockets.serve(websocket_handler, "0.0.0.0", WEBSOCKET_PORT):
                logger.info(f"WebSocket server running on port {WEBSOCKET_PORT}")
                await asyncio.Future()  # Run forever *inside* the `with` statement
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"Address already in use: {e}.  Retrying in {RECONNECT_DELAY} seconds...")
                await asyncio.sleep(RECONNECT_DELAY)
            else:
                logger.error(f"Server encountered an OSError: {e}") # Other OS errors are unlikely
                break # If not address in use, do not retry
        except Exception as e: # Generic exception
            logger.error(f"Server encountered an error: {e}")
            break # Break on other errors
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if robot:
            robot.stop()
    finally:
        loop.close()
