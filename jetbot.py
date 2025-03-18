# jetbot_control.py (JetBot - WebSocket Server + JetBot Control + Image Send)

import asyncio
import websockets
import json
import logging
from jetbot import Robot, Camera, bgr8_to_jpeg
import base64
import time

# --- 로깅 ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
WEBSOCKET_PORT = 8766  # 웹소켓 포트 변경 (8765 -> 8766)
CAMERA_WIDTH = 300
CAMERA_HEIGHT = 300

# --- JetBot ---
robot = Robot()
camera = Camera.instance(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

# --- JetBot 명령 처리 ---
async def handle_command(command: str, parameters: dict = None):
    logger.info(f"Handling JetBot command: {command}, Parameters: {parameters}")
    if command == "move_forward":
        robot.forward(0.4)
        await asyncio.sleep(1)
        robot.stop()
    elif command == "move_backward":
        robot.backward(0.4)
        await asyncio.sleep(1)
        robot.stop()
    elif command == "turn_left":
        robot.left(0.3)
        await asyncio.sleep(0.7)
        robot.stop()
    elif command == "turn_right":
        robot.right(0.3)
        await asyncio.sleep(0.7)
        robot.stop()
    elif command == "avoid_obstacle":
        if parameters and "direction" in parameters:
            direction = parameters["direction"]
            if direction == "left":
                robot.left(0.5)
                await asyncio.sleep(1.2)
            elif direction == "right":
                robot.right(0.5)
                await asyncio.sleep(1.2)
        else:
            robot.left(0.5)
            await asyncio.sleep(1.2)

        robot.forward(0.4)
        await asyncio.sleep(1)
        robot.stop()
    elif command == "none":
        pass
    else:
        logger.warning(f"Unknown command: {command}")

# --- 웹소켓 서버 ---
async def websocket_handler(websocket, path):
    logger.info(f"New WebSocket connection: {websocket.remote_address}")
    try:
        # 이미지 전송 태스크 시작
        image_task = asyncio.ensure_future(send_images(websocket))

        async for message in websocket:
            try:
                data = json.loads(message)
                logger.debug(f"Received message: {data}")
                if "command" in data:
                    await handle_command(data["command"], data.get("parameters"))
                else:
                    logger.warning("Invalid message format")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
            except Exception as e:
                logger.exception(f"Error handling command: {e}")
    except websockets.exceptions.ConnectionClosedOK:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        image_task.cancel()
        try:
            await image_task
        except asyncio.CancelledError:
            pass

async def start_websocket_server():
    server = await websockets.serve(websocket_handler, "0.0.0.0", WEBSOCKET_PORT)
    logger.info(f"WebSocket server started (port: {WEBSOCKET_PORT})")
    await server.wait_closed()

# --- 이미지 전송 함수 ---
async def send_images(websocket):
    while True:
        try:
            image = camera.value
            if image is None:
              print("Camera image is None")
              await asyncio.sleep(0.1)
              continue

            image_data = bgr8_to_jpeg(image)
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            await websocket.send(json.dumps({"image": image_base64}))
            await asyncio.sleep(0.1)  # 전송 간격 조절 (FPS)

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("WebSocket connection closed (image send)")
            break
        except Exception as e:
            logger.exception(f"Error sending image: {e}")
            break

async def main():
    await start_websocket_server()

if __name__ == "__main__":
    asyncio.ensure_future(main())
