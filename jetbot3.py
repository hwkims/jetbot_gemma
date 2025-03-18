# jetbot_control.py (JetBot - WebSocket Server + JetBot Control + Image Send)

import asyncio
import websockets
import json
import logging
from jetbot import Robot, Camera, bgr8_to_jpeg
import base64
import time
import random

# --- 로깅 ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
WEBSOCKET_PORT = 8766
CAMERA_WIDTH = 300
CAMERA_HEIGHT = 300

# --- JetBot ---
robot = Robot()
camera = Camera.instance(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

# --- JetBot 명령 처리: 단순화된 동작 ---
async def handle_command(command: str, parameters: dict = None):
    logger.info(f"Handling JetBot command: {command}, Parameters: {parameters}")

    try:
        if command == "forward_fast":
            robot.forward(0.6)
            await asyncio.sleep(1)  # 비동기 sleep
            robot.stop()
        elif command == "forward_medium":
            robot.forward(0.4)
            await asyncio.sleep(1)
            robot.stop()
        elif command == "forward_slow":
            robot.forward(0.2)
            await asyncio.sleep(1)
            robot.stop()
        elif command == "backward_fast":
            robot.backward(0.6)
            await asyncio.sleep(1)
            robot.stop()
        elif command == "backward_medium":
            robot.backward(0.4)
            await asyncio.sleep(1)
            robot.stop()
        elif command == "backward_slow":
            robot.backward(0.2)
            await asyncio.sleep(1)
            robot.stop()
        elif command == "left_fast":
            robot.left(0.5)
            await asyncio.sleep(0.7)
            robot.stop()
        elif command == "left_medium":
            robot.left(0.3)
            await asyncio.sleep(0.7)
            robot.stop()
        elif command == "left_slow":
            robot.left(0.1)
            await asyncio.sleep(0.7)
            robot.stop()
        elif command == "right_fast":
            robot.right(0.5)
            await asyncio.sleep(0.7)
            robot.stop()
        elif command == "right_medium":
            robot.right(0.3)
            await asyncio.sleep(0.7)
            robot.stop()
        elif command == "right_slow":
            robot.right(0.1)
            await asyncio.sleep(0.7)
            robot.stop()
        elif command == "avoid_obstacle":
            direction = parameters.get("direction", "left") if parameters else "left"
            if direction == "left":
                await handle_command("left_medium")
            elif direction == "right":
                await handle_command("right_medium")
            await handle_command("forward_medium")
        elif command == "stop":
            robot.stop()
        elif command == "rotate":  # 각도 파라미터 삭제
            if parameters and "angle" in parameters: # angle 파라미터 삭제
                angle = parameters["angle"]
                if angle > 0: # angle 파라미터 삭제
                    await handle_command("right_medium") #90도 대신
                else:
                    await handle_command("left_medium") # -90도 대신
            else:
                logger.warning("Rotate command missing 'angle' parameter") # angle 파라미터 삭제

        elif command == "random_action":
            random_action = random.choice(list(BASIC_COMMANDS.keys()))
            await handle_command(random_action)

        elif command == "custom_command":
            if parameters and "prompt" in parameters:
                logger.info(f"Received Custom command prompt: {parameters['prompt']}")
            else:
                logger.warning("custom_command missing prompt")
        elif command == "none":
            pass
        else:
            logger.warning(f"Unknown command: {command}")

    except Exception as e:
        logger.exception(f"Error in handle_command: {e}")

# 미리 정의된 기본 명령
BASIC_COMMANDS = {
    "forward_fast": {"speed": 0.6, "duration": 1.0},
    "forward_medium": {"speed": 0.4, "duration": 1.0},
    "forward_slow": {"speed": 0.2, "duration": 1.0},
    "backward_fast": {"speed": 0.6, "duration": 1.0},
    "backward_medium": {"speed": 0.4, "duration": 1.0},
    "backward_slow": {"speed": 0.2, "duration": 1.0},
    "left_fast": {"speed": 0.5, "duration": 0.7},
    "left_medium": {"speed": 0.3, "duration": 0.7},
    "left_slow": {"speed": 0.1, "duration": 0.7},
    "right_fast": {"speed": 0.5, "duration": 0.7},
    "right_medium": {"speed": 0.3, "duration": 0.7},
    "right_slow": {"speed": 0.1, "duration": 0.7},
}

# --- 웹소켓 서버 ---
async def websocket_handler(websocket, path):
    logger.info(f"New WebSocket connection: {websocket.remote_address}")
    try:
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
                logger.warning("Camera image is None")
                await asyncio.sleep(0.1)
                continue
            image_data = bgr8_to_jpeg(image)
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            await websocket.send(json.dumps({"image": image_base64}))
            #await asyncio.sleep(0.1) # 이미지 전송 간격 주석처리
        except websockets.exceptions.ConnectionClosedOK:
            logger.info("WebSocket connection closed (image send)")
            break
        except Exception as e:
            logger.exception(f"Error sending image: {e}")
            break

async def main():
    await start_websocket_server()

import sys # sys 임포트

if __name__ == "__main__":
    # Python 버전 확인 및 실행
    if sys.version_info >= (3, 7):  # Python 3.7 이상
        asyncio.run(main())
    else:  # Python 3.6 이하
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main())
        except RuntimeError as e:
            if "Event loop is already running" not in str(e):
                raise
            else: # 이미 실행중
                logger.info("Event loop already running.  Continuing...")
        finally:
            try:
                loop.close() # 닫기
            except RuntimeError as e:
                if "Cannot close a running event loop" not in str(e):
                    raise
                else: # 이미 실행중
                    logger.info("Event loop already running.  Continuing...")
