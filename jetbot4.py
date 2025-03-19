import asyncio
import websockets
import json
import logging
from jetbot import Robot, Camera, bgr8_to_jpeg
import base64

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 설정
WEBSOCKET_PORT = 8766
CAMERA_WIDTH = 300
CAMERA_HEIGHT = 300

# JetBot 초기화
robot = Robot()
camera = Camera.instance(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

# 명령 처리
async def handle_command(command: str, parameters: dict = None):
    parameters = parameters or {}
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
                robot.left(0.5); await asyncio.sleep(0.5)
                robot.right(0.5); await asyncio.sleep(0.5)
            robot.stop()
        elif command == "none":
            pass
        logger.info(f"Executed command: {command}")
    except Exception as e:
        logger.error(f"Command execution error: {e}")

# 이미지 캡처
async def capture_image():
    try:
        image = camera.value
        if image is not None:
            return base64.b64encode(bgr8_to_jpeg(image)).decode('utf-8')
        return None
    except Exception as e:
        logger.error(f"Image capture error: {e}")
        return None

# 웹소켓 핸들러
async def websocket_handler(websocket, path):
    logger.info("WebSocket connection established")
    try:
        async for message in websocket:
            data = json.loads(message)
            command = data.get("command", "none")
            parameters = data.get("parameters", {})
            await handle_command(command, parameters)
            if data.get("get_image", False):
                image_base64 = await capture_image()
                if image_base64:
                    await websocket.send(json.dumps({"image": image_base64}))
                else:
                    await websocket.send(json.dumps({"error": "Image capture failed"}))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")

# 메인 함수
async def main():
    server = await websockets.serve(websocket_handler, "0.0.0.0", WEBSOCKET_PORT)
    logger.info(f"WebSocket server running on port {WEBSOCKET_PORT}")
    # 서버가 종료될 때까지 대기
    await asyncio.Future()  # 무한 대기

# 실행 함수 (Python 3.6 호환)
def run():
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        robot.stop()
        camera.stop()
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        loop.close()

if __name__ == "__main__":
    run()
