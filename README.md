# JetBot-Ollama-Control
![image](https://github.com/user-attachments/assets/3579a3d2-2adf-4136-ae7b-ac8d0fabf516)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**JetBot을 Ollama, FastAPI, Edge TTS를 이용하여 제어하는 프로젝트입니다.**  JetBot의 카메라 피드를 실시간으로 받아 Ollama를 통해 분석하고,  분석 결과에 따라 JetBot을 제어하거나 사용자에게 음성으로 상황을 설명합니다.

## 기능

*   **실시간 영상 분석:** JetBot의 카메라 영상을 실시간으로 받아 `gemma3:4b` 모델을 이용해 분석합니다.
*   **JetBot 제어:**  분석 결과에 따라 JetBot에게  전진, 후진, 좌회전, 우회전 등의 명령을 내립니다.  장애물을 감지하면 자동으로 회피합니다 (회피 로직 개선 필요).
*   **상황 음성 설명:**  Edge TTS를 사용하여  분석 결과를 한국어 음성으로 사용자에게 알려줍니다.
*   **웹 인터페이스:**  웹 브라우저를 통해 JetBot을 제어하고, 실시간 영상 및 분석 결과를 확인할 수 있습니다.

## 구성 요소

*   **JetBot:**  NVIDIA Jetson Nano 기반의 로봇 플랫폼.
*   **Ollama:**  로컬 환경에서 대규모 언어 모델(LLM)을 실행하기 위한 도구. (`gemma3:4b` 모델 사용)
*   **FastAPI:**  빠르고 효율적인 Python 웹 API 프레임워크.
*   **Edge TTS:** Microsoft Edge의 텍스트 음성 변환 (TTS) 엔진.
*   **HTML/JavaScript/CSS:** 사용자 인터페이스 (UI).
*  **WebSockets:** JetBot과 PC간의 실시간 통신

## 설치 및 실행 방법

### 1. 사전 준비

*   **JetBot:**  JetBot이 올바르게 설정되어 있고, PC와 **동일한 네트워크**에 연결되어 있어야 합니다. JetBot의 IP 주소를 확인하세요.
*   **Ollama 설치:**  [Ollama 공식 웹사이트](https://ollama.com/)의 지침에 따라 Ollama를 설치합니다.
*   **`gemma3:4b` 모델 다운로드:**  Ollama를 설치한 후, 터미널에서 다음 명령을 실행하여 `gemma3:4b` 모델을 다운로드합니다.
    ```bash
    ollama run gemma3:4b
    ```

### 2.  프로젝트  설치

1.  **코드 다운로드:**  이 저장소(repository)를 클론(clone)하거나 ZIP 파일로 다운로드합니다.

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Python 가상 환경 생성 (선택 사항이지만 권장):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate    # Windows
    ```

3.  **필수 Python 패키지 설치:**

    ```bash
    pip install -r requirements.txt
    ```
    (`requirements.txt` 파일은 저장소에 포함되어 있어야 합니다.  없다면 다음 명령어로 설치하세요.)

    ```bash
      pip install fastapi uvicorn httpx websockets python-multipart edge-tts
    ```

### 3. 환경 설정

*   `main.py` 파일을 열어 다음 설정 변수들을 확인하고 필요에 따라 수정합니다.

    *   `OLLAMA_HOST`: Ollama가 실행 중인 호스트 주소 (기본값: `http://localhost:11434`).
    *   `MODEL_NAME`: 사용할 Ollama 모델 이름 (기본값: `gemma3:4b`).
    *   `JETBOT_WEBSOCKET_URL`: JetBot의 웹소켓 주소 (예: `ws://<jetbot_ip_address>:8766`).  JetBot의 IP 주소와 포트 번호를 확인하여 수정해야 합니다.
    *    `VOICE`: edge-tts에서 사용할 음성.
    *   `TTS_RATE`: 음성 속도.

### 4. 실행

1.  **FastAPI 서버 실행:**

    ```bash
    uvicorn main:app --reload
    ```

2.  **웹 브라우저 열기:**  웹 브라우저에서 `http://localhost:8000`에 접속합니다.

3.  **JetBot 제어:** 웹 페이지의 버튼을 사용하여 JetBot을 제어하고, 실시간 영상 및 분석 결과를 확인합니다.

## 사용 예시
* **"주변 설명"** 버튼을 누르면, JetBot은 현재 카메라에 보이는 환경을 설명하는 음성 응답을 제공합니다.
* **"전진", "후진", "좌회전", "우회전"** 버튼을 누르면 Jetbot이 해당 방향으로 움직입니다.
* JetBot이 장애물이나 특정 객체를 감지하면, 자동으로 회피하거나 사용자에게 알립니다.

## 개선할 점

*   **장애물 회피 로직 개선:** 현재는 단순한 조건("obstacle" 또는 "object" 문자열 포함 여부)으로 장애물을 판단하고,  무조건 왼쪽으로 회피합니다.  LLM의 출력을 더 정교하게 분석하여,  장애물의 위치와 종류에 따라 적절한 회피 방향(좌/우/후진 등)을 결정하도록 개선해야 합니다.
*   **객체 인식 및 추적:** 특정 객체를 인식하고 추적하는 기능을 추가할 수 있습니다.
* **명령어 다양화:** "정지", "속도 조절" 등의 명령어를 추가.
*   **UI 개선:**  사용자 인터페이스를 더욱 직관적이고 편리하게 개선할 수 있습니다.
* **에러 핸들링 강화**: 더 robust한 예외처리를 통해 예기치 않은 상황에서도 안정적으로 작동하도록 개선.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하십시오.
