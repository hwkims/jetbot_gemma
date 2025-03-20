# Jetbot Autonomous Control with Ollama and FastAPI

This project implements a web-based interface for controlling a Jetbot robot, leveraging the power of Ollama's large language and vision models (LLMs/LMMs) for autonomous navigation and interaction. It uses FastAPI for the backend, WebSockets for real-time communication, and edge-tts for text-to-speech.

## Features

*   **Autonomous Navigation:** The Jetbot can navigate autonomously, avoiding obstacles, following paths, and responding to its environment.
*   **Real-time Video Stream:** A live video feed from the Jetbot's camera is displayed in the web interface.
*   **Manual Control:**  The Jetbot can be controlled manually using buttons for forward, backward, left, right, and stop.
*   **Voice Control (STT):**  Issue commands to the Jetbot using your voice (Korean speech recognition).
*   **Text-to-Speech (TTS):** The Jetbot responds with spoken messages using edge-tts.
*   **Custom Commands:** Send arbitrary text commands to the Jetbot.
*   **Ollama Integration:** Uses Ollama to process images and determine the Jetbot's actions.
*   **Robust Connection:**  Handles WebSocket disconnections and reconnections gracefully.
*   **Python 3.6 Compatibility (Jetbot):** The Jetbot server code is compatible with Python 3.6.
*   **OpenCV Optional (Jetbot):** Uses OpenCV for faster image encoding if available, falls back to `bgr8_to_jpeg` otherwise.

## Prerequisites

1.  **Jetbot:** A fully assembled and configured Jetbot robot.  This project assumes you're using the standard Jetbot image and have the `jetbot` Python library installed.
2.  **Ollama:** Ollama must be installed and running on a machine accessible to the FastAPI server.  Download and install a suitable vision model (e.g., `llava:7b`):
    ```bash
    ollama pull llava:7b
    ```
3.  **Python:**
    *   **FastAPI Server:** Python 3.7+ is recommended for the FastAPI server.
    *   **Jetbot:** Python 3.6 is required on the Jetbot.
4.  **Python Packages:** Install the required Python packages (see "Installation" below).
5.  **Network:** The Jetbot and the machine running the FastAPI server must be on the same local network.

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create `static` Directory:**

    ```bash
     mkdir static
    ```
3.  **Create Files:** Create the following files with the provided code:
    *   `fastapi_server.py` (FastAPI server code)
    *   `jetbot_server.py` (Jetbot server code)
    *   `static/index.html` (HTML/JavaScript for the web interface)
    *    `static/ding.mp3` (a short sound file for notification; you can use any `.mp3`)

4.  **Install Python Packages (FastAPI Server):**

    It's highly recommended to use a virtual environment:

    ```bash
    python3 -m venv venv  # Or python3.7 -m venv venv, etc.
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

    Then install the packages:

    ```bash
    pip install fastapi uvicorn httpx websockets pydantic edge-tts speechrecognition python-multipart
    ```

5.  **Install Python Packages (Jetbot):**

    On the Jetbot (using SSH or a terminal on the Jetbot itself):

    ```bash
    pip install websockets  #  The jetbot image *should* already have the jetbot package and opencv.
    pip install opencv-python # Should already be there, but just in case.
    ```
    If `pip` gives you trouble on the Jetbot (due to Python 3.6), try `pip3 install websockets opencv-python`.

## Configuration

1.  **`fastapi_server.py`:**
    *   `OLLAMA_HOST`:  Set this to the correct address of your Ollama server (default: `http://localhost:11434`).
    *   `MODEL_NAME`:  Make sure this matches the name of the Ollama model you downloaded (e.g., `llava:7b`).
    *   `JETBOT_WEBSOCKET_URL`:  Set this to the IP address of your Jetbot and the correct port (default: `ws://192.168.137.233:8766`).  You'll need to find your Jetbot's IP address (e.g., using `ifconfig` on the Jetbot).
    *   `TTS_VOICE`: Choose a text-to-speech voice.  You can use `edge-tts --list-voices` to see available voices.  `ko-KR-HyunsuNeural` is a good Korean voice; `en-US-JennyNeural` is a good English voice.

2.  **`jetbot_server.py`:**
    *   `WEBSOCKET_PORT`: This should match the port used in `JETBOT_WEBSOCKET_URL` on the FastAPI server (default: `8766`).

## Running the Project

1.  **Start the Jetbot Server (on the Jetbot):**

    ```bash
    python jetbot_server.py
    ```

2.  **Start the FastAPI Server (on your computer):**

    ```bash
    python fastapi_server.py
    ```

3.  **Access the Web Interface:**

    Open a web browser and go to `http://localhost:8000` (or the IP address of your FastAPI server if it's not on the same machine as your browser).

## Usage

*   **Manual Control:** Use the "Forward," "Backward," "Left," "Right," and "Stop" buttons to control the Jetbot directly.
*   **Autonomous Control:** Enter the desired number of iterations and click "Autonomous."
*   **Describe:** Click "Describe" to get a spoken description of the Jetbot's surroundings.
*   **Voice Control:** Click "Voice Control" and speak a command (in Korean).  The recognized text will appear in the "Custom Command" input field.
*   **Custom Command:** Type a command into the "Custom Command" input field and click "Execute."
*   **Iterations:** Enter the desired number of iterations to run.
*   **Webcam Feed:** The live video feed from the Jetbot's camera will be displayed.
* **Ding Sound:** Ding sound will be play whenever the buttons are pressed.

## Troubleshooting

*   **Jetbot not connecting:**
    *   Verify the Jetbot's IP address in `fastapi_server.py`.
    *   Make sure the Jetbot server (`jetbot_server.py`) is running on the Jetbot.
    *   Check that the Jetbot and FastAPI server are on the same network.
    *   Check for firewall issues.
*   **Ollama not responding:**
    *   Ensure Ollama is installed and running.
    *   Verify the `OLLAMA_HOST` and `MODEL_NAME` settings in `fastapi_server.py`.
    *   Check the Ollama logs for errors.
*   **No image displayed:**
    *   Make sure the Jetbot server is running.
    *   Check the browser's developer console for any JavaScript errors.
    *   Verify that the Jetbot's camera is working correctly.
*  **STT not working**:
    *   Make sure you have granted microphone permissions to your web page.
    * Check if `speech_recognition` is installed properly.
* **TTS not working**:
    * Check if `edge-tts` is installed correctly.
    * Check if the chosen voice is valid.
    * Make sure that you have audio output enabled.
* **Error message in the UI** Check the logs of both the Jetbot server and FastAPI server, along with the browser console, for more detailed error.

## Code

(See below for the final, corrected code for all three files.)
