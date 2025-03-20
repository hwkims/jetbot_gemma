# JetBot Control with Gemma 3 and FastAPI

This project allows you to control a JetBot using voice commands and text input, leveraging the power of Google's Gemma 3 language model for natural language understanding and image description. The system uses FastAPI for the backend server, websockets for real-time communication with the JetBot, and `edge-tts` for text-to-speech.

## Project Structure

The project consists of three main components:

*   **`main.py` (FastAPI Server):** This is the core backend application, running on your development machine (not the JetBot). It handles:
    *   Serving the web interface (`index.html`).
    *   Receiving user commands (text and voice).
    *   Communicating with the Ollama server (where Gemma 3 runs).
    *   Communicating with the JetBot via websockets.
    *   Generating Text-to-Speech (TTS) audio.
    *   Managing a simple memory of past interactions.

*   **`jetbot_server.py` (JetBot Server):** This script runs on the JetBot itself. It handles:
    *   Connecting to the FastAPI server via websockets.
    *   Receiving commands (forward, backward, left, right, stop, dance, describe).
    *   Controlling the JetBot's motors based on the received commands.
    *   Capturing images from the JetBot's camera.
    *   Sending captured images back to the FastAPI server.

*   **`static/index.html` (Web Interface):** This is the frontend, providing a user interface to:
    *   Display the live camera feed from the JetBot.
    *   Provide buttons for common commands.
    *   Accept text input for custom commands.
    *   Use voice input for commands.
    *   Display responses from the Gemma 3 model.
    *   Play Text-to-Speech audio.
    * Contains `ding.mp3` file, that gives a sound whenever a button is clicked.

## Prerequisites

*   **JetBot:** A fully assembled and configured JetBot with the `jetbot` package installed.
*   **Ollama:** Ollama installed and running, with the `gemma3:4b` model downloaded.  You can use a different Gemma model, but you'll need to update the `MODEL_NAME` variable in `main.py`.  Ollama can run on the same machine as the FastAPI server or on a separate machine.
*   **Python 3.6+:**  Python 3.6 or higher is required on both your development machine (for `main.py`) and the JetBot (for `jetbot_server.py`).  The provided `jetbot_server.py` is compatible with Python 3.6.
*   **Network Connectivity:** Your development machine and the JetBot must be on the same network and able to communicate with each other.
* **edge-tts:** Make sure that the correct voice is installed. You can use `edge-tts --list-voices` command to check available voices.

## Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/hwkims/jetbot_gemma.git
    cd jetbot_gemma
    ```

2.  **Install Dependencies (Development Machine):**
    ```bash
    pip install fastapi uvicorn httpx websockets edge-tts python-multipart pydantic
    ```

3.  **Install Dependencies (JetBot):**
    ```bash
    pip install websockets  # You probably already have jetbot
    ```

4.  **Create `static` Directory:** Create a directory named `static` in the same directory as `main.py`.

5.  **Place Files:**
    *   Put your `index.html`, `ding.mp3`, and any other static files (CSS, JavaScript) into the `static` directory.
    * Copy `jetbot_server.py` to your Jetbot. The recommended method is to upload it into the Jetbot's jupyter notebook.

6.  **Configuration (`main.py`):**
    *   **`OLLAMA_HOST`:** Set this to the address of your Ollama server (e.g., `http://localhost:11434` if running locally).
    *   **`MODEL_NAME`:**  Set this to the correct Gemma 3 model name (e.g., `gemma3:4b`).
    *   **`JETBOT_WEBSOCKET_URL`:**  Set this to the JetBot's IP address and the WebSocket port (e.g., `ws://192.168.1.100:8766`).  You'll need to find your JetBot's IP address (see Troubleshooting).
    *   **`TTS_VOICE`:** Set appropriate TTS voice.

7.  **Configuration (`jetbot_server.py`):**
    *   **`WEBSOCKET_PORT`:**  Make sure this matches the port in `JETBOT_WEBSOCKET_URL` in `main.py`.  The default (8766) should work unless you have a port conflict.

## Running the Application

1.  **Start JetBot Server (on JetBot):**
    *   On your JetBot, open a terminal or a Jupyter Notebook cell and run:
        ```bash
        python jetbot_server.py
        ```
        You should see a message like "WebSocket server running on port 8766".  This script must be running *continuously* for the system to work.

2.  **Start FastAPI Server (on Development Machine):**
    *   Open a terminal on your development machine and navigate to the project directory (where `main.py` is located).
    *   Run:
        ```bash
        uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        ```
        The `--reload` option is useful for development (it automatically restarts the server when you make changes to `main.py`), but you can remove it for production use.

3.  **Access the Web Interface:**
    *   Open a web browser on your development machine and go to `http://localhost:8000`.

## Usage

*   **Basic Controls:** Use the buttons in the web interface to move the JetBot (forward, backward, left, right, stop, dance).
*   **Describe:** Click the "Describe" button to get an image description from Gemma 3.
*   **Custom Commands:** Type commands in the text input field and click "Execute" (e.g., "move forward for 3 seconds", "turn left 90 degrees").
*   **Voice Commands:** Click the "Voice Input" button and speak your commands (in Korean).  Requires a microphone and browser support for speech recognition.
* **Observe:**  The image from the Jetbot's camera, and TTS responses from Gemma 3 are displayed on the web interface.

## Troubleshooting

*   **`timed out during opening handshake` or Image Not Appearing:** This almost always indicates a problem with the WebSocket connection between the FastAPI server and the JetBot server.
    *   **JetBot Server Running?**  Make absolutely sure `jetbot_server.py` is running on the JetBot *before* you start the FastAPI server or try to use the web interface.  Check for any error messages when starting `jetbot_server.py`.
    *   **Correct IP Address?** Verify the JetBot's IP address using `ifconfig` on the JetBot and update `JETBOT_WEBSOCKET_URL` in `main.py` accordingly.
    *   **Network Connectivity?**  Can your development machine `ping` the JetBot's IP address?  Are both devices on the same network?
    *   **Firewall?** Make sure no firewalls are blocking the connection (ports 8000 and 8766).
    * **Port Conflict?** Check if some other process is blocking the port 8766 on your Jetbot.

*   **`304 Not Modified` (Image not updating):** This indicates browser caching. The provided code includes `Cache-Control` headers to prevent this, but you can also try:
    *   Hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R).
    *   Clear your browser's cache.
    *   Try a different browser.

*   **Gemma 3 Not Responding or Giving Incorrect Descriptions:**
    *   **Ollama Running?**  Make sure the Ollama server is running and the specified model (`gemma3:4b` or your chosen model) is downloaded.
    *   **Prompt Engineering:**  The prompts in `main.py` are carefully crafted.  If you modify them, be very precise and test thoroughly.  Small changes in wording can have a large impact.  Refer to the Ollama and Gemma documentation for best practices.
    *   **Model Limitations:** Gemma 3, like all LLMs, can sometimes hallucinate or give unexpected responses.

*   **TTS Not Working:**
    *   **`edge-tts` Installed?**  Make sure `edge-tts` is installed correctly.
    *   **Correct Voice?**  Verify that `TTS_VOICE` in `main.py` is set to a valid voice name. Use `edge-tts --list-voices` to see available voices.
    *  **Audio Output:** Check your system's audio output settings.

* **JetBot Not Moving:**
    * **`jetbot` package installed?** Make sure that the `jetbot` package is properly installed on your Jetbot.
    * **Robot properly initialized?** Check if there were errors during the robot initialization.

## Code Overview

*   **`main.py`:**
    *   **FastAPI Setup:**  Creates the FastAPI application, sets up CORS, and mounts the static file directory.
    *   **Endpoints:**
        *   `/`: Serves the `index.html` page.
        *   `/api/generate`:  Handles the main logic for processing commands, interacting with Gemma 3 and the JetBot, and generating responses.
    *   **`OllamaRequest` (Pydantic Model):** Defines the structure of the request body for `/api/generate`.
    *   **`JETBOT_COMMANDS`:**  A dictionary mapping command names to JetBot actions, parameters, and TTS responses.
    *   **`generate_tts()`:**  Generates TTS audio from text using `edge-tts`.
    *   **`query_gemma3()`:**  Sends prompts and images to the Ollama server.
    *   **`send_command_to_jetbot()`:** Sends commands to the JetBot via websockets.
    *   **`load_memory()` and `save_memory()`:**  Handles loading and saving a simple conversation history.

*   **`jetbot_server.py`:**
    *   **WebSocket Server:**  Uses the `websockets` library to create a WebSocket server.
    *   **`handle_command()`:**  Executes JetBot commands based on received messages.
    *   **`capture_image()`:**  Captures images from the JetBot's camera.
    *   **`websocket_handler()`:**  Handles incoming WebSocket connections and messages, including reconnection logic.
    *   **`main()`:**  Starts the WebSocket server and keeps it running, with error handling for address-in-use situations.

* **`static/index.html`**
    *  **HTML Structure:** Sets up basic page structure (title, image display area, buttons, input fields, etc.).
    * **CSS Styles:** Basic styles.
    * **JavaScript Logic:** Handles user interactions.

## Contributing

Pull requests are welcome!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
