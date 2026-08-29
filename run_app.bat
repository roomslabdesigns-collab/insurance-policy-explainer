@echo off
REM ============================================================================
REM Starts the local LLM server (in its own window) and then the Streamlit
REM app. If the LLM server isn't running, the app still starts and shows a
REM clear "server not running" message instead of crashing -- but you need
REM the server running to actually get answers, so this script starts both.
REM ============================================================================

if not exist "models\qwen2.5-1.5b-instruct-q4_k_m.gguf" (
    echo Model not found. Run download_model.bat first.
    exit /b 1
)

echo Starting the local LLM server in a new window (llama_cpp.server on port 8000)...
start "Insurance Policy Explainer - LLM Server" cmd /k ".\venv\Scripts\python.exe -m llama_cpp.server --model models\qwen2.5-1.5b-instruct-q4_k_m.gguf --n_ctx 2048 --host 127.0.0.1 --port 8000"

echo Waiting ~15 seconds for the model to finish loading...
timeout /t 15 /nobreak >nul

echo Starting the Streamlit app (http://localhost:8501)...
.\venv\Scripts\python.exe -m streamlit run app.py
