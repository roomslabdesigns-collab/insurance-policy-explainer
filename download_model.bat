@echo off
REM Downloads the local LLM (Qwen2.5-3B-Instruct, GGUF, Q4_K_M quantization,
REM ~2.1GB) from Hugging Face into .\models\. Free, no account or API key
REM needed. Safe to re-run -- skips the download if the file already exists.
REM
REM This is the default model as of the Phase 12 model comparison: it beat
REM the earlier, lighter Qwen2.5-1.5B on every accuracy metric while keeping
REM a clean 0%% Wrong-but-Confident rate (see docs/evaluation.md) -- at the
REM cost of ~2x response time and ~700MB more RAM. If your machine is tight
REM on RAM or you want faster responses, download the 1.5B model instead:
REM   curl.exe -L -o "models\qwen2.5-1.5b-instruct-q4_k_m.gguf" "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
REM   then set LLM_MODEL_PATH and LLM_MODEL_NAME in a .env file (see .env.example).

if exist "models\qwen2.5-3b-instruct-q4_k_m.gguf" (
    echo Model already present at models\qwen2.5-3b-instruct-q4_k_m.gguf -- skipping download.
    exit /b 0
)

if not exist "models" mkdir models

echo Downloading Qwen2.5-3B-Instruct (Q4_K_M, ~2.1GB)... this may take several minutes.
curl.exe -L -o "models\qwen2.5-3b-instruct-q4_k_m.gguf" "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

if errorlevel 1 (
    echo Download failed. Check your internet connection and try again.
    exit /b 1
)

echo Done. Model saved to models\qwen2.5-3b-instruct-q4_k_m.gguf
