@echo off
REM Downloads the local LLM (Qwen2.5-1.5B-Instruct, GGUF, Q4_K_M quantization,
REM ~1GB) from Hugging Face into .\models\. Free, no account or API key needed.
REM Safe to re-run -- skips the download if the file already exists.

if exist "models\qwen2.5-1.5b-instruct-q4_k_m.gguf" (
    echo Model already present at models\qwen2.5-1.5b-instruct-q4_k_m.gguf -- skipping download.
    exit /b 0
)

if not exist "models" mkdir models

echo Downloading Qwen2.5-1.5B-Instruct (Q4_K_M, ~1GB)... this may take a few minutes.
curl.exe -L -o "models\qwen2.5-1.5b-instruct-q4_k_m.gguf" "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

if errorlevel 1 (
    echo Download failed. Check your internet connection and try again.
    exit /b 1
)

echo Done. Model saved to models\qwen2.5-1.5b-instruct-q4_k_m.gguf
