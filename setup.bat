@echo off
REM ============================================================================
REM One-time environment setup. Safe to re-run.
REM
REM What this does, step by step (nothing hidden):
REM   1. Creates a Python virtual environment in .\venv
REM   2. Installs PyTorch (CPU-only build -- smaller and sufficient; no GPU needed)
REM   3. Installs the rest of requirements.txt
REM   4. Installs llama-cpp-python with server support from a prebuilt-wheel
REM      index (avoids needing CMake/Visual Studio Build Tools on Windows)
REM
REM After this finishes, run download_model.bat, then run_app.bat.
REM ============================================================================

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create the virtual environment. Is Python installed and on PATH?
    exit /b 1
)

echo [2/4] Installing PyTorch (CPU build)...
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo [3/4] Installing project dependencies...
.\venv\Scripts\python.exe -m pip install -r requirements.txt

echo [4/4] Installing llama-cpp-python with server support (prebuilt wheel)...
.\venv\Scripts\python.exe -m pip install "llama-cpp-python[server]==0.3.35" --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

echo.
echo Setup complete. Next steps:
echo   1. download_model.bat   (downloads the local LLM, ~1GB, one time)
echo   2. run_app.bat          (starts the LLM server + the Streamlit app)
