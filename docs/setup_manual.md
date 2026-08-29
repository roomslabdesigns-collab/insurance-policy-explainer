# Manual Setup (what the `.bat` scripts actually do)

For non-Windows users, or anyone who wants to run each step themselves and see exactly what's happening.

## Prerequisites

- Python 3.11+ (tested on 3.13)
- ~2GB free disk space (mostly the local model)
- No GPU required, no paid API, no account of any kind

## 1. Virtual environment and dependencies

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip

# CPU-only PyTorch -- smaller than the default GPU-bundled wheel, and this
# project never uses a GPU, so there's nothing to gain from the larger one.
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

.\venv\Scripts\python.exe -m pip install -r requirements.txt

# llama-cpp-python with server support, from a prebuilt-wheel index. Installing
# it from plain PyPI can trigger a from-source build requiring CMake and
# Visual Studio Build Tools -- this avoids that entirely on Windows.
.\venv\Scripts\python.exe -m pip install "llama-cpp-python[server]==0.3.35" --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

(On macOS/Linux: drop the CPU-index line for torch if you have a GPU you want to use, and `llama-cpp-python[server]` will typically build from source automatically via pip — this needs a C++ compiler present, which is standard on those platforms.)

## 2. Download the local LLM

Qwen2.5-1.5B-Instruct, GGUF format, Q4_K_M quantization (~1GB). Free, public, no login needed.

```powershell
curl.exe -L -o "models\qwen2.5-1.5b-instruct-q4_k_m.gguf" "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
```

## 3. Start the local LLM server

```powershell
.\venv\Scripts\python.exe -m llama_cpp.server --model models\qwen2.5-1.5b-instruct-q4_k_m.gguf --n_ctx 2048 --host 127.0.0.1 --port 8000
```

Leave this running in its own terminal. Verify it's up:

```powershell
curl.exe http://127.0.0.1:8000/v1/models
```

## 4. Start the app

In a second terminal:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501.

## Why two processes?

The LLM is served by its own long-lived process (`llama_cpp.server`), and Streamlit talks to it over local HTTP — the same architecture as talking to any LLM API, just entirely on your own machine. This means the ~1.5-2GB the model occupies in RAM is loaded exactly once, regardless of how many times Streamlit reruns its script (which happens on every click) or how many questions get asked.

## Memory footprint (8GB RAM target)

| Component | Approx. RAM |
|---|---|
| LLM server (model + KV cache) | ~1.8-2.1GB |
| Streamlit app | ~50-90MB |
| Embedding model (loaded on first question, cached after) | ~300-400MB |

Comfortably within an 8GB machine alongside the OS and a browser.
