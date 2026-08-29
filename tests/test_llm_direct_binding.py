"""
Phase 1 verification script.

Loads the local GGUF model directly via llama-cpp-python (no server needed for
this smoke test) and runs one prompt. If this prints a coherent answer without
crashing or swapping to disk for minutes, your environment is ready for Phase 2.
"""

from pathlib import Path
from llama_cpp import Llama

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Did the download finish?"
        )

    print(f"Loading model from {MODEL_PATH} ...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,       # keep context small for now — plenty for this test, easy on RAM
        n_threads=4,      # adjust to your CPU core count
        verbose=False,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. Answer in one short sentence.",
        },
        {
            "role": "user",
            "content": "In one sentence, what is the capital of France?",
        },
    ]

    print("Generating...")
    result = llm.create_chat_completion(messages=messages, max_tokens=64, temperature=0.1)
    answer = result["choices"][0]["message"]["content"]

    print("\n--- MODEL OUTPUT ---")
    print(answer.strip())
    print("--------------------")
    print("\nPhase 1 environment check: PASS")

if __name__ == "__main__":
    main()
