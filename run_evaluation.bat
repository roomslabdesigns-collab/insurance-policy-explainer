@echo off
REM Runs the end-to-end evaluation (Phase 11) against the golden dataset.
REM Requires the LLM server to already be running -- start it first with
REM run_app.bat, or start it standalone:
REM   .\venv\Scripts\python.exe -m llama_cpp.server --model models\qwen2.5-1.5b-instruct-q4_k_m.gguf --n_ctx 2048 --host 127.0.0.1 --port 8000
REM
REM Takes a few minutes (real local LLM calls for ~40 questions). Results are
REM saved to data\evaluation\e2e_runs\<timestamp>\ and never overwrite a
REM previous run.

.\venv\Scripts\python.exe tests\test_end_to_end_evaluation.py
