@echo off
REM Launches the analytics dashboard (Phase 11) on a separate port from the
REM main app, so both can run side by side.
.\venv\Scripts\python.exe -m streamlit run app\analytics\dashboard.py --server.port 8502
