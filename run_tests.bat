@echo off
REM ============================================================================
REM Runs every verification script in tests\, in build order, and reports a
REM summary at the end. This project uses standalone, narrated verification
REM scripts rather than pytest (each one explains what it's checking and why
REM as it runs) -- see README.md "Testing" section for why.
REM
REM Scripts marked [LLM] need the local LLM server running first (run_app.bat,
REM or start it standalone -- see run_evaluation.bat for the command).
REM ============================================================================

setlocal enabledelayedexpansion
set PY=.\venv\Scripts\python.exe
set FAIL_COUNT=0

echo ===== tests\test_pdf_extraction.py =====
%PY% tests\test_pdf_extraction.py || set /a FAIL_COUNT+=1

echo.
echo ===== tests\test_clause_chunking.py =====
%PY% tests\test_clause_chunking.py || set /a FAIL_COUNT+=1

echo.
echo ===== tests\test_semantic_search.py =====
%PY% tests\test_semantic_search.py || set /a FAIL_COUNT+=1

echo.
echo ===== tests\test_retrieval_evaluation.py =====
%PY% tests\test_retrieval_evaluation.py || set /a FAIL_COUNT+=1

echo.
echo ===== tests\test_llm_direct_binding.py [LLM not required - loads model directly] =====
%PY% tests\test_llm_direct_binding.py || set /a FAIL_COUNT+=1

echo.
echo ===== tests\test_llm_integration.py [LLM server required] =====
%PY% tests\test_llm_integration.py || set /a FAIL_COUNT+=1

echo.
echo ===== tests\test_structured_answers.py [LLM server required] =====
%PY% tests\test_structured_answers.py || set /a FAIL_COUNT+=1

echo.
echo ===== tests\test_guardrails.py [LLM server required for live checks] =====
%PY% tests\test_guardrails.py || set /a FAIL_COUNT+=1

echo.
echo ===== tests\test_streamlit_app.py [LLM server required for live checks] =====
%PY% tests\test_streamlit_app.py || set /a FAIL_COUNT+=1

echo.
echo ===== tests\test_evidence_highlighting.py =====
%PY% tests\test_evidence_highlighting.py || set /a FAIL_COUNT+=1

echo.
echo ============================================================
if %FAIL_COUNT%==0 (
    echo ALL TEST SCRIPTS PASSED
) else (
    echo %FAIL_COUNT% SCRIPT(S) FAILED -- see output above for details
)
echo (Full end-to-end evaluation against the golden dataset is separate: run_evaluation.bat)
endlocal
