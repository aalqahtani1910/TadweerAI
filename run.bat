@echo off
REM ──────────────────────────────────────────────────────────────────────────────
REM TadweerAI launcher – installs deps, runs Streamlit unbuffered, shows logs.
REM ──────────────────────────────────────────────────────────────────────────────

REM 1) Ensure UTF-8 output so Arabic logs display correctly
chcp 65001 >nul

REM 2) (Re)install dependencies – comment out after first successful install if you like
echo Installing Python packages…
pip install -r requirements.txt
if errorlevel 1 (
  echo [Error] Failed to install dependencies.
  pause
  exit /b 1
)

REM 3) Launch Streamlit via Python in unbuffered mode
echo.
echo Starting TadweerAI with full console output...
python -u -m streamlit run app.py --server.headless true --server.address 127.0.0.1

REM 4) Keep window open after Streamlit exits (so you can read logs or errors)
echo.
echo [Press any key to close this window]
pause >nul
