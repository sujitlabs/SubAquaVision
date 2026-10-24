@echo off
title SubAquaVision - Underwater Object Detection
color 0B
echo ================================================================
echo               SubAquaVision Underwater AI System                
echo ================================================================
echo.
echo [*] Checking Python environment...
set PY_EXE="C:\Users\Sujit kumar\AppData\Local\Programs\Python\Python314\python.exe"
if exist %PY_EXE% (
    set TARGET_PY=%PY_EXE%
) else (
    set TARGET_PY=python
)
echo [*] Starting Streamlit server...
echo [*] The app will open in your default browser automatically.
echo.
%TARGET_PY% -m streamlit run app.py
echo.
echo Server stopped.
pause
