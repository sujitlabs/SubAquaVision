@echo off
title SubAquaVisionUltra - Real-Time YOLOv8 + ByteTrack 30 FPS HUD
color 0B
echo ================================================================
echo             SubAquaVision ULTRA - 30 FPS Real-Time AI
echo ================================================================
echo.
echo [*] Checking Python environment...
set PY_EXE="C:\Users\Sujit kumar\AppData\Local\Programs\Python\Python314\python.exe"
if exist %PY_EXE% (
    set TARGET_PY=%PY_EXE%
) else (
    set TARGET_PY=python
)
echo [*] Starting SubAquaVisionUltra Streamlit Server on Port 8503...
echo [*] The app will open in your default browser at http://localhost:8503
echo.
%TARGET_PY% -m streamlit run app.py --server.port 8503
echo.
echo Server stopped.
pause
