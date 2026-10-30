@echo off
title SubAquaVisionUltra - Real-Time YOLOv8 + ByteTrack 30 FPS HUD
color 0B
set PY_EXE="C:\Users\Sujit kumar\AppData\Local\Programs\Python\Python314\python.exe"
if exist %PY_EXE% (
    set TARGET_PY=%PY_EXE%
) else (
    set TARGET_PY=python
)
%TARGET_PY% -m streamlit run app.py --server.port 8503
pause
