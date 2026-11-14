@echo off
cd /d "%~dp0"
set YOLO_OFFLINE=true
set YOLO_AUTOINSTALL=false
set HF_HUB_DISABLE_TELEMETRY=1
set DO_NOT_TRACK=1
"%~dp0.venv\Scripts\python.exe" -m cvat_agent.ui
if errorlevel 1 pause
