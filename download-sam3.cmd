@echo off
cd /d "%~dp0"
set HF_HUB_DISABLE_TELEMETRY=1
set DO_NOT_TRACK=1
echo First request model access at https://huggingface.co/facebook/sam3
echo Enter your Hugging Face token only in the login prompt below.
"%~dp0.venv\Scripts\python.exe" -m cvat_agent.sam3_login
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -m cvat_agent.sam3_backend --download
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo SAM3 download did not finish. Check login and model access.
pause
exit /b 1
