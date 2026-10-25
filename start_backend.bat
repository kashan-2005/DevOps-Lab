@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Flask API on http://127.0.0.1:5000  - keep this window open
python backend\app.py
pause
