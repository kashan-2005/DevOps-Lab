@echo off
cd /d "%~dp0"
echo === Creating Python environment (.venv) ===
py -3.12 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
echo === Checking the model ===
python backend\test_pipeline.py
python -m pytest -q backend\tests
echo === Installing Angular packages ===
cd frontend
call npm ci
echo.
echo Setup finished. Now run start_backend.bat, then start_frontend.bat
pause
