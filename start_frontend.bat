@echo off
cd /d "%~dp0frontend"
echo Angular on http://localhost:4200  - keep this window open
call npx ng serve --open
pause
