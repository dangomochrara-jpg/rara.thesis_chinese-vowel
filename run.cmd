@echo off
setlocal
title Vowel Live Compare - One-click Start
cd /d %~dp0

REM Detect Python
where python >nul 2>&1
if %errorlevel% neq 0 (
  where py >nul 2>&1
  if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.10+ and re-run.
    pause
    exit /b 1
  )
  set "PY=py -3"
) else (
  set "PY=python"
)

REM Create venv if missing
if not exist ".venv\Scripts\activate.bat" (
  echo Creating virtual environment .venv ...
  %PY% -m venv .venv
)

REM Activate venv
call .\.venv\Scripts\activate.bat

REM Upgrade pip
%PY% -m pip install --upgrade pip

REM Install dependencies
if exist requirements.txt (
  %PY% -m pip install -r requirements.txt
) else (
  echo requirements.txt not found. Installing default dependencies...
  %PY% -m pip install numpy matplotlib sounddevice librosa scipy praat-parselmouth
)

REM Run app
%PY% vowel_live_compare.py

echo.
echo Press any key to close...
pause >nul
