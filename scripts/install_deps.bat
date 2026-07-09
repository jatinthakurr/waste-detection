@echo off
REM Use the running Python to install requirements
python -m pip install --upgrade pip
python -m pip install -r "%~dp0\..\requirements.txt"
if %ERRORLEVEL% neq 0 (
  echo Installation failed. Try running the command above manually.
) else (
  echo Dependencies installed successfully.
)
pause
