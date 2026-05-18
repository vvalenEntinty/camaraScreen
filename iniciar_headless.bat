@echo off
cd /d "%~dp0"

echo Verificando dependencias...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ERROR: Fallo la instalacion de dependencias.
    pause
    exit /b 1
)

echo Iniciando en segundo plano...
start "" pythonw "%~dp0hand_controller.py" --headless
