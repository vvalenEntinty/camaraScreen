@echo off
cd /d "%~dp0"

echo Verificando dependencias...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Fallo la instalacion. Asegurate de tener Python y pip instalados.
    pause
    exit /b 1
)

echo Iniciando controlador de mano...
python hand_controller.py
if %errorlevel% neq 0 (
    echo.
    echo El programa termino con un error.
    pause
)
