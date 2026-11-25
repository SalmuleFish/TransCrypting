@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo   TransCrypting - Instalacion Local en Windows
echo ==================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No se encontro Python en el PATH. Instala Python 3.11 o 3.12 primero.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creando entorno virtual .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Fallo la creacion de .venv.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Entorno virtual .venv detectado.
)

echo [2/3] Instalando dependencias en .venv...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    pause
    exit /b 1
)

set "CSC_EXE=%SystemRoot%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if exist "%CSC_EXE%" (
    echo [3/3] Compilando lanzador TransCrypting.exe...
    "%CSC_EXE%" /nologo /target:winexe /optimize+ /win32icon:frontend\icon.ico /reference:System.Windows.Forms.dll /out:TransCrypting.exe launcher\TransCryptingLauncher.cs
    if errorlevel 1 (
        echo [AVISO] No se pudo compilar TransCrypting.exe. Puedes abrir el programa con: .venv\Scripts\python.exe run.py
    ) else (
        echo [OK] TransCrypting.exe creado correctamente.
    )
) else (
    echo [AVISO] No se encontro csc.exe; puedes iniciar TransCrypting con: .venv\Scripts\python.exe run.py
)

echo.
echo Instalacion completada.
echo Asegurate de tener Ollama y ffmpeg instalados localmente y al menos un modelo descargado (ej. ollama pull gemma4:e4b).
pause
