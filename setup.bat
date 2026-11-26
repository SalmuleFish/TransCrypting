@echo off
setlocal EnableDelayedExpansion
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

set "PROFILE="
if /I "%~1"=="--profile" (
    set "PROFILE=%~2"
)

if not defined PROFILE (
    echo Elige que motor de transcripcion de voz quieres instalar:
    echo   [1] Ligero y rapido ^(Recomendado^): ONNX ^(Parakeet v3 + Canary 1B^) — sin CUDA pesado ni PyTorch
    echo   [2] Whisper clasico ^(CPU / NVIDIA CUDA^): faster-whisper + CUDA
    echo   [3] Equilibrado: ONNX + Whisper CUDA ^(Parakeet, Canary y Whisper^)
    echo   [4] Completo: Todos los motores ^(incluye Qwen3-ASR PyTorch CUDA^)
    echo.
    set /p "PROFILE=Selecciona una opcion [1-4] (Enter = 1): "
    if not defined PROFILE set "PROFILE=1"
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [1/4] Creando entorno virtual .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Fallo la creacion de .venv.
        pause
        exit /b 1
    )
) else (
    echo.
    echo [1/4] Entorno virtual .venv detectado.
)

echo [2/4] Instalando base comun en .venv...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements-base.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion base.
    pause
    exit /b 1
)

echo [3/4] Instalando motor de transcripcion ^(perfil %PROFILE%^)...
if "%PROFILE%"=="1" (
    ".venv\Scripts\pip.exe" install "onnx-asr[hub]>=0.12.0"
) else if "%PROFILE%"=="2" (
    ".venv\Scripts\pip.exe" install "faster-whisper>=1.1.0" "av>=11.0.0,<17.0.0" nvidia-cublas-cu12 nvidia-cudnn-cu12
) else if "%PROFILE%"=="3" (
    ".venv\Scripts\pip.exe" install "onnx-asr[hub]>=0.12.0" "faster-whisper>=1.1.0" "av>=11.0.0,<17.0.0" nvidia-cublas-cu12 nvidia-cudnn-cu12
) else if "%PROFILE%"=="4" (
    ".venv\Scripts\pip.exe" install "onnx-asr[hub]>=0.12.0" "faster-whisper>=1.1.0" "av>=11.0.0,<17.0.0" nvidia-cublas-cu12 nvidia-cudnn-cu12 "transformers>=5.13.0" accelerate soundfile
    if not errorlevel 1 (
        ".venv\Scripts\pip.exe" install torch --index-url https://download.pytorch.org/whl/cu128
    )
) else (
    echo [AVISO] Opcion '%PROFILE%' no reconocida; usando perfil 1 ^(ONNX ligero^).
    ".venv\Scripts\pip.exe" install "onnx-asr[hub]>=0.12.0"
)

if errorlevel 1 (
    echo [ERROR] Fallo la instalacion del motor de transcripcion seleccionado.
    pause
    exit /b 1
)

set "CSC_EXE=%SystemRoot%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if exist "%CSC_EXE%" (
    echo [4/4] Compilando lanzador TransCrypting.exe...
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
echo NOTA: Los pesos de los modelos NO se descargan durante la instalacion.
echo Abre Configuracion dentro de la app para descargar solo el modelo que quieras usar.
echo Asegurate de tener Ollama y ffmpeg instalados localmente ^(ej. ollama pull gemma4:e4b^).
pause
