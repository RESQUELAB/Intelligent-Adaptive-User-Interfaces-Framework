@echo off
setlocal enabledelayedexpansion

REM ============================================
REM  RL4UI - Setup Completo
REM ============================================

REM --- 1. Crear .env desde .env.example si no existe ---
if not exist ".env" (
    echo [1/5] Creating .env from .env.example...
    copy /Y .env.example .env
) else (
    echo [1/5] .env already exists
)

REM --- 2. Auto-detectar IP y anadir VIDEO_SERVER_HOST ---
echo [2/5] Detecting server IP...
call set_ip_env.bat

REM --- 3. Descargar Orchestrator (Electron) ---
echo.
echo [3/5] Checking Orchestrator...
set "ZIP_PATH=.\orchestrator.zip"
set "EXE_PATH=electron_app\adaptiveuiserver.exe"
set "ORCH_URL=https://github.com/RESQUELAB/UIAdaptationManager/releases/download/adaptiveappserver-v1.0.0/orchestrator.zip"

if not exist "electron_app" mkdir electron_app

if not exist "%EXE_PATH%" (
    echo   Downloading orchestrator.zip...
    curl -L -o "%ZIP_PATH%" "%ORCH_URL%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to download orchestrator
        pause
        exit /b
    )
    echo   Extracting...
    powershell -Command "Expand-Archive -Path '%ZIP_PATH%' -DestinationPath '.' -Force"
    del "%ZIP_PATH%"
    echo   Done.
) else (
    echo   Already downloaded.
)

REM --- 4. Copiar .env al Orchestrator ---
copy /Y .env electron_app\.env

REM --- 5. Docker: build y levantar servicios ---
echo.
echo [5/5] Building and starting Docker services...
echo Preparing Docker build folders...

REM Django App
if not exist django_app\source_code\human-feedback-api\manage.py (
    echo   Copying source code to django_app...
    xcopy /E /I /Y rl-teacher-ui-adapt\ django_app\source_code\
)

REM RLHF Server 1
if not exist rlhf_server_1\source_code\human-feedback-api\manage.py (
    echo   Copying source code to rlhf_server_1...
    xcopy /E /I /Y rl-teacher-ui-adapt\ rlhf_server_1\source_code\
)

REM RLHF Server 2
if not exist rlhf_server_2\source_code\human-feedback-api\manage.py (
    echo   Copying source code to rlhf_server_2...
    xcopy /E /I /Y rl-teacher-ui-adapt\ rlhf_server_2\source_code\
)

REM Video Server
if not exist video_server\source_code\app\run_server.py (
    echo   Copying source code to video_server...
    xcopy /E /I /Y rl-teacher-ui-adapt\human-feedback-api\video_server video_server\app\
)

echo   Building docker images...
docker-compose build
echo   Launching services...
docker-compose up -d

echo.
echo Waiting for services to be ready...
timeout /t 10

echo.
echo Starting Orchestrator...
start "" "electron_app\adaptiveuiserver.exe"

echo.
echo ============================================
echo  RL4UI Setup Complete!
echo ============================================
echo  Django:        http://localhost:8000
echo  Video Server:  http://localhost:5000
echo  RLHF Server 1: ws://localhost:9998
echo  RLHF Server 2: ws://localhost:9997
echo  Orchestrator:  Running
echo ============================================
echo.
echo  To setup the client app, run:
echo    cd ..\examples ^&^& get_client.bat
echo ============================================

endlocal
