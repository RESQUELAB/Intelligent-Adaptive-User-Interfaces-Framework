@echo off
setlocal enabledelayedexpansion

REM ============================================
REM  RL4UI - Setup Completo
REM ============================================

REM --- 1. Crear .env desde .env.example si no existe ---
if not exist ".env" (
    echo [1/7] Creating .env from .env.example...
    copy /Y .env.example .env
) else (
    echo [1/7] .env already exists
)

REM --- 2. Auto-detectar IP y anadir VIDEO_SERVER_HOST ---
echo [2/7] Detecting server IP...
call set_ip_env.bat

REM --- 3. Descargar Orchestrator (Electron) ---
echo.
echo [3/7] Checking Orchestrator...
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

REM --- 4. Descargar clips pre-entrenados (pickle files) ---
set "CLIPS_DIR=rl-teacher-ui-adapt\clips"
set "CLIPS_ZIP=clips.zip"
set "CLIPS_URL=https://github.com/RESQUELAB/Intelligent-Adaptive-User-Interfaces-Framework/releases/download/v1.0.0-clips/clips.zip"

echo.
echo [4/7] Checking pre-trained clips...
if not exist "%CLIPS_DIR%" mkdir "%CLIPS_DIR%"
if exist "%CLIPS_DIR%\UIAdaptation-v0-courses-1.clip" goto :clips_ok
echo   Downloading pre-trained clips (~468MB^)...
curl -L -o "%CLIPS_ZIP%" "%CLIPS_URL%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download clips
    pause
    exit /b
)
echo   Extracting clips...
powershell -Command "Expand-Archive -Path '%CLIPS_ZIP%' -DestinationPath '%CLIPS_DIR%' -Force"
del "%CLIPS_ZIP%"
echo   Done. 40 clips ready.
goto :clips_done
:clips_ok
echo   Pre-trained clips already present.
:clips_done

REM --- 5. Descargar videos pre-generados ---
set "VIDEOS_DIR=videos"
set "VIDEOS_ZIP=clips_videos.zip"
set "VIDEOS_URL=https://github.com/RESQUELAB/Intelligent-Adaptive-User-Interfaces-Framework/releases/download/v1.0.0-clips-videos/clips_videos.zip"

echo.
echo [5/7] Checking pre-generated videos...
if not exist "%VIDEOS_DIR%" mkdir "%VIDEOS_DIR%"
if exist "%VIDEOS_DIR%\clip_creation-courses-1.mp4" goto :videos_ok
echo   Downloading pre-generated videos (~19MB^)...
curl -L -o "%VIDEOS_ZIP%" "%VIDEOS_URL%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download videos
    pause
    exit /b
)
echo   Extracting videos...
powershell -Command "Expand-Archive -Path '%VIDEOS_ZIP%' -DestinationPath '%VIDEOS_DIR%' -Force"
del "%VIDEOS_ZIP%"
echo   Done. 40 videos ready.
goto :videos_done
:videos_ok
echo   Pre-generated videos already present.
:videos_done

REM --- 6. Copiar .env al Orchestrator ---
copy /Y .env electron_app\.env

REM --- 7. Docker: build y levantar servicios ---
echo.
echo [7/7] Building and starting Docker services...
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
