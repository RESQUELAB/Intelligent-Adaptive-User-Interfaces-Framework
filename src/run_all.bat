@echo off

call set_ip_env.bat

echo Paths and download info
set "ZIP_PATH=./orchestrator.zip"
set "EXE_PATH=electron_app\adaptiveuiserver.exe"
set "DOWNLOAD_URL=https://github.com/RESQUELAB/UIAdaptationManager/releases/download/adaptiveappserver-v1.0.0/orchestrator.zip"

echo Create electron_app folder if it doesn't exist
if not exist "electron_app" (
    mkdir electron_app
)

echo Check if executable exists, download and extract if not
if not exist "%EXE_PATH%" (
    echo [INFO] Executable not found. Downloading orchestrator.zip...
    curl -L -o "%ZIP_PATH%" "%DOWNLOAD_URL%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to download orchestrator.zip
        pause
        exit /b
    )
    echo [INFO] Extracting orchestrator.zip...
    powershell -Command "Expand-Archive -Path '%ZIP_PATH%' -DestinationPath '.' -Force"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to extract orchestrator.zip
        pause
        exit /b
    )
    del "%ZIP_PATH%"
    echo [INFO] Extraction complete.
)

echo Copying .env to Electron app folder...
copy /Y .env electron_app\.env

echo Building docker image...
docker-compose build
echo Launching backend services...
docker-compose up -d

echo Waiting for services to be ready...
timeout /t 5

echo Starting Electron app...
start "" electron_app\adaptiveuiserver.exe