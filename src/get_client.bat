@echo off
setlocal
set ZIP_URL=https://github.com/RESQUELAB/Adaptive-app/releases/download/adaptive_app_v1.0.0/adaptiveapp-v1.0.0.zip
set ZIP_FILE=client_app.zip

echo Downloading Adaptive App client...
curl -L -o %ZIP_FILE% %ZIP_URL%
if exist client_app (
    echo Removing previous extracted folder...
    rmdir /s /q client_app
)
echo Extracting...
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath ."
if %errorlevel% neq 0 (
    echo [ERROR] Failed to extract %ZIP_FILE%
    pause
    exit /b
)
ren adaptiveapp-v1.0.0 client_app
del /f /q %ZIP_FILE%
echo Done. The client is in the 'adaptiveapp-v1.0.0' folder.
endlocal