@echo off
setlocal enabledelayedexpansion
set ZIP_URL=https://github.com/RESQUELAB/Adaptive-app/releases/download/adaptive_app_v1.0.1/adaptiveapp-v1.0.1.zip
set ZIP_FILE=client_app.zip

echo Downloading Adaptive App client...
curl -L -o %ZIP_FILE% %ZIP_URL%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download client app
    pause
    exit /b
)

if exist client_app (
    echo Removing previous extracted folder...
    rmdir /s /q client_app
)
echo Extracting...
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath . -Force"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to extract %ZIP_FILE%
    pause
    exit /b
)
ren adaptiveapp-v1.0.1 client_app
del /f /q %ZIP_FILE%

:: --- Detect server IP ---
echo Detecting server IP...
set "SERVER_IP="

:: Try to read from .env first
if exist "..\core-environment\.env" (
    for /f "tokens=1,* delims==" %%A in ('findstr "VIDEO_SERVER_HOST=" "..\core-environment\.env"') do (
        set "SERVER_IP=%%B"
    )
)

:: Fallback: auto-detect from ipconfig
if not defined SERVER_IP (
    echo VIDEO_SERVER_HOST not found in .env, detecting from network...
    for /f "tokens=1,* delims=:" %%A in ('ipconfig ^| findstr /R "IPv4.*"') do (
        set "LINE=%%A:%%B"
        echo !LINE! | findstr /C:"IPv4" >nul
        if not errorlevel 1 (
            set "CANDIDATE=%%B"
            set "CANDIDATE=!CANDIDATE: =!"
            if not defined SERVER_IP (
                set "SERVER_IP=!CANDIDATE!"
            )
        )
    )
)

if not defined SERVER_IP (
    echo [ERROR] Could not detect server IP
    pause
    exit /b
)

echo Server IP: %SERVER_IP%

:: --- Write config.json ---
echo Updating config.json -> TARGET_SERVER = %SERVER_IP%
powershell -NoProfile -Command "$json = '{\"TARGET_SERVER\":\"%SERVER_IP%\"}'; $json | Set-Content 'client_app\resources\app\config.json' -Encoding UTF8; Write-Host 'config.json updated'"

echo Done. The AUI client is in the 'client_app' folder.
pause
endlocal
