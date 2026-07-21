@echo off
setlocal

set ZIP_URL=https://github.com/RESQUELAB/Adaptive-app/releases/download/adaptive_app_v1.0.1/adaptiveapp-v1.0.1.zip
set ZIP_FILE=client_app.zip

echo [1/3] Downloading Adaptive App client...
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

echo [2/3] Reading VIDEO_SERVER_HOST from .env...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$envFile = Get-Content '..\core-environment\.env' -Raw;" ^
  "if ($envFile -match 'VIDEO_SERVER_HOST=(.+)') {" ^
  "  $ip = $matches[1].Trim();" ^
  "  Write-Host \"  Server IP: $ip\";" ^
  "  $json = '{\"TARGET_SERVER\":\"' + $ip + '\"}';" ^
  "  [System.IO.File]::WriteAllText((Resolve-Path 'client_app\resources\app\config.json').Path, $json);" ^
  "  Write-Host '  config.json written successfully'" ^
  "} else {" ^
  "  Write-Host '  [ERROR] VIDEO_SERVER_HOST not found in .env'; exit 1" ^
  "}"

echo [3/3] Done. Client app is in 'client_app' folder.
pause
endlocal
