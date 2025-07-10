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

:: --- Modify config.json with value from .env ---
echo Updating config.json with VIDEO_SERVER_HOST from .env...
powershell -NoProfile -Command ^
  "$envPath = '.env';" ^
  "$envMap = @{};" ^
  "Get-Content $envPath | ForEach-Object {" ^
  "  if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.+)$') {" ^
  "    $key = $matches[1].Trim(); $val = $matches[2].Trim(); $envMap[$key] = $val" ^
  "  }" ^
  "};" ^
  "$target = $envMap['VIDEO_SERVER_HOST'];" ^
  "if (-not $target) { Write-Host '[ERROR] VIDEO_SERVER_HOST not found in .env'; exit 1 }" ^
  "$configPath = 'client_app\\resources\\app\\config.json';" ^
  "$json = Get-Content $configPath -Raw | ConvertFrom-Json;" ^
  "$json.TARGET_SERVER = $target;" ^
  "$json | ConvertTo-Json -Depth 10 | Set-Content $configPath -Encoding UTF8;" ^
  "Write-Host ' Updated config.json -> TARGET_SERVER = ' $target"

if %errorlevel% neq 0 (
    echo [ERROR] Failed to update config.json
    pause
    exit /b
)
echo Done. The client is in the 'client_app' folder.
pause
endlocal