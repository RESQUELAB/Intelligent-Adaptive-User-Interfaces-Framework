@echo off

call set_ip_env.bat

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