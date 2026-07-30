@echo off
REM ================================================================
REM  setup_docker.bat - Instala Docker Desktop + WSL2 + Ubuntu
REM  para el framework Intelligent-Adaptive-User-Interfaces-Framework
REM
REM  Requiere: Windows 10/11, administrador.
REM  Ejecutar UNA SOLA VEZ en una terminal administrador.
REM ================================================================
setlocal enabledelayedexpansion

echo ============================================
echo  RL4UI - Docker Environment Setup
echo ============================================
echo.
echo Este script instala y configura:
echo   1. WSL2 (Windows Subsystem for Linux)
echo   2. Ubuntu 24.04 LTS
echo   3. Docker Desktop (con backend WSL2)
echo.

REM --- 1. Activar WSL2 ---
echo [1/4] Enabling WSL2...
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart >nul 2>&1
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart >nul 2>&1
echo   Done.

REM --- 2. Instalar kernel de WSL2 ---
echo [2/4] Installing WSL2 kernel...
wsl --set-default-version 2 >nul 2>&1
echo   Done (si falla, descargar manualmente:
echo   https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)

REM --- 3. Instalar Ubuntu ---
echo [3/4] Installing Ubuntu...
wsl --install -d Ubuntu >nul 2>&1
if %errorlevel% neq 0 (
    echo   Ubuntu ya instalado o con errores. Verificar con: wsl -l -v
) else (
    echo   Done. Al finalizar la instalacion se abrira una consola de Ubuntu.
    echo   Complete el usuario/contraseña de Linux cuando se le solicite.
)

REM --- 4. Instalar Docker Desktop ---
echo [4/4] Downloading Docker Desktop...
set "DOCKER_INSTALLER=%TEMP%\Docker Desktop Installer.exe"
curl -L -o "%DOCKER_INSTALLER%" "https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe"
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo descargar Docker Desktop.
    echo   Descargar manualmente de: https://docs.docker.com/desktop/setup/install/windows-install/
    pause
    exit /b
)

echo   Installing Docker Desktop (esto puede tardar varios minutos)...
start /wait "" "%DOCKER_INSTALLER%" install --accept-license --backend=wsl-2
del "%DOCKER_INSTALLER%"

echo.
echo ============================================
echo  Setup completo.
echo ============================================
echo.
echo  PASOS SIGUIENTES:
echo    1. REINICIAR el equipo.
echo    2. Abrir Docker Desktop (dejarlo iniciar unos segundos).
echo    3. Abrir terminal y ir a core-environment\.
echo    4. Ejecutar: run_all.bat
echo.
echo  Si Docker Desktop no reconoce WSL2, ir a:
echo    Settings ^> Resources ^> WSL Integration
echo    y activar la integracion con Ubuntu.
echo ============================================
echo.
pause

endlocal
