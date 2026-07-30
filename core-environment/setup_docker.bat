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
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
echo   Done (puede requerir reinicio).

REM --- 2. Configurar WSL2 por defecto ---
echo [2/4] Setting WSL2 as default...
wsl --set-default-version 2
echo.

REM --- 3. Instalar Ubuntu ---
echo [3/4] Installing Ubuntu...
echo   Esto descarga Ubuntu desde Microsoft Store. Puede tardar varios minutos.
echo   Al finalizar, se ABRIRA UNA VENTANA DE UBUNTU para que crees
echo   un usuario y contrasena de Linux. COMPLETA ESE PASO antes de continuar.
echo.
wsl --install -d Ubuntu
echo.
echo   Si ya estaba instalado, verifica con: wsl -l -v

REM --- 4. Instalar Docker Desktop ---
echo [4/4] Installing Docker Desktop via winget...
echo   Esto descargara e instalara Docker Desktop automaticamente.
echo   Puede tardar varios minutos.
echo.
winget install --id Docker.DockerDesktop --accept-source-agreements --accept-package-agreements
if %errorlevel% neq 0 (
    echo [ERROR] La instalacion fallo.
    echo   Instalar manualmente desde: https://docs.docker.com/desktop/setup/install/windows-install/
    pause
    exit /b
)

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
