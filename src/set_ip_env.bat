@echo off
setlocal enabledelayedexpansion

set ENV_FILE=.env

findstr /B "VIDEO_SERVER_HOST=" "%ENV_FILE%" >nul
if %errorlevel%==0 (
    echo VIDEO_SERVER_HOST ya está definido en %ENV_FILE%. No se modifica.
    goto :EOF
)

echo Buscando IP de adaptadores de red físicos...

set IP=

for /f "tokens=1,* delims=:" %%A in ('ipconfig ^| findstr /R "IPv4.*"') do (
    set line=%%A:%%B
    echo !line! | findstr /C:"IPv4" >nul
    if not errorlevel 1 (
        set ip_candidate=%%B
        set ip_candidate=!ip_candidate: =!
        set IP=!ip_candidate!
        goto :write_env
    )
)

echo No se encontró una IP válida.
goto :EOF

:write_env
echo IP detectada: !IP!
echo VIDEO_SERVER_HOST=!IP!>>"%ENV_FILE%"
echo VIDEO_SERVER_HOST agregado a %ENV_FILE%.

