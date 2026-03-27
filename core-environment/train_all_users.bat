@echo off
REM Load .env variables into environment
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    set "%%A=%%B"
)

REM Set experiment name
set EXPERIMENT_NAME=cso

REM Call Docker with env vars
docker exec -e POSTGRES_DB=%POSTGRES_DB% ^
             -e POSTGRES_USER=%POSTGRES_USER% ^
             -e POSTGRES_PASSWORD=%POSTGRES_PASSWORD% ^
             -e POSTGRES_HOST=%POSTGRES_HOST% ^
             -e POSTGRES_PORT=%POSTGRES_PORT% ^
             -it src-django_app-1 python /app/rl_teacher/train_all_users.py -n %EXPERIMENT_NAME%

pause