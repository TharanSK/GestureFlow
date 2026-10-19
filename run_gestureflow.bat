@echo off
title GestureFlow Controller
echo ===================================================================
echo   GestureFlow: Camera-Based OS Gesture Controller
echo ===================================================================
echo.
cd /d "%~dp0"

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=py"
    goto RUN
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=python"
    goto RUN
)

if exist "D:\Users\THARAN S K\AppData\Local\Programs\Python\Python313\python.exe" (
    set "PY_CMD=D:\Users\THARAN S K\AppData\Local\Programs\Python\Python313\python.exe"
    goto RUN
)

echo [ERROR] Python was not found in PATH or standard installation directory!
pause
exit /b 1

:RUN
echo [GestureFlow] Starting GUI Dashboard on your screen...
"%PY_CMD%" src\main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] GestureFlow exited with code %ERRORLEVEL%.
)
echo.
echo GestureFlow has stopped.
pause
