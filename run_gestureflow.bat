@echo off
title GestureFlow Controller
echo ===================================================================
echo   GestureFlow: Camera-Based OS Gesture Controller
echo ===================================================================
echo.
echo Starting GestureFlow on your primary display...
echo.
cd /d "%~dp0"
py src\main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] GestureFlow exited with error code %ERRORLEVEL%.
    pause
)
