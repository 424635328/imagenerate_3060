@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title LANDSCAPE-ART - Landscape Image Generator
echo Starting LANDSCAPE-ART app...
if "%PYTHON%"=="" set "PYTHON=python"
start "LANDSCAPE-ART" cmd /k ""%PYTHON%" "%CD%\app.py""
timeout /t 10 >nul
start "" http://127.0.0.1:7860
