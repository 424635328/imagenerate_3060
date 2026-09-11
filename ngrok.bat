@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Clear proxy variables only in this child process; some free tunnel clients
rem refuse to run behind an HTTP proxy (ngrok: ERR_NGROK_9009).
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=

if "%PORT%"=="" set "PORT=8001"

rem IMPORTANT: never call a bare "ngrok" here.  cmd resolves executables from the
rem current directory first, and this directory contains ngrok.bat, so a bare
rem "ngrok" would re-enter this same script forever
rem ("Maximum setlocal recursion level reached").  Resolve the real ngrok.exe.
set "NGROK_EXE="
for /f "delims=" %%I in ('where ngrok.exe 2^>nul') do if not defined NGROK_EXE set "NGROK_EXE=%%I"

if not defined NGROK_EXE (
  echo [ERROR] ngrok.exe was not found in PATH.
  echo         Install it, or set NGROK_EXE to its full path before running this script.
  pause
  exit /b 1
)

echo Using tunnel client: %NGROK_EXE%
echo Forwarding http://localhost:%PORT% ...
"%NGROK_EXE%" http %PORT%
if errorlevel 1 pause
