@echo off
setlocal
cd /d "%~dp0.."
python scripts\reproduce.py %*
if errorlevel 1 exit /b %errorlevel%
