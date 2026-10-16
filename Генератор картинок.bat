@echo off
chcp 65001 >nul
title Generator Kartinok
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" app.py
echo.
echo Server stopped.
pause >nul
