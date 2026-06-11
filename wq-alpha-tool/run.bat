@echo off
REM Double-click de chay tool (tu bypass execution policy cho rieng phien nay)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
pause
