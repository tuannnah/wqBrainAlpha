@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Dang nhap WorldQuant BRAIN ===
echo.
echo Khi hien link QR: mo link, quet/approve tren dien thoai,
echo roi quay lai cua so nay va nhan ENTER.
echo.
venv\Scripts\python main.py login
echo.
pause
