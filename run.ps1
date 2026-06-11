# Launcher cho WorldQuant Brain Auto-Alpha Tool — chạy wizard theo từng bước.
# Chạy: chuột phải > Run with PowerShell, hoặc:  .\run.ps1

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot

$Py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

# Tạo venv + cài dependencies nếu chưa có (lần đầu hơi lâu).
if (-not (Test-Path $Py)) {
    Write-Host "Chưa có venv — đang tạo và cài dependencies..." -ForegroundColor Yellow
    python -m venv venv
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -r requirements.txt
    Write-Host "Đã cài xong môi trường." -ForegroundColor Green
}

# Tạo .env từ mẫu nếu chưa có (để trống credentials -> wizard sẽ hỏi khi đăng nhập).
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Đã tạo .env (credentials để trống -> nhập trực tiếp khi đăng nhập)." -ForegroundColor Yellow
    Write-Host "Nếu dùng DeepSeek (LLM), mở .env điền DEEPSEEK_API_KEY." -ForegroundColor DarkGray
}

# Chạy wizard từng bước (login -> fetch fields -> ...).
& $Py "main.py" start

