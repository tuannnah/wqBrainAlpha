# Launcher menu cho WorldQuant Brain Auto-Alpha Tool
# Chạy: chuột phải > Run with PowerShell, hoặc:  .\run.ps1

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot

$Py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

function Initialize-Env {
    # Tạo venv + cài dependencies nếu chưa có.
    if (-not (Test-Path $Py)) {
        Write-Host "Chưa có venv — đang tạo và cài dependencies (lần đầu, hơi lâu)..." -ForegroundColor Yellow
        python -m venv venv
        & $Py -m pip install --upgrade pip
        & $Py -m pip install -r requirements.txt
        Write-Host "Đã cài xong môi trường." -ForegroundColor Green
    }
    # Tạo .env từ template nếu chưa có.
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Host "Đã tạo .env từ mẫu — vui lòng điền WQ_EMAIL/WQ_PASSWORD (và DEEPSEEK_API_KEY nếu dùng LLM)." -ForegroundColor Yellow
        $open = Read-Host "Mở .env để sửa ngay? (y/n)"
        if ($open -eq "y") { notepad ".env" }
    }
}

function Invoke-Tool {
    param([string[]]$ToolArgs)
    Write-Host ">> python main.py $($ToolArgs -join ' ')" -ForegroundColor DarkGray
    & $Py "main.py" @ToolArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Lệnh kết thúc với lỗi (exit $LASTEXITCODE). Xem logs\ để biết chi tiết." -ForegroundColor Red
    }
}

function Show-Menu {
    Write-Host ""
    Write-Host "=== WorldQuant Brain Auto-Alpha Tool ===" -ForegroundColor Cyan
    Write-Host " 1) Đăng nhập (login)"
    Write-Host " 2) Tải data fields (fetch-fields)"
    Write-Host " 3) Tải operators (fetch-operators)"
    Write-Host " 4) Mô phỏng một biểu thức (simulate)"
    Write-Host " 5) Sinh alpha bằng template (generate)"
    Write-Host " 6) Chạy Genetic Algorithm (run-ga)"
    Write-Host " 7) Xem alpha tốt nhất (top)"
    Write-Host " 8) DeepSeek brainstorm ý tưởng (llm-ideas)"
    Write-Host " 9) DeepSeek sinh alpha từ ý tưởng (llm-generate)"
    Write-Host "10) Nộp alpha (submit)"
    Write-Host "11) Mở Dashboard (streamlit)"
    Write-Host " 0) Thoát"
    Write-Host ""
}

Initialize-Env

while ($true) {
    Show-Menu
    $choice = Read-Host "Chọn"
    switch ($choice) {
        "1" { Invoke-Tool @("login") }
        "2" {
            $region = Read-Host "Region [USA]"; if (-not $region) { $region = "USA" }
            $universe = Read-Host "Universe [TOP3000]"; if (-not $universe) { $universe = "TOP3000" }
            Invoke-Tool @("fetch-fields", "--region", $region, "--universe", $universe)
        }
        "3" { Invoke-Tool @("fetch-operators") }
        "4" {
            $expr = Read-Host "Biểu thức (vd: rank(close))"
            if ($expr) { Invoke-Tool @("simulate", "--expr", $expr) }
        }
        "5" {
            $count = Read-Host "Số lượng [100]"; if (-not $count) { $count = "100" }
            Invoke-Tool @("generate", "--count", $count)
        }
        "6" {
            $pop = Read-Host "Population [30]"; if (-not $pop) { $pop = "30" }
            $gen = Read-Host "Generations [10]"; if (-not $gen) { $gen = "10" }
            $useLlm = Read-Host "Dùng seed từ DeepSeek? (y/n)"
            $gaArgs = @("run-ga", "--population", $pop, "--generations", $gen)
            if ($useLlm -eq "y") { $gaArgs += "--seed-llm" }
            Invoke-Tool $gaArgs
        }
        "7" {
            $n = Read-Host "Số dòng [20]"; if (-not $n) { $n = "20" }
            Invoke-Tool @("top", "--n", $n, "--sort", "score")
        }
        "8" {
            $n = Read-Host "Số ý tưởng [10]"; if (-not $n) { $n = "10" }
            Invoke-Tool @("llm-ideas", "--count", $n)
        }
        "9" {
            $idea = Read-Host "Ý tưởng (vd: momentum ngắn hạn kết hợp volume)"
            if ($idea) {
                $count = Read-Host "Số alpha [5]"; if (-not $count) { $count = "5" }
                Invoke-Tool @("llm-generate", "--idea", $idea, "--count", $count)
            }
        }
        "10" {
            $real = Read-Host "Nộp THẬT? Gõ 'yes' để nộp, Enter để chỉ xem (dry-run)"
            if ($real -eq "yes") { Invoke-Tool @("submit", "--no-dry-run") }
            else { Invoke-Tool @("submit", "--dry-run") }
        }
        "11" {
            Write-Host "Mở dashboard tại http://localhost:8501 — nhấn Ctrl+C để dừng." -ForegroundColor Green
            & $Py -m streamlit run "dashboard\app.py"
        }
        "0" { break }
        default { Write-Host "Lựa chọn không hợp lệ." -ForegroundColor Red }
    }
}

Write-Host "Tạm biệt!" -ForegroundColor Cyan

