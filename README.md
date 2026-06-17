# WorldQuant Brain Auto-Alpha Tool

Công cụ tự động nghiên cứu, sinh, mô phỏng và nộp Alpha trên nền tảng
WorldQuant BRAIN. Xây dựng theo 4 phase:

1. **Phase 1** — Đăng nhập, lấy data-fields/operators, mô phỏng được một alpha thật.
2. **Phase 2** — Sinh alpha bằng template + Genetic Algorithm, chấm điểm, lọc.
3. **Phase 3** — Sinh alpha có LLM hỗ trợ (DeepSeek, API tương thích Anthropic).
4. **Phase 4** — Submission Manager + Dashboard (Streamlit).

## Cài đặt

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1        # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env             # rồi điền credentials thật
```

## Cấu hình

Sửa `.env` (không commit) — xem `.env.example` cho danh sách biến.

**Đăng nhập:** để trống `WQ_EMAIL`/`WQ_PASSWORD` trong `.env` → tool sẽ hỏi
email/mật khẩu ngay trong PowerShell mỗi lần chạy (mật khẩu ẩn). Nếu tài khoản
cần **xác thực QR/biometric**, tool tự mở trình duyệt tới đường dẫn xác thực;
bạn quét QR xong thì quay lại console nhấn Enter (thử lại tối đa 3 lần).

## Cách dùng nhanh (khuyến nghị)

Double-click **`run.bat`** (hoặc chuột phải `run.ps1` > Run with PowerShell).
Script tự tạo venv, cài dependencies, tạo `.env` lần đầu, rồi mở **wizard chạy
theo từng bước** (một phiên duy nhất giữ đăng nhập):

1. **Đăng nhập** — nhập email/mật khẩu ngay trong console (mật khẩu ẩn); tự
   mở trình duyệt nếu cần quét QR.
2. **Tải data fields** — chỉ mở sau khi đăng nhập. Lần đầu tải về và lưu DB;
   lần sau hỏi *Dùng lại* (khỏi tải) hay *Tải mới*.
3. **Tải operators** — tương tự, cache vào DB.
4. **Mô phỏng / Sinh alpha / GA** — mở dần khi đủ điều kiện bước trước.

Tương đương: `python main.py start`.

## Dùng bằng dòng lệnh

```powershell
python main.py login                          # Đăng nhập (dùng session cũ nếu còn hạn)
python main.py login --force                  # Ép đăng nhập lại
python main.py probe-fields                   # In JSON thật của /data-fields (kiểm format)
python main.py fetch-fields                   # Tải 1 lần -> lưu DB; lần sau load từ DB
python main.py fetch-fields --reload          # Ép tải lại (ghi đè cache)
python main.py cache-status                   # Xem các tổ hợp đã cache
python main.py fetch-operators                # Lấy & cache operators
python main.py simulate --expr "rank(close)"  # Chạy một simulation
python main.py generate --count 100           # Sinh alpha (Phase 2)
python main.py run-ga --population 30 --generations 10
python main.py research --direction "mean-reversion theo thanh khoản" --max-sims 20
python main.py top --n 20                      # Xem alpha tốt nhất
python main.py check-deepseek                  # Test DeepSeek API bằng chat "hello"
python main.py llm-ideas --count 10            # DeepSeek (Phase 3)
python main.py submit --dry-run                # Nộp alpha (Phase 4)
streamlit run dashboard/app.py                 # Dashboard
```

## Test

```powershell
pytest -q
```

Test dùng mock — không gọi WQ Brain thật.
