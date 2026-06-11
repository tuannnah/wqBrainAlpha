# WorldQuant Brain Auto-Alpha Tool

Công cụ tự động nghiên cứu, sinh, mô phỏng và nộp Alpha trên nền tảng
WorldQuant BRAIN. Xây dựng theo 4 phase:

1. **Phase 1** — Đăng nhập, lấy data-fields/operators, mô phỏng được một alpha thật.
2. **Phase 2** — Sinh alpha bằng template + Genetic Algorithm, chấm điểm, lọc.
3. **Phase 3** — Sinh alpha có LLM hỗ trợ (DeepSeek, API tương thích OpenAI).
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

## Sử dụng (Phase 1)

```powershell
python main.py login                          # Kiểm tra đăng nhập
python main.py fetch-fields                   # Lấy & cache data fields
python main.py fetch-operators                # Lấy & cache operators
python main.py simulate --expr "rank(close)"  # Chạy một simulation
```

## Test

```powershell
pytest -q
```

Test dùng mock — không gọi WQ Brain thật.
