# WorldQuant Brain Alpha Generator

Tool Python CLI **nghiên cứu Alpha tự động** cho WorldQuant Brain: đồng bộ
metadata tài khoản, dùng DeepSeek sinh ý tưởng và biểu thức Alpha, mô phỏng,
đánh giá và đưa Alpha đạt chuẩn vào hàng chờ duyệt thủ công. Tool **không tự
submit** Alpha.

## Tổng Quan Quy Trình

1. Nhập email và mật khẩu WorldQuant BRAIN trong console.
2. Tạo một snapshot metadata mới (kéo toàn bộ dataset/field/operator từ BRAIN)
   hoặc chọn snapshot cũ.
3. Engine tự chọn dữ liệu, tạo ý tưởng và biểu thức Alpha bằng DeepSeek.
4. Mỗi Alpha được kiểm tra cục bộ trước khi gửi mô phỏng.
5. Đánh giá kết quả; nếu Alpha đủ tiềm năng sẽ tạo biến thể có mục tiêu.
6. Alpha đạt chuẩn được lưu vào hàng chờ với trạng thái `PENDING_REVIEW`.
7. Lượt chạy dừng khi đủ 10 Alpha mới đạt chuẩn hoặc khi người dùng gõ `quit`.

## Cấu Hình DeepSeek

Tool đọc API key **chỉ** từ biến môi trường `DEEPSEEK_API_KEY`. Thiếu biến này
là lỗi cấu hình và engine sẽ không bắt đầu.

Đặt key trên Windows (PowerShell):

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "your-key", "User")
```

Mở lại terminal để biến môi trường có hiệu lực. Model và các giới hạn nghiên
cứu nằm trong `research_config.json` và có thể chỉnh mà không sửa code.

## Cài Đặt Trên Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chạy từ source:

```powershell
.\.venv\Scripts\python.exe main.py
```

Build file `.exe`:

```powershell
.\.venv\Scripts\python.exe build_windows.py
```

File build nằm trong `dist/Alpha_Tool.exe`.

## Đăng Nhập WorldQuant BRAIN

Mỗi lần chạy, chương trình yêu cầu nhập:

- Email WorldQuant BRAIN.
- Mật khẩu. Mật khẩu không hiển thị trong console khi nhập (dùng `getpass`).

Email và mật khẩu không được lưu xuống ổ đĩa.

Nếu WorldQuant yêu cầu xác thực QR:

1. Chương trình tự mở trang xác thực trong trình duyệt mặc định.
2. Quét mã QR và hoàn tất bước xác thực.
3. Quay lại console và nhấn Enter để chương trình kiểm tra đăng nhập.

Nếu trình duyệt không tự mở, chương trình vẫn in đường dẫn để mở thủ công.

## Menu Chính

Sau khi đăng nhập, chương trình hiển thị:

1. **Tạo Metadata DB mới**: đồng bộ toàn bộ dataset, data field, operator và
   scope mà tài khoản có quyền vào một snapshot SQLite có nhãn.
2. **Chọn Metadata DB cũ**: liệt kê các snapshot `READY` thuộc đúng email đang
   đăng nhập để tái sử dụng.
3. **Xem Alpha chờ duyệt**: in các Alpha `PENDING_REVIEW` kèm WorldQuant Alpha
   ID và biểu thức, không chạy nghiên cứu.

Mỗi email có nhiều Metadata DB nhưng chỉ một Research DB; dữ liệu được phân
vùng theo email (đường dẫn dùng mã băm, không chứa email thô).

## Lượt Nghiên Cứu

- Người dùng **không** phải chọn dataset, field hay nhập ý tưởng — engine tự
  làm dựa trên metadata đã đồng bộ.
- Mỗi ý tưởng có tối đa 3 lô, mỗi lô tối đa 5 Alpha gốc với hypothesis khác
  nhau.
- Chỉ khi có Alpha qua quality gate, engine mới tạo tối đa 5 biến thể có mục
  tiêu cho mỗi Alpha cha (không tạo biến thể của biến thể).
- Lượt chạy kết thúc khi đạt mục tiêu 10 Alpha mới đạt chuẩn hoặc khi gõ
  `quit` rồi Enter (dừng an toàn: hoàn tất việc đang chạy rồi lưu lại).

## Vị Trí Dữ Liệu

Toàn bộ DB và log runtime nằm trong thư mục dữ liệu người dùng:

```text
%LOCALAPPDATA%\WorldQuantBrainAlpha\
└── accounts\<account_hash>\
    ├── metadata\<snapshot_id>.sqlite
    ├── research.sqlite
    └── logs\<run_id>.log
```

Các thư mục này không được đóng gói vào executable và không được commit vào Git.

## Kiểm Tra Alpha Thủ Công

Alpha đạt chuẩn vào hàng chờ `PENDING_REVIEW`. Tool **không tự submit**; người
dùng tự xem lại và quyết định submit trên nền tảng WorldQuant.

## Kiểm Thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

## Lưu Ý Bảo Mật

- Mật khẩu, API key và Authorization header không bao giờ được ghi vào log
  hoặc DB.
- Raw response của DeepSeek và WorldQuant chỉ được lưu sau khi lọc thông tin
  nhạy cảm.

## Giấy Phép

MIT License
