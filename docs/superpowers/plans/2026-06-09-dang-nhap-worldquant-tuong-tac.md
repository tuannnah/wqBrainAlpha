# Kế Hoạch Triển Khai Đăng Nhập WorldQuant Tương Tác

> **Dành cho agent thực thi:** BẮT BUỘC dùng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực hiện lần lượt từng task. Các bước dùng checkbox `- [ ]` để theo dõi.

**Mục tiêu:** Thay việc đọc `brain_credentials.txt` bằng nhập email/mật khẩu trong console, đồng thời mở trình duyệt để người dùng hoàn tất xác thực QR khi WorldQuant yêu cầu.

**Kiến trúc:** `main.py` chỉ chịu trách nhiệm nhập và kiểm tra dữ liệu tài khoản. `BrainBatchAlpha` nhận trực tiếp email/mật khẩu, quản lý `requests.Session`, phân loại phản hồi xác thực và điều phối bước xác thực bổ sung. Các phụ thuộc bên ngoài như session, mở trình duyệt và chờ người dùng được truyền vào để test không gọi API thật.

**Công nghệ:** Python 3.12, `requests`, `HTTPBasicAuth`, `getpass`, `webbrowser`, `unittest`, PyInstaller.

---

### Task 1: Nhập tài khoản trong console

**File:**
- Sửa: `main.py`
- Tạo: `tests/test_interactive_login.py`

- [x] **Bước 1: Viết test đỏ cho nhập tài khoản**

```python
class ConsoleCredentialsTest(unittest.TestCase):
    def test_yeu_cau_nhap_lai_khi_email_hoac_mat_khau_rong(self):
        input_mock = Mock(side_effect=["", "user@example.com"])
        password_mock = Mock(side_effect=["", "secret"])

        email, password = prompt_credentials(input_func=input_mock, password_func=password_mock)

        self.assertEqual((email, password), ("user@example.com", "secret"))
```

- [x] **Bước 2: Chạy test và xác nhận thất bại do chưa có hàm**

Chạy: `python -m unittest tests.test_interactive_login.ConsoleCredentialsTest -v`

Kỳ vọng: `FAIL` hoặc `ImportError` vì `prompt_credentials` chưa tồn tại.

- [x] **Bước 3: Cài đặt tối thiểu**

```python
def prompt_credentials(input_func=input, password_func=getpass):
    while True:
        email = input_func("\nEmail WorldQuant BRAIN: ").strip()
        password = password_func("Mật khẩu: ")
        if email and password:
            return email, password
        print("❌ Email và mật khẩu không được để trống")
```

Trong `main()`, gọi hàm này sau khi chế độ chạy hợp lệ và truyền kết quả vào:

```python
email, password = prompt_credentials()
brain = BrainBatchAlpha(email, password)
```

- [x] **Bước 4: Chạy test và xác nhận qua**

Chạy: `python -m unittest tests.test_interactive_login.ConsoleCredentialsTest -v`

Kỳ vọng: `OK`.

### Task 2: Xác thực trực tiếp và phân loại lỗi

**File:**
- Sửa: `brain_batch_alpha.py`
- Sửa: `tests/test_interactive_login.py`

- [x] **Bước 1: Viết test đỏ cho xác thực thành công, timeout và sai tài khoản**

Test dùng fake session có `post()` trả response kiểm soát được. Kiểm tra:

```python
client = BrainBatchAlpha("user@example.com", "secret", session=session)
self.assertEqual(session.auth.username, "user@example.com")
self.assertEqual(session.auth.password, "secret")
session.post.assert_called_once_with(
    "https://api.worldquantbrain.com/authentication",
    timeout=30,
)
```

Với HTTP `401`, constructor phải phát sinh `AuthenticationError` có thông báo chung, không chứa email hoặc mật khẩu. Với `requests.RequestException`, lỗi phải được chuyển thành thông báo kết nối.

- [x] **Bước 2: Chạy test và xác nhận thất bại đúng nguyên nhân**

Chạy: `python -m unittest tests.test_interactive_login.BrainAuthenticationTest -v`

Kỳ vọng: `FAIL` vì constructor và exception mới chưa tồn tại.

- [x] **Bước 3: Cài đặt xác thực cơ bản**

```python
class AuthenticationError(RuntimeError):
    pass


class BrainBatchAlpha:
    AUTH_TIMEOUT_SECONDS = 30

    def __init__(self, email, password, session=None, browser_open=None, confirmation_input=None):
        self.session = session or requests.Session()
        self.browser_open = browser_open or webbrowser.open
        self.confirmation_input = confirmation_input or input
        self.session.auth = HTTPBasicAuth(email, password)
        self._setup_authentication()
```

`_setup_authentication()` gọi endpoint với timeout, chấp nhận `200/201`, chuyển `401/403` thành lỗi tài khoản và chuyển lỗi mạng thành `AuthenticationError` không tiết lộ dữ liệu nhạy cảm.

- [x] **Bước 4: Chạy test và xác nhận qua**

Chạy: `python -m unittest tests.test_interactive_login.BrainAuthenticationTest -v`

Kỳ vọng: `OK`.

### Task 3: Xử lý xác thực QR trong trình duyệt

**File:**
- Sửa: `brain_batch_alpha.py`
- Sửa: `tests/test_interactive_login.py`

- [x] **Bước 1: Viết test đỏ cho URL challenge**

Bao phủ URL trong `Location`, `authentication`, `verificationUrl`, `verification_url`, `url` và object lồng trong `challenge`. Kiểm tra URL tương đối được ghép với `API_BASE_URL`.

```python
self.assertEqual(
    BrainBatchAlpha._extract_verification_url(response),
    "https://platform.worldquantbrain.com/verify",
)
```

- [x] **Bước 2: Viết test đỏ cho luồng mở trình duyệt**

Fake session trả challenge trước, sau đó trả HTTP `201`. Test kiểm tra URL được mở một lần, URL vẫn được in làm fallback, hàm chờ Enter được gọi, và request xác thực được gửi lại.

- [x] **Bước 3: Chạy test và xác nhận thất bại**

Chạy: `python -m unittest tests.test_interactive_login.BrainVerificationTest -v`

Kỳ vọng: `FAIL` vì chưa có xử lý challenge.

- [x] **Bước 4: Cài đặt luồng challenge tối thiểu**

```python
MAX_VERIFICATION_ATTEMPTS = 3
VERIFICATION_STATUS_CODES = {202}

def _complete_additional_verification(self, response):
    verification_url = self._extract_verification_url(response)
    if not verification_url:
        raise AuthenticationError(
            "WorldQuant yêu cầu xác thực bổ sung nhưng không trả về đường dẫn xác thực."
        )

    print(f"Đường dẫn xác thực: {verification_url}")
    try:
        self.browser_open(verification_url)
    except Exception:
        print("⚠️ Không thể tự mở trình duyệt. Hãy mở đường dẫn phía trên thủ công.")

    for _ in range(self.MAX_VERIFICATION_ATTEMPTS):
        self.confirmation_input(
            "Quét QR và hoàn tất xác thực trong trình duyệt, sau đó nhấn Enter..."
        )
        response = self._request_authentication()
        if self._is_authenticated(response):
            return
    raise AuthenticationError("Xác thực bổ sung chưa hoàn tất.")
```

Phân loại challenge dựa trên status `202` hoặc sự hiện diện của URL xác thực trong response. Không in response body.

- [x] **Bước 5: Chạy toàn bộ test đăng nhập**

Chạy: `python -m unittest tests.test_interactive_login -v`

Kỳ vọng: tất cả test qua.

### Task 4: Loại bỏ file credentials khỏi đóng gói

**File:**
- Sửa: `build.py`
- Sửa: `build_windows.py`
- Sửa: `create_zipapp.py`
- Sửa: `tests/test_windows_only_structure.py`

- [x] **Bước 1: Viết test đỏ cho cấu trúc đóng gói**

```python
def test_build_scripts_khong_dung_file_credentials(self):
    for relative_path in ("build.py", "build_windows.py", "create_zipapp.py"):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        self.assertNotIn("brain_credentials.txt", text, relative_path)
```

- [x] **Bước 2: Chạy test và xác nhận thất bại**

Chạy: `python -m unittest tests.test_windows_only_structure.WindowsOnlyStructureTest.test_build_scripts_khong_dung_file_credentials -v`

Kỳ vọng: `FAIL` vì ba script còn tham chiếu credentials.

- [x] **Bước 3: Xóa xử lý credentials**

`build.py` và `build_windows.py` chỉ sao chép hoặc tạo `alpha_ids.txt`. `create_zipapp.py` chỉ giữ:

```python
config_files = ["alpha_ids.txt"]
```

Giữ `brain_credentials.txt` trong `.gitignore` để tránh vô tình commit file cũ; không tự động xóa file cục bộ hoặc trong `dist`.

- [x] **Bước 4: Chạy test và xác nhận qua**

Chạy: `python -m unittest tests.test_windows_only_structure -v`

Kỳ vọng: `OK`.

### Task 5: Cập nhật tài liệu tiếng Việt

**File:**
- Sửa: `README.md`

- [x] **Bước 1: Viết test đỏ cho hướng dẫn đăng nhập mới**

Mở rộng test cấu trúc để README không còn hướng dẫn tạo credentials và có các cụm `getpass`/mật khẩu ẩn, xác thực QR, trình duyệt.

- [x] **Bước 2: Chạy test và xác nhận thất bại**

Chạy: `python -m unittest tests.test_windows_only_structure -v`

Kỳ vọng: `FAIL` vì README còn hướng dẫn file credentials.

- [x] **Bước 3: Sửa README**

Mô tả:

- chương trình hỏi email và mật khẩu mỗi lần chạy;
- mật khẩu không hiển thị và không được lưu;
- khi có QR, trình duyệt mặc định được mở;
- người dùng hoàn tất xác thực rồi quay lại console nhấn Enter;
- file credentials cũ không còn được dùng.

- [x] **Bước 4: Chạy test và xác nhận qua**

Chạy: `python -m unittest tests.test_windows_only_structure -v`

Kỳ vọng: `OK`.

### Task 6: Kiểm chứng toàn bộ

**File:**
- Kiểm tra: toàn bộ file đã sửa

- [x] **Bước 1: Chạy toàn bộ test**

Chạy: `python -m unittest discover -s tests -v`

Kỳ vọng: tất cả test qua, không có lỗi hoặc warning ngoài dự kiến.

- [x] **Bước 2: Kiểm tra dependency**

Chạy: `python -m pip check`

Kỳ vọng: `No broken requirements found.`

- [x] **Bước 3: Kiểm tra cú pháp**

Chạy: `python -m compileall -q main.py brain_batch_alpha.py tests`

Kỳ vọng: exit code `0`.

- [x] **Bước 4: Rà diff**

Chạy: `git diff --check`

Kỳ vọng: không có lỗi whitespace. Xác nhận không có email/mật khẩu thật trong diff và không có thay đổi ngoài phạm vi.
