"""WorldQuant Brain HTTP client: authentication + session + xác thực QR.

Cơ chế:
- POST /authentication bằng HTTP Basic Auth (email + password).
- 200/201 → đăng nhập xong, session cookie lưu trong cookie jar của httpx.Client.
- Tài khoản cần biometric/persona (quét QR): WQ trả 202 hoặc kèm URL xác thực
  trong header/JSON → mở trình duyệt cho user quét QR, chờ Enter rồi thử lại
  (tối đa 3 lần).
- GET/POST tự re-auth một lần khi session hết hạn (401).
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

DEFAULT_SESSION_FILE = Path(".wq_session")


class AuthError(RuntimeError):
    """Lỗi xác thực WorldQuant Brain."""


class WQBrainClient:
    BASE_URL = "https://api.worldquantbrain.com"
    AUTH_TIMEOUT = 30.0
    MAX_VERIFICATION_ATTEMPTS = 3
    VERIFICATION_STATUS_CODES = {202}
    VERIFICATION_HEADER_NAMES = {"location", "x-authentication-url", "x-verification-url"}
    VERIFICATION_JSON_KEYS = {
        "authentication",
        "authenticationurl",
        "inquiryurl",
        "url",
        "verificationurl",
    }

    RATE_LIMIT_STATUS = 429
    MAX_RATE_LIMIT_RETRIES = 5
    DEFAULT_RETRY_AFTER = 5.0

    def __init__(
        self,
        email: str,
        password: str,
        client: httpx.Client | None = None,
        browser_open=None,
        confirmation_input=None,
        sleep_func=None,
        session_file: Path | None = DEFAULT_SESSION_FILE,
    ):
        import time

        self.email = email
        self.password = password
        self.client = client or httpx.Client(base_url=self.BASE_URL, timeout=self.AUTH_TIMEOUT)
        self.browser_open = browser_open or webbrowser.open
        self.confirmation_input = confirmation_input or input
        self._sleep = sleep_func or time.sleep
        self.session_file = session_file
        self._authenticated = False
        self._load_session()

    # ------------------------------------------------------- session on disk
    def _load_session(self) -> None:
        if not self.session_file or not self.session_file.exists():
            return
        try:
            data = json.loads(self.session_file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        # Gắn cookie kèm ĐÚNG domain host. Nếu set không domain (domain=''),
        # khi server trả Set-Cookie (host-only) httpx sẽ giữ CẢ HAI cookie cùng
        # tên rồi gửi cả hai lên — WQ đọc cái cũ trước -> 401. Set kèm domain để
        # cookie mới của server THAY THẾ cookie nạp từ file thay vì nhân đôi.
        domain = urlparse(self.BASE_URL).hostname or ""
        for name, value in data.items():
            self.client.cookies.set(name, value, domain=domain)
        logger.info("Đã nạp session từ {}", self.session_file)

    def _save_session(self) -> None:
        if not self.session_file:
            return
        data = {c.name: c.value for c in self.client.cookies.jar}
        try:
            self.session_file.write_text(json.dumps(data), encoding="utf-8")
            try:
                self.session_file.chmod(0o600)
            except (OSError, NotImplementedError):
                pass  # Windows có thể bỏ qua chmod
        except OSError as exc:
            logger.warning("Không lưu được session: {}", exc)

    def _has_session(self) -> bool:
        return len(self.client.cookies.jar) > 0

    def is_session_valid(self) -> bool:
        """Kiểm tra session cookie còn dùng được qua GET /users/self/."""
        try:
            resp = self.client.get("/users/self/")
        except httpx.RequestError:
            return False
        return resp.status_code == 200

    # ------------------------------------------------------------------ auth
    def _request_authentication(self) -> httpx.Response:
        try:
            return self.client.post("/authentication", auth=(self.email, self.password))
        except httpx.RequestError as exc:
            raise AuthError(
                "Không thể kết nối đến WorldQuant BRAIN. Kiểm tra mạng và thử lại."
            ) from exc

    def _authenticate_with_backoff(self) -> httpx.Response:
        """Gọi /authentication, tự in Retry-After + chờ + thử lại khi gặp 429."""
        for attempt in range(self.MAX_RATE_LIMIT_RETRIES + 1):
            resp = self._request_authentication()
            if resp.status_code == self.RATE_LIMIT_STATUS and attempt < self.MAX_RATE_LIMIT_RETRIES:
                wait = self._retry_after(resp)
                logger.warning(
                    "Đăng nhập bị giới hạn tần suất (429) — chờ {}s rồi thử lại ({}/{})",
                    wait,
                    attempt + 1,
                    self.MAX_RATE_LIMIT_RETRIES,
                )
                print(f"⏳ Bị giới hạn tần suất (429). Chờ {wait:g} giây rồi thử đăng nhập lại...")
                self._sleep(wait)
                continue
            return resp
        return resp

    def authenticate(self, force: bool = False) -> None:
        # Đã đăng nhập trong phiên này -> no-op (kể cả persona/QR không set cookie).
        # Tránh bắt quét QR lại mỗi khi gọi lệnh data trên cùng client.
        if not force and self._authenticated:
            return
        # Tái dùng session đã lưu nếu còn hạn (khỏi đăng nhập lại).
        if not force and self._has_session() and self.is_session_valid():
            self._authenticated = True
            logger.success("Session còn hạn, bỏ qua đăng nhập")
            print("✅ Dùng lại phiên đăng nhập trước.")
            return

        resp = self._authenticate_with_backoff()

        if resp.status_code in (200, 201):
            self._authenticated = True
            self._save_session()
            logger.success("Đăng nhập WQ Brain thành công")
            print("✅ Xác thực thành công!")
            return

        # Cần xác thực bổ sung (quét QR).
        if resp.status_code in self.VERIFICATION_STATUS_CODES or self._extract_verification_url(resp):
            self._complete_additional_verification(resp)
            self._authenticated = True
            self._save_session()
            logger.success("Đăng nhập WQ Brain thành công (sau xác thực QR)")
            print("✅ Xác thực thành công!")
            return

        if resp.status_code in (401, 403):
            raise AuthError(
                "Email hoặc mật khẩu không đúng, hoặc tài khoản không có quyền truy cập."
            )

        raise AuthError(f"Xác thực thất bại: HTTP {resp.status_code}")

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    # ------------------------------------------------------- QR verification
    def _complete_additional_verification(self, response: httpx.Response) -> None:
        verification_url = self._extract_verification_url(response)
        if not verification_url:
            raise AuthError(
                "WorldQuant yêu cầu xác thực bổ sung nhưng không trả về đường dẫn xác thực."
            )

        print(f"🔐 Đường dẫn xác thực (quét QR): {verification_url}")
        try:
            opened = self.browser_open(verification_url)
            if opened is False:
                print("⚠️ Không tự mở được trình duyệt. Hãy mở đường dẫn phía trên thủ công.")
        except Exception:
            print("⚠️ Không tự mở được trình duyệt. Hãy mở đường dẫn phía trên thủ công.")

        for _ in range(self.MAX_VERIFICATION_ATTEMPTS):
            self.confirmation_input(
                "Quét QR và hoàn tất xác thực trong trình duyệt, sau đó nhấn Enter..."
            )
            # Hoàn tất bằng cách POST vào CHÍNH URL persona/inquiry vừa quét —
            # KHÔNG POST /authentication lại (sẽ sinh inquiry mới, lặp vô tận).
            resp = self._post_verification(verification_url)
            if resp.status_code in (200, 201):
                return
            # Có thể WQ trả inquiry mới (vd. inquiry trước hết hạn) — cập nhật URL.
            next_url = self._extract_verification_url(resp)
            if next_url:
                verification_url = next_url
            if (
                resp.status_code in self.VERIFICATION_STATUS_CODES
                or resp.status_code in (401, 403)
                or next_url
            ):
                continue
            raise AuthError(f"Xác thực thất bại: HTTP {resp.status_code}")

        raise AuthError("Xác thực bổ sung chưa hoàn tất sau ba lần kiểm tra.")

    def _post_verification(self, url: str) -> httpx.Response:
        """POST vào URL persona/inquiry (kèm Basic Auth) để hoàn tất xác thực QR."""
        try:
            return self.client.post(url, auth=(self.email, self.password))
        except httpx.RequestError as exc:
            raise AuthError(
                "Không thể kết nối đến WorldQuant BRAIN. Kiểm tra mạng và thử lại."
            ) from exc

    @classmethod
    def _extract_verification_url(cls, response) -> str | None:
        for name, value in getattr(response, "headers", {}).items():
            if name.lower() in cls.VERIFICATION_HEADER_NAMES:
                url = cls._normalize_verification_url(value)
                if url:
                    return url
        try:
            data = response.json()
        except (TypeError, ValueError):
            return None
        return cls._find_verification_url(data)

    @classmethod
    def _find_verification_url(cls, value) -> str | None:
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = key.lower().replace("_", "").replace("-", "")
                if normalized in cls.VERIFICATION_JSON_KEYS:
                    if isinstance(nested, str):
                        url = cls._normalize_verification_url(nested)
                        if url:
                            return url
                    found = cls._find_verification_url(nested)
                    if found:
                        return found
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    found = cls._find_verification_url(nested)
                    if found:
                        return found
        elif isinstance(value, list):
            for nested in value:
                found = cls._find_verification_url(nested)
                if found:
                    return found
        return None

    @classmethod
    def _normalize_verification_url(cls, value) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or not value.startswith(("http://", "https://", "/")):
            return None
        url = urljoin(f"{cls.BASE_URL}/", value)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return url

    # --------------------------------------------------------------- requests
    def _ensure_auth(self) -> None:
        if not self._authenticated:
            self.authenticate()

    def _retry_after(self, resp: httpx.Response) -> float:
        try:
            return float(resp.headers.get("Retry-After", self.DEFAULT_RETRY_AFTER))
        except (TypeError, ValueError):
            return self.DEFAULT_RETRY_AFTER

    def _send_with_rate_limit(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Gửi request, tự chờ Retry-After và thử lại khi gặp 429."""
        for attempt in range(self.MAX_RATE_LIMIT_RETRIES + 1):
            resp = self.client.request(method, path, **kwargs)
            if resp.status_code == self.RATE_LIMIT_STATUS and attempt < self.MAX_RATE_LIMIT_RETRIES:
                wait = self._retry_after(resp)
                logger.warning(
                    "Bị giới hạn tần suất (429) — chờ {}s rồi thử lại ({}/{})",
                    wait,
                    attempt + 1,
                    self.MAX_RATE_LIMIT_RETRIES,
                )
                self._sleep(wait)
                continue
            return resp
        return resp

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        self._ensure_auth()
        resp = self._send_with_rate_limit(method, path, **kwargs)
        if resp.status_code == 401:
            logger.warning("Session hết hạn (401), thử re-authenticate")
            self._authenticated = False
            self.authenticate()
            resp = self._send_with_rate_limit(method, path, **kwargs)
        return resp

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return self._request("POST", path, **kwargs)

    def close(self) -> None:
        self.client.close()
