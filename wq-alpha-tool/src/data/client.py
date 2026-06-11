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

import webbrowser
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger


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

    def __init__(
        self,
        email: str,
        password: str,
        client: httpx.Client | None = None,
        browser_open=None,
        confirmation_input=None,
    ):
        self.email = email
        self.password = password
        self.client = client or httpx.Client(base_url=self.BASE_URL, timeout=self.AUTH_TIMEOUT)
        self.client.auth = httpx.BasicAuth(email, password)
        self.browser_open = browser_open or webbrowser.open
        self.confirmation_input = confirmation_input or input
        self._authenticated = False

    # ------------------------------------------------------------------ auth
    def _request_authentication(self) -> httpx.Response:
        try:
            return self.client.post("/authentication")
        except httpx.RequestError as exc:
            raise AuthError(
                "Không thể kết nối đến WorldQuant BRAIN. Kiểm tra mạng và thử lại."
            ) from exc

    def authenticate(self) -> None:
        resp = self._request_authentication()

        if resp.status_code in (200, 201):
            self._authenticated = True
            logger.success("Đăng nhập WQ Brain thành công")
            print("✅ Xác thực thành công!")
            return

        # Cần xác thực bổ sung (quét QR).
        if resp.status_code in self.VERIFICATION_STATUS_CODES or self._extract_verification_url(resp):
            self._complete_additional_verification(resp)
            self._authenticated = True
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
            resp = self._request_authentication()
            if resp.status_code in (200, 201):
                return
            if (
                resp.status_code in self.VERIFICATION_STATUS_CODES
                or resp.status_code in (401, 403)
                or self._extract_verification_url(resp)
            ):
                continue
            raise AuthError(f"Xác thực thất bại: HTTP {resp.status_code}")

        raise AuthError("Xác thực bổ sung chưa hoàn tất sau ba lần kiểm tra.")

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

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        self._ensure_auth()
        resp = self.client.request(method, path, **kwargs)
        if resp.status_code == 401:
            logger.warning("Session hết hạn (401), thử re-authenticate")
            self._authenticated = False
            self.authenticate()
            resp = self.client.request(method, path, **kwargs)
        return resp

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return self._request("POST", path, **kwargs)

    def close(self) -> None:
        self.client.close()
