"""WorldQuant Brain HTTP client: authentication + session quản lý.

Cơ chế:
- POST /authentication bằng HTTP Basic Auth (email + password) ở lần đầu.
- Response 201 kèm session cookie (lưu trong cookie jar của httpx.Client).
- Một số tài khoản cần biometric/persona: 401 kèm header WWW-Authenticate
  chứa link xác thực — log link ra cho user mở thủ công.
- GET/POST tự re-auth một lần khi session hết hạn (401).
"""

from __future__ import annotations

import httpx
from loguru import logger


class AuthError(RuntimeError):
    """Lỗi xác thực WorldQuant Brain."""


class WQBrainClient:
    BASE_URL = "https://api.worldquantbrain.com"

    def __init__(self, email: str, password: str, client: httpx.Client | None = None):
        self.email = email
        self.password = password
        self.client = client or httpx.Client(base_url=self.BASE_URL, timeout=30.0)
        self._authenticated = False

    # ------------------------------------------------------------------ auth
    def authenticate(self) -> None:
        """POST /authentication với Basic Auth; lưu session cookie."""
        resp = self.client.post("/authentication", auth=(self.email, self.password))
        if resp.status_code in (200, 201):
            self._authenticated = True
            logger.success("Đăng nhập WQ Brain thành công")
            return

        if resp.status_code == 401:
            challenge = resp.headers.get("WWW-Authenticate", "")
            if challenge:
                logger.error(
                    "Tài khoản cần xác thực bổ sung (biometric/persona). "
                    "Mở link sau trong trình duyệt để xác thực: {}",
                    challenge,
                )
            raise AuthError(
                "Xác thực thất bại (401). Có thể cần biometric — kiểm tra log."
            )

        raise AuthError(f"Auth thất bại: {resp.status_code} {resp.text}")

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    # --------------------------------------------------------------- requests
    def _ensure_auth(self) -> None:
        if not self._authenticated:
            self.authenticate()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        self._ensure_auth()
        resp = self.client.request(method, path, **kwargs)
        if resp.status_code == 401:
            # Session có thể đã hết hạn — re-auth một lần rồi retry.
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
