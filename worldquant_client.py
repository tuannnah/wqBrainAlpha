"""Client WorldQuant BRAIN: xác thực, metadata API và simulation có cấu trúc."""

import json
import webbrowser
from time import sleep
from urllib.parse import urljoin, urlparse

import requests
from requests.auth import HTTPBasicAuth

from research_models import Scope, SimulationResult


class AuthenticationError(RuntimeError):
    """Lỗi đăng nhập WorldQuant BRAIN."""


class WorldQuantApiError(RuntimeError):
    """Lỗi HTTP khi gọi WorldQuant API (không phải rate limit)."""

    def __init__(self, path, status_code):
        super().__init__(f"WorldQuant API lỗi {status_code} tại {path}")
        self.path = path
        self.status_code = status_code


class WorldQuantRateLimitError(RuntimeError):
    """WorldQuant trả về 429; cần chờ Retry-After giây."""

    def __init__(self, retry_after_seconds):
        super().__init__(f"Bị giới hạn tần suất, chờ {retry_after_seconds} giây")
        self.retry_after_seconds = retry_after_seconds


class WorldQuantClient:
    API_BASE_URL = "https://api.worldquantbrain.com"
    AUTH_TIMEOUT_SECONDS = 30
    REQUEST_TIMEOUT_SECONDS = 30
    MAX_VERIFICATION_ATTEMPTS = 3
    VERIFICATION_STATUS_CODES = {202}
    VERIFICATION_HEADER_NAMES = {
        "location",
        "x-authentication-url",
        "x-verification-url",
    }
    VERIFICATION_JSON_KEYS = {
        "authentication",
        "authenticationurl",
        "inquiryurl",
        "url",
        "verificationurl",
    }
    DEFAULT_POLL_TIMEOUT_SECONDS = 900

    def __init__(
        self,
        email,
        password,
        session=None,
        browser_open=None,
        confirmation_input=None,
        sleep_func=sleep,
        poll_timeout_seconds=None,
    ):
        self.session = session or requests.Session()
        self.session.auth = HTTPBasicAuth(email, password)
        self.browser_open = browser_open or webbrowser.open
        self.confirmation_input = confirmation_input or input
        self.sleep = sleep_func
        self.poll_timeout_seconds = (
            poll_timeout_seconds or self.DEFAULT_POLL_TIMEOUT_SECONDS
        )
        self._setup_authentication()

    # -- Authentication ----------------------------------------------------

    def _request_authentication(self):
        try:
            return self.session.post(
                f"{self.API_BASE_URL}/authentication",
                timeout=self.AUTH_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise AuthenticationError(
                "Không thể kết nối đến WorldQuant BRAIN. "
                "Hãy kiểm tra kết nối mạng và thử lại."
            ) from exc

    def _setup_authentication(self):
        response = self._request_authentication()
        if response.status_code in [200, 201]:
            print("✅ Xác thực thành công!")
            return

        if (
            response.status_code in self.VERIFICATION_STATUS_CODES
            or self._extract_verification_url(response)
        ):
            self._complete_additional_verification(response)
            print("✅ Xác thực thành công!")
            return

        if response.status_code in [401, 403]:
            raise AuthenticationError(
                "Email hoặc mật khẩu không đúng, hoặc tài khoản không có quyền truy cập."
            )

        raise AuthenticationError(
            f"Xác thực thất bại: HTTP {response.status_code}."
        )

    @classmethod
    def _extract_verification_url(cls, response):
        for name, value in getattr(response, "headers", {}).items():
            if name.lower() in cls.VERIFICATION_HEADER_NAMES:
                verification_url = cls._normalize_verification_url(value)
                if verification_url:
                    return verification_url

        try:
            response_data = response.json()
        except (TypeError, ValueError):
            return None

        return cls._find_verification_url(response_data)

    @classmethod
    def _find_verification_url(cls, value):
        if isinstance(value, dict):
            for key, nested_value in value.items():
                normalized_key = key.lower().replace("_", "").replace("-", "")
                if normalized_key not in cls.VERIFICATION_JSON_KEYS:
                    continue

                if isinstance(nested_value, str):
                    verification_url = cls._normalize_verification_url(nested_value)
                    if verification_url:
                        return verification_url

                verification_url = cls._find_verification_url(nested_value)
                if verification_url:
                    return verification_url

            for nested_value in value.values():
                if isinstance(nested_value, (dict, list)):
                    verification_url = cls._find_verification_url(nested_value)
                    if verification_url:
                        return verification_url

        if isinstance(value, list):
            for nested_value in value:
                verification_url = cls._find_verification_url(nested_value)
                if verification_url:
                    return verification_url

        return None

    @classmethod
    def _normalize_verification_url(cls, value):
        if not isinstance(value, str):
            return None

        value = value.strip()
        if not value or not (value.startswith(("http://", "https://", "/"))):
            return None

        verification_url = urljoin(f"{cls.API_BASE_URL}/", value)
        parsed_url = urlparse(verification_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            return None

        return verification_url

    def _complete_additional_verification(self, response):
        verification_url = self._extract_verification_url(response)
        if not verification_url:
            raise AuthenticationError(
                "WorldQuant yêu cầu xác thực bổ sung nhưng không trả về "
                "đường dẫn xác thực."
            )

        print(f"🔐 Đường dẫn xác thực: {verification_url}")
        try:
            browser_opened = self.browser_open(verification_url)
            if browser_opened is False:
                print(
                    "⚠️ Không thể tự mở trình duyệt. "
                    "Hãy mở đường dẫn phía trên thủ công."
                )
        except Exception:
            print(
                "⚠️ Không thể tự mở trình duyệt. "
                "Hãy mở đường dẫn phía trên thủ công."
            )

        for _ in range(self.MAX_VERIFICATION_ATTEMPTS):
            self.confirmation_input(
                "Quét QR và hoàn tất xác thực trong trình duyệt, "
                "sau đó nhấn Enter..."
            )
            response = self._request_authentication()

            if response.status_code in [200, 201]:
                return

            if (
                response.status_code in self.VERIFICATION_STATUS_CODES
                or response.status_code in [401, 403]
                or self._extract_verification_url(response)
            ):
                continue

            raise AuthenticationError(
                f"Xác thực thất bại: HTTP {response.status_code}."
            )

        raise AuthenticationError(
            "Xác thực bổ sung chưa hoàn tất sau ba lần kiểm tra."
        )

    # -- Metadata API ------------------------------------------------------

    def _get(self, path, params=None):
        response = self.session.get(
            f"{self.API_BASE_URL}{path}",
            params=params,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )
        return self._handle_response(response, path)

    @staticmethod
    def _handle_response(response, path):
        if response.status_code == 429:
            raise WorldQuantRateLimitError(
                float(response.headers.get("Retry-After", 1))
            )
        if response.status_code >= 400:
            raise WorldQuantApiError(path, response.status_code)
        return response.json()

    def get_configuration(self):
        return self._get("/configuration")

    def get_categories(self):
        return self._get("/data-categories")

    def get_operators(self):
        return self._get("/operators")

    @staticmethod
    def _scope_params(scope):
        return {
            "instrumentType": scope.instrument_type,
            "region": scope.region,
            "delay": scope.delay,
            "universe": scope.universe,
        }

    def _iter_pages(self, path, params, limit):
        offset = 0
        while True:
            page_params = dict(params)
            page_params["limit"] = limit
            page_params["offset"] = offset
            data = self._get(path, page_params)
            results = data.get("results", []) or []
            for row in results:
                yield row
            count = data.get("count", 0) or 0
            offset += limit
            if offset >= count or not results:
                break

    def iter_datasets(self, scope, limit=50):
        yield from self._iter_pages("/data-sets", self._scope_params(scope), limit)

    def iter_data_fields(self, scope, dataset_id=None, limit=50):
        params = self._scope_params(scope)
        if dataset_id:
            params["dataset.id"] = dataset_id
        yield from self._iter_pages("/data-fields", params, limit)

    # -- Scope extraction --------------------------------------------------

    @staticmethod
    def _find_child(children, name):
        for child in children or []:
            if child.get("name") == name:
                return child
        return None

    @staticmethod
    def _options(node):
        return node.get("options", []) if node else []

    def extract_scopes(self, configuration):
        children = (
            configuration.get("actions", {})
            .get("POST", {})
            .get("settings", {})
            .get("children", [])
        )
        instrument_node = self._find_child(children, "instrumentType")
        scopes = []
        for instrument_opt in self._options(instrument_node):
            instrument = instrument_opt.get("value")
            region_node = self._find_child(
                instrument_opt.get("children", []), "region"
            )
            for region_opt in self._options(region_node):
                region = region_opt.get("value")
                sub_children = region_opt.get("children", [])
                universe_opts = self._options(
                    self._find_child(sub_children, "universe")
                )
                delay_opts = self._options(self._find_child(sub_children, "delay"))
                for universe_opt in universe_opts:
                    for delay_opt in delay_opts:
                        scopes.append(Scope(
                            instrument,
                            region,
                            int(delay_opt.get("value")),
                            universe_opt.get("value"),
                        ))
        unique = []
        seen = set()
        for scope in scopes:
            if scope not in seen:
                seen.add(scope)
                unique.append(scope)
        return unique

    # -- Simulation --------------------------------------------------------

    def simulate_alpha(self, payload):
        creation = self.session.post(
            f"{self.API_BASE_URL}/simulations",
            json=payload,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )
        if creation.status_code == 429:
            raise WorldQuantRateLimitError(
                float(creation.headers.get("Retry-After", 1))
            )
        if creation.status_code != 201:
            return self._creation_error(creation)

        location = creation.headers.get("Location")
        if not location:
            return SimulationResult(
                worldquant_alpha_id=None,
                status="REQUEST_ERROR",
                error_code="REQUEST_ERROR",
                error_message="Thiếu URL tiến độ simulation (Location).",
            )

        alpha_id = self._poll_simulation(location)
        if alpha_id is None:
            return SimulationResult(
                worldquant_alpha_id=None,
                status="TIMEOUT",
                error_code="TIMEOUT",
                error_message="Simulation vượt quá thời gian chờ.",
            )

        alpha_detail = self.session.get(
            f"{self.API_BASE_URL}/alphas/{alpha_id}",
            timeout=self.REQUEST_TIMEOUT_SECONDS,
        )
        alpha_data = alpha_detail.json()
        is_data = alpha_data.get("is", {}) or {}
        return SimulationResult(
            worldquant_alpha_id=alpha_id,
            status="COMPLETED",
            metrics=is_data,
            checks=is_data.get("checks", []) or [],
            raw_response=alpha_data,
        )

    def _poll_simulation(self, location):
        total_wait = 0.0
        while True:
            progress = self.session.get(
                location, timeout=self.REQUEST_TIMEOUT_SECONDS
            )
            retry_after = float(progress.headers.get("Retry-After", 0))
            if retry_after == 0:
                return progress.json().get("alpha")
            total_wait += retry_after
            if total_wait > self.poll_timeout_seconds:
                return None
            self.sleep(retry_after)

    def _creation_error(self, response):
        try:
            data = response.json()
        except (TypeError, ValueError):
            data = {}
        message = ""
        if isinstance(data, dict):
            message = str(data.get("message") or data.get("detail") or "")
        haystack = (json.dumps(data, ensure_ascii=False) + " " + message).lower()

        if response.status_code in (401, 403) or "authoriz" in haystack \
                or "permission" in haystack:
            error_code = "DATASET_AUTHORIZATION_ERROR"
        elif "compile" in haystack or "syntax" in haystack \
                or response.status_code == 400:
            error_code = "COMPILE_ERROR"
        else:
            error_code = "REQUEST_ERROR"

        return SimulationResult(
            worldquant_alpha_id=None,
            status="FAILED",
            error_code=error_code,
            error_message=message or f"HTTP {response.status_code}",
            raw_response=data if isinstance(data, dict) else {"raw": data},
        )
