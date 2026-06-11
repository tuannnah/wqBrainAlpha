"""Test WQBrainClient với httpx.MockTransport (không gọi mạng thật)."""

from __future__ import annotations

import httpx
import pytest

from src.data.client import AuthError, WQBrainClient


def _client_with(handler, **kwargs) -> WQBrainClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url=WQBrainClient.BASE_URL, transport=transport)
    return WQBrainClient("user@example.com", "secret", client=http, **kwargs)


def test_authenticate_thanh_cong_status_201():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/authentication"
        assert request.headers.get("Authorization", "").startswith("Basic ")
        return httpx.Response(201)

    client = _client_with(handler)
    client.authenticate()
    assert client.authenticated is True


def test_authenticate_401_raise_autherror():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, headers={"WWW-Authenticate": "https://verify.example/x"})

    client = _client_with(handler)
    with pytest.raises(AuthError):
        client.authenticate()


def test_xac_thuc_qr_mo_trinh_duyet_va_thu_lai():
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            # Lần đầu: WQ yêu cầu xác thực QR, trả URL trong JSON.
            return httpx.Response(
                202, json={"inquiryUrl": "https://platform.worldquantbrain.com/verify"}
            )
        return httpx.Response(201)  # sau khi user quét QR

    opened = []
    waited = []
    client = _client_with(
        handler,
        browser_open=lambda url: opened.append(url) or True,
        confirmation_input=lambda prompt: waited.append(prompt),
    )
    client.authenticate()

    assert client.authenticated is True
    assert opened == ["https://platform.worldquantbrain.com/verify"]  # đã mở trình duyệt 1 lần
    assert len(waited) == 1  # đã chờ user nhấn Enter


def test_extract_verification_url_tu_header():
    resp = httpx.Response(202, headers={"Location": "/authentication/verify/abc"})
    url = WQBrainClient._extract_verification_url(resp)
    assert url == "https://api.worldquantbrain.com/authentication/verify/abc"


def test_get_tu_reauth_khi_session_het_han():
    state = {"auth_count": 0, "data_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/authentication":
            state["auth_count"] += 1
            return httpx.Response(201)
        # Lần gọi data đầu trả 401, lần thứ hai (sau re-auth) trả 200.
        state["data_calls"] += 1
        if state["data_calls"] == 1:
            return httpx.Response(401)
        return httpx.Response(200, json={"ok": True})

    client = _client_with(handler)
    resp = client.get("/data-fields")
    assert resp.status_code == 200
    assert state["auth_count"] == 2  # auth ban đầu + re-auth
