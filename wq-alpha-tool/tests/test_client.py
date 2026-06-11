"""Test WQBrainClient với httpx.MockTransport (không gọi mạng thật)."""

from __future__ import annotations

import httpx
import pytest

from src.data.client import AuthError, WQBrainClient


def _client_with(handler) -> WQBrainClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url=WQBrainClient.BASE_URL, transport=transport)
    return WQBrainClient("user@example.com", "secret", client=http)


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
