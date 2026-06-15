"""Test WQBrainClient với httpx.MockTransport (không gọi mạng thật)."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import httpx
import pytest

from src.data.client import AuthError, WQBrainClient


def _tmp_session() -> Path:
    return Path(tempfile.gettempdir()) / f"wq_session_test_{uuid.uuid4().hex}"


def _client_with(handler, **kwargs) -> WQBrainClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url=WQBrainClient.BASE_URL, transport=transport)
    kwargs.setdefault("session_file", _tmp_session())
    return WQBrainClient("user@example.com", "secret", client=http, **kwargs)


def test_session_con_han_bo_qua_dang_nhap():
    import json

    state = {"auth_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/authentication":
            state["auth_calls"] += 1
            return httpx.Response(201)
        if request.url.path == "/users/self/":
            return httpx.Response(200, json={"id": "u1"})
        return httpx.Response(404)

    # Session file có sẵn cookie -> client coi như đã có phiên.
    session_file = _tmp_session()
    session_file.write_text(json.dumps({"JSESSIONID": "abc"}), encoding="utf-8")

    client = _client_with(handler, session_file=session_file)
    client.authenticate()
    assert client.authenticated is True
    assert state["auth_calls"] == 0  # KHÔNG gọi /authentication vì session còn hạn


def test_session_luu_va_nap_lai():
    import json

    sf = _tmp_session()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    c1 = _client_with(handler, session_file=sf)
    c1.client.cookies.set("JSESSIONID", "xyz")
    c1._save_session()

    saved = json.loads(sf.read_text(encoding="utf-8"))
    assert saved.get("JSESSIONID") == "xyz"

    c2 = _client_with(handler, session_file=sf)  # nạp lại từ file
    assert c2._has_session() is True


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


def test_persona_hoan_tat_bang_post_vao_url_inquiry():
    """Sau khi quét QR, phải POST vào chính URL persona/inquiry để hoàn tất.

    Tái hiện bug: nếu POST /authentication lần nữa thì WQ sinh inquiry MỚI,
    inquiry vừa quét không bao giờ được hoàn tất -> lặp 3 lần rồi AuthError.
    """
    persona_path = "/authentication/persona"
    state = {"auth_calls": 0, "persona_posts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/authentication":
            state["auth_calls"] += 1
            # Mỗi lần POST /authentication -> 401 kèm inquiry MỚI.
            return httpx.Response(
                401,
                headers={
                    "WWW-Authenticate": "persona",
                    "Location": f"{persona_path}?inquiry=inq_{state['auth_calls']}",
                },
            )
        if request.url.path == persona_path:
            # Quét QR xong + POST vào URL inquiry -> hoàn tất.
            state["persona_posts"] += 1
            return httpx.Response(201)
        return httpx.Response(404)

    opened = []
    waited = []
    client = _client_with(
        handler,
        browser_open=lambda url: opened.append(url) or True,
        confirmation_input=lambda prompt: waited.append(prompt),
    )
    client.authenticate()

    assert client.authenticated is True
    assert state["auth_calls"] == 1  # CHỈ gọi /authentication 1 lần (không sinh inquiry mới)
    assert state["persona_posts"] == 1  # hoàn tất bằng POST vào URL persona
    assert len(opened) == 1  # mở đúng URL inquiry để quét QR
    assert "inq_1" in opened[0]
    assert len(waited) == 1


def test_authenticate_lan_hai_la_no_op_khi_da_dang_nhap():
    """Đã đăng nhập trong phiên (kể cả persona không set cookie) -> authenticate()
    lần 2 không được gọi lại /authentication hay /users/self/ (không bắt QR lại).

    Tái hiện bug: chọn 4/5 sau khi đăng nhập lại đòi xác thực QR.
    """
    persona_path = "/authentication/persona"
    state = {"auth_calls": 0, "persona_posts": 0, "self_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/authentication":
            state["auth_calls"] += 1
            return httpx.Response(
                401,
                headers={"WWW-Authenticate": "persona", "Location": f"{persona_path}?inquiry=x"},
            )
        if request.url.path == persona_path:
            state["persona_posts"] += 1
            return httpx.Response(201)  # đăng nhập xong nhưng KHÔNG set cookie phiên
        if request.url.path == "/users/self/":
            state["self_calls"] += 1
            return httpx.Response(200, json={"id": "u1"})
        return httpx.Response(404)

    client = _client_with(
        handler,
        browser_open=lambda url: True,
        confirmation_input=lambda prompt: None,
    )
    client.authenticate()
    assert client.authenticated is True
    assert state["auth_calls"] == 1

    # Lần 2 (vd. menu chọn 4/5 truyền lại cùng client) -> phải no-op.
    client.authenticate()
    assert state["auth_calls"] == 1  # KHÔNG gọi lại /authentication
    assert state["self_calls"] == 0  # KHÔNG cần kiểm tra session qua mạng


def test_extract_verification_url_tu_header():
    resp = httpx.Response(202, headers={"Location": "/authentication/verify/abc"})
    url = WQBrainClient._extract_verification_url(resp)
    assert url == "https://api.worldquantbrain.com/authentication/verify/abc"


def test_get_tu_backoff_khi_429():
    state = {"data_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/authentication":
            return httpx.Response(201)
        state["data_calls"] += 1
        if state["data_calls"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"message": "rate limit"})
        return httpx.Response(200, json={"results": []})

    slept = []
    client = _client_with(handler, sleep_func=lambda s: slept.append(s))
    resp = client.get("/data-fields")
    assert resp.status_code == 200
    assert slept == [2.0]  # đã chờ đúng Retry-After rồi thử lại
    assert state["data_calls"] == 2


def test_authenticate_tu_doi_va_thu_lai_khi_429(capsys):
    state = {"auth_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/authentication":
            state["auth_calls"] += 1
            if state["auth_calls"] == 1:
                # Lần đầu bị giới hạn tần suất, server báo chờ 7 giây.
                return httpx.Response(429, headers={"Retry-After": "7"})
            return httpx.Response(201)  # lần hai đăng nhập được
        return httpx.Response(404)

    slept = []
    client = _client_with(handler, sleep_func=lambda s: slept.append(s))
    client.authenticate()

    assert client.authenticated is True
    assert state["auth_calls"] == 2  # 429 rồi mới 201
    assert slept == [7.0]  # đã chờ đúng Retry-After
    out = capsys.readouterr().out
    assert "7" in out  # đã in ra số giây phải chờ cho người dùng


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
