"""Test tools/verify_datasets.py (Task 7) — phần thuần (lọc dataset quan tâm, dựng JSON, bảng
tóm tắt) qua TDD với dữ liệu giả, cộng hành vi lịch sự khi session hết hạn/API lỗi giữa chừng
(client giả — KHÔNG gọi mạng thật, KHÔNG đăng nhập)."""

from __future__ import annotations

import httpx

from src.data.client import AuthError
from tools.verify_datasets import (
    build_verified_json,
    fetch_datasets,
    fetch_fields_for_dataset,
    field_entry,
    filter_interest_datasets,
    is_interest_dataset,
    main,
    render_summary_table,
)


def _ds(id_, name="", category=None, subcategory=None):
    d = {"id": id_, "name": name}
    if category is not None:
        d["category"] = category
    if subcategory is not None:
        d["subcategory"] = subcategory
    return d


# --------------------------------------------------------------- is_interest_dataset
def test_khop_dataset_short_interest_theo_category():
    ds = _ds("shortinterest30", "Short Interest",
             category={"id": "shortinterest", "name": "Short Interest"})
    assert is_interest_dataset(ds)


def test_khop_dataset_earnings_theo_ten():
    ds = _ds("earnings7", "Earnings Data")
    assert is_interest_dataset(ds)


def test_khop_dataset_analyst_theo_subcategory():
    ds = _ds("model17", "Model", subcategory={"id": "est", "name": "Estimates Models (Analyst)"})
    assert is_interest_dataset(ds)


def test_khong_khop_dataset_price_volume():
    ds = _ds("pv1", "Price Volume Data", category={"id": "pv", "name": "Price Volume"})
    assert not is_interest_dataset(ds)


def test_khop_khong_phan_biet_hoa_thuong():
    ds = _ds("OPT6", "OPTION VOLATILITY", category={"id": "OPTION", "name": "OPTION"})
    assert is_interest_dataset(ds)


def test_category_dang_chuoi_tho_khong_vo():
    """Phòng trường hợp API trả category dạng chuỗi thay vì dict — không được raise."""
    ds = {"id": "news35", "name": "News", "category": "News"}
    assert is_interest_dataset(ds)


# --------------------------------------------------------------- filter_interest_datasets
def test_loc_dung_tap_con_giu_thu_tu():
    datasets = [
        _ds("pv1", "Price Volume", category={"id": "pv", "name": "Price Volume"}),
        _ds("shortinterest30", "Short Interest", category={"id": "shortinterest", "name": "Short interest"}),
        _ds("news59", "News", category={"id": "news", "name": "News"}),
    ]
    out = filter_interest_datasets(datasets)
    assert [d["id"] for d in out] == ["shortinterest30", "news59"]


def test_loc_rong_khi_khong_dataset_nao_khop():
    datasets = [_ds("pv1", "Price Volume", category={"id": "pv", "name": "Price Volume"})]
    assert filter_interest_datasets(datasets) == []


# --------------------------------------------------------------- field_entry
def test_field_entry_lay_id_va_coverage():
    raw = {"id": "days_to_cover", "description": "Days to cover", "coverage": 0.42}
    assert field_entry(raw) == {"id": "days_to_cover", "coverage": 0.42}


def test_field_entry_coverage_thieu_tra_none():
    raw = {"id": "shares_short"}
    assert field_entry(raw) == {"id": "shares_short", "coverage": None}


# --------------------------------------------------------------- build_verified_json
def test_build_verified_json_dung_thu_tu_va_dataset_thieu_field_ra_rong():
    datasets = [_ds("a"), _ds("b")]
    fields_by_dataset = {"a": [{"id": "f1", "coverage": 0.9}]}
    out = build_verified_json(datasets, fields_by_dataset)
    assert out == {"a": [{"id": "f1", "coverage": 0.9}], "b": []}
    assert list(out.keys()) == ["a", "b"]


# --------------------------------------------------------------- render_summary_table
def test_render_summary_table_co_dataset_va_field_mau():
    verified = {"shortinterest30": [{"id": "days_to_cover", "coverage": 0.5}]}
    table = render_summary_table(verified)
    assert "shortinterest30" in table
    assert "days_to_cover" in table


def test_render_summary_table_dataset_rong_van_xuat_hien():
    verified = {"shortinterest30": []}
    table = render_summary_table(verified)
    assert "shortinterest30" in table


# --------------------------------------------------------------- phân trang nhiều trang
class _FakeClientPhanTrang:
    """Client giả trả dữ liệu THEO TRANG thật sự: count > page_size, dữ liệu thật ở offset>0,
    trang sau cuối rỗng — để kích hoạt nhánh ghép trang 2+ (dataset lớn như news/model17 có
    >50 field; sai phân trang là mất field ÂM THẦM)."""

    def __init__(self, datasets, fields, co_count=True):
        self._datasets = datasets
        self._fields = fields
        self._co_count = co_count

    def get(self, path, params=None, **kwargs):  # noqa: ARG002
        params = params or {}
        offset = params.get("offset", 0)
        limit = params.get("limit", 50)
        src = self._datasets if path == "/data-sets" else self._fields
        payload = {"results": src[offset:offset + limit]}
        if self._co_count:
            payload["count"] = len(src)
        return _FakeResponse(payload)


def test_fetch_datasets_ghep_du_3_trang():
    datasets = [_ds(f"ds{i}") for i in range(5)]
    client = _FakeClientPhanTrang(datasets=datasets, fields=[])
    out = fetch_datasets(client, page_size=2)  # 5 phần tử / trang 2 -> 3 trang (2+2+1)
    assert [d["id"] for d in out] == ["ds0", "ds1", "ds2", "ds3", "ds4"]


def test_fetch_fields_ghep_du_3_trang():
    fields = [{"id": f"f{i}", "coverage": 0.1 * i} for i in range(5)]
    client = _FakeClientPhanTrang(datasets=[], fields=fields)
    out = fetch_fields_for_dataset(client, "news59", page_size=2)
    assert [e["id"] for e in out] == ["f0", "f1", "f2", "f3", "f4"]
    assert out[3]["coverage"] == 0.1 * 3  # coverage từ trang 2 không bị rơi rớt


def test_fetch_datasets_khong_co_count_dung_o_trang_rong():
    """API không trả `count` -> vòng lặp phải dừng nhờ trang rỗng, vẫn ghép đủ mọi trang."""
    datasets = [_ds(f"ds{i}") for i in range(4)]
    client = _FakeClientPhanTrang(datasets=datasets, fields=[], co_count=False)
    out = fetch_datasets(client, page_size=2)  # trang 3 rỗng -> dừng
    assert [d["id"] for d in out] == ["ds0", "ds1", "ds2", "ds3"]


# --------------------------------------------------------------- main(): hành vi lịch sự
class _FakeClientSessionInvalid:
    def is_session_valid(self) -> bool:
        return False


def test_main_bao_loi_lich_su_khi_session_het_han(capsys):
    """Session hết hạn -> main() trả exit code khác 0, in hướng dẫn đăng nhập, KHÔNG raise
    (không traceback) — đúng yêu cầu brief Task 7."""
    code = main(client=_FakeClientSessionInvalid())
    assert code != 0
    out = capsys.readouterr().out
    assert "run.bat" in out
    assert "đăng nhập" in out.lower()


class _FakeClientApiFailsMidway:
    def is_session_valid(self) -> bool:
        return True

    def get(self, path, params=None, **kwargs):  # noqa: ARG002
        raise RuntimeError("giả lập lỗi mạng giữa chừng")


def test_main_bao_loi_lich_su_khi_goi_api_that_bai_giua_chung(capsys):
    """Lỗi giữa chừng khi gọi /data-sets (mất mạng/429/...) -> vẫn trả exit code khác 0 và in
    thông báo gọn — nếu exception thoát ra ngoài main(), test này tự fail vì pytest sẽ raise."""
    code = main(client=_FakeClientApiFailsMidway())
    assert code != 0
    out = capsys.readouterr().out
    assert "Lỗi khi gọi API" in out


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClientHappyPath:
    """Client giả mô phỏng 1 dataset quan tâm (shortinterest30) với 2 field, để kiểm tra
    main() chạy hết vòng đời (session hợp lệ -> tải dataset -> lọc -> tải field -> ghi JSON)
    mà không đụng mạng thật."""

    def is_session_valid(self) -> bool:
        return True

    def get(self, path, params=None, **kwargs):  # noqa: ARG002
        offset = (params or {}).get("offset", 0)
        if offset > 0:
            return _FakeResponse({"results": [], "count": 0})
        if path == "/data-sets":
            return _FakeResponse({
                "results": [
                    {"id": "pv1", "name": "Price Volume", "category": {"id": "pv", "name": "Price Volume"}},
                    {"id": "shortinterest30", "name": "Short Interest",
                     "category": {"id": "shortinterest", "name": "Short interest"}},
                ],
                "count": 2,
            })
        if path == "/data-fields":
            return _FakeResponse({
                "results": [
                    {"id": "days_to_cover", "coverage": 0.55},
                    {"id": "shares_short", "coverage": 0.55},
                ],
                "count": 2,
            })
        raise AssertionError(f"path không mong đợi: {path}")


def test_main_chay_het_vong_doi_khi_session_hop_le(tmp_path, monkeypatch, capsys):
    """Happy path: chỉ dataset quan tâm (shortinterest30) được tải field, pv1 bị loại; JSON
    được ghi vào logs/verified_fields_<ngày>.json dưới ROOT giả (monkeypatch ROOT)."""
    import tools.verify_datasets as vd

    monkeypatch.setattr(vd, "ROOT", tmp_path)
    code = main(client=_FakeClientHappyPath())
    assert code == 0

    out_files = list((tmp_path / "logs").glob("verified_fields_*.json"))
    assert len(out_files) == 1
    import json

    data = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert list(data.keys()) == ["shortinterest30"]
    assert {e["id"] for e in data["shortinterest30"]} == {"days_to_cover", "shares_short"}

    out = capsys.readouterr().out
    assert "  - shortinterest30:" in out
    assert "  - pv1:" not in out  # pv1 (Price Volume) không thuộc nhóm quan tâm -> không được liệt kê


# --------------------------------------------------------------- chống TREO khi 401 giữa chừng
class _FakeClientCoConfirmationInput(_FakeClientSessionInvalid):
    """Client giả có thuộc tính confirmation_input (như WQBrainClient thật) — để kiểm tra
    main() thay nó bằng hàm KHÔNG tương tác trước khi làm bất cứ việc gì."""

    def __init__(self):
        self.confirmation_input = input  # mặc định của WQBrainClient thật


def test_main_thay_confirmation_input_bang_ham_khong_tuong_tac():
    """WQBrainClient._request() gặp 401 giữa chừng sẽ tự authenticate() lại; nhánh QR chờ
    input() -> script có thể TREO. main() phải thay confirmation_input bằng hàm ném EOFError
    (không chặn) NGAY từ đầu — kể cả khi session đã hết hạn từ trước."""
    client = _FakeClientCoConfirmationInput()
    main(client=client)
    try:
        client.confirmation_input("Quét QR...")
        raise AssertionError("confirmation_input phải ném EOFError, không được chờ input")
    except EOFError:
        pass


class _FakeClientNemEOFErrorGiuaChung:
    """Mô phỏng đúng chuỗi sự kiện treo: session hợp lệ lúc đầu, 401 giữa chừng -> client thật
    authenticate() lại -> nhánh QR gọi confirmation_input đã bị thay -> EOFError nổi lên."""

    def is_session_valid(self) -> bool:
        return True

    def get(self, path, params=None, **kwargs):  # noqa: ARG002
        raise EOFError("giả lập input() trên stdin đã đóng khi client tự xác thực QR lại")


def test_main_eoferror_giua_chung_thoat_lich_su(capsys):
    code = main(client=_FakeClientNemEOFErrorGiuaChung())
    assert code != 0
    out = capsys.readouterr().out
    assert "run.bat" in out  # hướng dẫn đăng nhập lại, không traceback


class _FakeClientNemAuthError:
    def is_session_valid(self) -> bool:
        return True

    def get(self, path, params=None, **kwargs):  # noqa: ARG002
        raise AuthError("Email hoặc mật khẩu không đúng, hoặc tài khoản không có quyền truy cập.")


def test_main_autherror_giua_chung_in_huong_dan_dang_nhap(capsys):
    code = main(client=_FakeClientNemAuthError())
    assert code != 0
    out = capsys.readouterr().out
    assert "run.bat" in out


# --------------------------------------------------------------- phân biệt lỗi mạng vs hết hạn
class _FakeClientLoiMang:
    def is_session_valid(self) -> bool:
        return True

    def get(self, path, params=None, **kwargs):  # noqa: ARG002
        raise httpx.ConnectError("giả lập đứt mạng")


def test_main_loi_mang_khong_in_huong_dan_dang_nhap(capsys):
    """Đứt mạng KHÔNG phải session hết hạn -> in 'thử lại sau', KHÔNG bảo user đăng nhập lại."""
    code = main(client=_FakeClientLoiMang())
    assert code != 0
    out = capsys.readouterr().out
    assert "Lỗi mạng" in out
    assert "run.bat" not in out


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://api.worldquantbrain.com/data-sets")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)


class _FakeClient401GiuaChung:
    def is_session_valid(self) -> bool:
        return True

    def get(self, path, params=None, **kwargs):  # noqa: ARG002
        raise _http_status_error(401)


def test_main_401_giua_chung_in_huong_dan_dang_nhap(capsys):
    code = main(client=_FakeClient401GiuaChung())
    assert code != 0
    out = capsys.readouterr().out
    assert "run.bat" in out


class _FakeClient500GiuaChung:
    def is_session_valid(self) -> bool:
        return True

    def get(self, path, params=None, **kwargs):  # noqa: ARG002
        raise _http_status_error(500)


def test_main_500_giua_chung_bao_thu_lai_sau_khong_bat_dang_nhap(capsys):
    """5xx là lỗi phía server, KHÔNG phải session -> 'thử lại sau', không hướng dẫn đăng nhập."""
    code = main(client=_FakeClient500GiuaChung())
    assert code != 0
    out = capsys.readouterr().out
    assert "thử lại sau" in out
    assert "run.bat" not in out
