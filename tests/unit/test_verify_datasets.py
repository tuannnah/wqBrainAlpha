"""Test tools/verify_datasets.py (Task 7) — phần thuần (lọc dataset quan tâm, dựng JSON, bảng
tóm tắt) qua TDD với dữ liệu giả, cộng hành vi lịch sự khi session hết hạn/API lỗi giữa chừng
(client giả — KHÔNG gọi mạng thật, KHÔNG đăng nhập)."""

from __future__ import annotations

from tools.verify_datasets import (
    build_verified_json,
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
