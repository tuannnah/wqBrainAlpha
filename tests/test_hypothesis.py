"""Test sinh giả thuyết thị trường có cấu trúc 4 phần (GĐ2: T2.3)."""

from __future__ import annotations

import json

from src.llm.hypothesis import Hypothesis, HypothesisGenerator
from tests.fakes import FakeDeepSeek


def test_generate_parse_du_4_phan():
    payload = {
        "observation": "cổ phiếu thanh khoản cao đảo chiều nhanh",
        "background": "lý thuyết vi cấu trúc thị trường",
        "economic_rationale": "nhà tạo lập điều chỉnh tồn kho",
        "implementation_spec": "dùng volume và spread, cửa sổ 5-20",
    }
    ds = FakeDeepSeek([json.dumps(payload)])
    gen = HypothesisGenerator(ds)
    h = gen.generate("mean-reversion theo thanh khoản")

    assert isinstance(h, Hypothesis)
    assert h.observation.startswith("cổ phiếu")
    assert "tạo lập" in h.economic_rationale
    assert h.implementation_spec


def test_prompt_chua_huong_nghien_cuu():
    ds = FakeDeepSeek([json.dumps({"observation": "x"})])
    HypothesisGenerator(ds).generate("momentum ngành công nghệ")
    system, user = ds.calls[0]
    assert "momentum ngành công nghệ" in (system + user)


def test_generate_an_toan_khi_thieu_phan():
    # Model trả thiếu vài phần -> không crash, điền chuỗi rỗng.
    ds = FakeDeepSeek([json.dumps({"observation": "chỉ có quan sát"})])
    h = HypothesisGenerator(ds).generate("bất kỳ")
    assert h.observation == "chỉ có quan sát"
    assert h.background == ""
    assert h.economic_rationale == ""


def test_generate_an_toan_khi_json_hong():
    ds = FakeDeepSeek(["không phải json gì cả"])
    h = HypothesisGenerator(ds).generate("x")
    assert isinstance(h, Hypothesis)
    assert h.observation == ""


def test_hypothesis_to_dict_roundtrip():
    h = Hypothesis("a", "b", "c", "d")
    d = h.to_dict()
    assert d == {
        "observation": "a",
        "background": "b",
        "economic_rationale": "c",
        "implementation_spec": "d",
    }


from src.llm.hypothesis import ground_fields


def test_ground_fields_bo_field_bia_giu_field_that():
    out = ground_fields(["opt6_real", "bia_field"], ["opt6_real", "pcr_oi_30"])
    # min_k=2 mặc định: chỉ 1 field hợp lệ ("opt6_real") nên sẽ augment thêm
    # từ palette cho đủ min_k -> luôn ra đúng 2 phần tử, field bịa bị loại.
    assert out[0] == "opt6_real"
    assert "bia_field" not in out


def test_ground_fields_augment_khi_thieu_min_k():
    # LLM toàn field bịa -> augment từ palette cho đủ min_k=2.
    out = ground_fields(["bia1", "bia2"], ["a", "b", "c"], min_k=2)
    assert out == ("a", "b")


def test_ground_fields_giu_thu_tu_va_khu_trung_lap():
    out = ground_fields(["a", "a", "b"], ["a", "b", "c"], min_k=2)
    assert out == ("a", "b")


def test_ground_fields_rong_tra_tuple_rong():
    assert ground_fields(None, [], min_k=2) == ()
    assert ground_fields("chuoi_don", [], min_k=2) == ()


def test_hypothesis_co_field_mac_dinh_rong():
    assert Hypothesis("a", "b", "c", "d").fields == ()
