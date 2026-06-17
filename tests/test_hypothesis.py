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
