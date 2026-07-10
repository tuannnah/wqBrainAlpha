"""Test LLMAlphaGenerator — phần ngữ cảnh prompt (blacklist field chết)."""

from __future__ import annotations

import json

from src.llm.generator import LLMAlphaGenerator
from src.simulation.pre_filter import PreFilter
from tests.fakes import FakeDeepSeek, FakeSymbolRepo


class _TField:
    def __init__(self, id, type="MATRIX"):
        self.id = id
        self.type = type
        self.description = ""
        self.dataset_id = ""


class _TFieldRepo:
    def __init__(self, fields):
        self._fields = fields

    def load_cached(self, region=None, universe=None, delay=None):
        return self._fields


def _gen(ds, fields, pf):
    ops = FakeSymbolRepo(["rank", "ts_zscore", "vec_avg"])
    return LLMAlphaGenerator(ds, _TFieldRepo(fields), ops, pf)


def test_generate_one_prompt_co_quy_tac_vector():
    pf = PreFilter(
        known_operators={"rank", "ts_zscore", "vec_avg"},
        known_fields={"svec"},
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"rank", "ts_zscore"},
    )
    ds = FakeDeepSeek([json.dumps({"expression": "rank(vec_avg(svec))"})])
    _gen(ds, [_TField("svec", "VECTOR")], pf)._generate_one("dùng svec")
    system = ds.calls[0][0]
    assert "VECTOR" in system and "vec_avg" in system


def test_generate_one_autowrap_field_vector():
    pf = PreFilter(
        known_operators={"ts_zscore", "vec_avg"},
        known_fields={"svec"},
        field_types={"svec": "VECTOR"},
        matrix_only_ops={"ts_zscore"},
    )
    ds = FakeDeepSeek([json.dumps({"expression": "ts_zscore(svec, 20)"})])
    out = _gen(ds, [_TField("svec", "VECTOR")], pf)._generate_one("svec")
    assert out == "ts_zscore(vec_avg(svec), 20)"
    assert len(ds.calls) == 1  # auto-wrap sửa ngay, không round-trip thêm


def _make_generator(blacklist=None):
    return LLMAlphaGenerator(
        FakeDeepSeek(),
        field_repo=FakeSymbolRepo(["close", "volume", "news12_sent"]),
        operator_repo=FakeSymbolRepo(["rank", "ts_mean"]),
        prefilter=None,
        blacklist=blacklist,
    )


def test_prompt_y_tuong_co_dong_cam_field():
    gen = _make_generator(blacklist={"opt6_1dorhv", "asset_growth_rate"})
    prompt = gen.build_ideas_system_prompt()
    assert "TUYỆT ĐỐI KHÔNG dùng field" in prompt
    assert "opt6_1dorhv" in prompt
    assert "asset_growth_rate" in prompt


def test_prompt_y_tuong_khong_co_dong_cam_khi_blacklist_rong():
    gen = _make_generator(blacklist=None)
    prompt = gen.build_ideas_system_prompt()
    assert "TUYỆT ĐỐI KHÔNG dùng field" not in prompt


def test_prompt_tiem_ho_da_bao_hoa():
    """Pha 2.3: họ đã bão hoà (từ exhaustion guard) tiêm vào prompt để LLM KHÔNG tái
    sinh reversal/họ đã cạn."""
    gen = _make_generator()
    gen.set_saturated_families(["pv_reversal", "momentum"])
    prompt = gen.build_ideas_system_prompt()
    assert "BÃO HOÀ" in prompt
    assert "pv_reversal" in prompt
    assert "momentum" in prompt


def test_prompt_khong_tiem_khi_khong_co_ho_bao_hoa():
    gen = _make_generator()
    prompt = gen.build_ideas_system_prompt()
    assert "BÃO HOÀ" not in prompt


def test_prompt_hypothesis_first_4_phan():
    """Fix gap Pha 2.3: prompt ép cấu trúc hypothesis-first 4 phần (observation -> theoretical
    basis -> economic mechanism -> specification) thay vì chỉ 'một câu ngắn'."""
    gen = _make_generator()
    prompt = gen.build_ideas_system_prompt()
    # 4 mốc cấu trúc (không phân biệt hoa thường)
    low = prompt.lower()
    assert "quan sát" in low          # observation
    assert "học thuật" in low or "nền tảng" in low  # theoretical basis
    assert "cơ chế" in low            # economic mechanism
    assert "cách khai thác" in low or "cụ thể hoá" in low or "specification" in low


def test_parse_ideas_loai_bo_metric_bia():
    """LLM hay nhét metric BỊA (sharpe=2.1, fitness=0.92) vào text hướng — đó là
    số tự bịa, không phải đo thật; phải bị tước để không nhiễm xuống downstream."""
    import json

    gen = _make_generator()
    content = json.dumps({"ideas": [
        "scale(ts_decay_linear(group_neutralize(x, sector), 5))  (sharpe=2.1, fitness=0.92) — option IV term structure spread",
        "News novelty reversal sau coverage dày  sharpe=1.3 fitness=0.45",
    ]})
    ideas = gen._parse_ideas(content)
    assert all("sharpe" not in i.lower() for i in ideas), ideas
    assert all("fitness" not in i.lower() for i in ideas), ideas
    # Nội dung ý nghĩa (nguồn dữ liệu/hiện tượng) phải được giữ lại.
    assert any("option IV term structure" in i for i in ideas)
    assert any("News novelty reversal" in i for i in ideas)
