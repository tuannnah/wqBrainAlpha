"""HYPOTHESIS_CORES: core hypothesis-driven cho 4 HỌ NHÂN TỐ MỚI (analyst_revision/
short_interest/earnings_drift/value_quality) — mở khỏi 5 cụm đã bão hòa (pv_reversal/
momentum/fundamental ratio đơn/options_iv/news_social). Field analyst4/short-interest CHƯA
verify live (field-validity guard ở closed_loop_adapters tự lọc nếu account thiếu) — test ở
đây chỉ đảm bảo core parse được + phân loại family đúng, KHÔNG khẳng định field tồn tại thật."""

from __future__ import annotations

import src.operators_local  # noqa: F401  # đăng ký operator (ts_backfill/ts_delta/rank…)
from src.generation.hypothesis_seeds import (
    HYPOTHESIS_CORES,
    HYPOTHESIS_FIELDS,
    hypothesis_fields_in,
)
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor
from src.reporting.diagnostics import classify_family

_NEW_FAMILIES = frozenset(
    {"analyst_revision", "short_interest", "earnings_drift", "value_quality"}
)


def test_moi_core_parse_duoc_qua_registry():
    for expr in HYPOTHESIS_CORES:
        parse(expr)  # không raise -> hợp lệ theo registry local (operator tồn tại + đúng arity)


def test_moi_core_trong_ngan_sach_do_sau():
    for expr in HYPOTHESIS_CORES:
        depth = DepthVisitor().visit(parse(expr))
        assert depth <= 7, f"{expr} depth={depth} vượt MAX_DEPTH"


def test_moi_field_nam_trong_hypothesis_fields():
    reg = default_registry()
    for expr in HYPOTHESIS_CORES:
        fields = hypothesis_fields_in(expr, reg)
        for f in fields:
            assert f in HYPOTHESIS_FIELDS, f"field lạ {f} trong {expr}"


def test_moi_core_phan_loai_ra_family_moi_rieng_biet():
    """classify_family phải trả nhãn MỚI (không rơi vào pv_reversal/momentum/fundamental cũ)
    cho MỌI core hypothesis — đúng yêu cầu Task 4 (family-budget coi đây là orthogonal)."""
    seen: set[str] = set()
    for expr in HYPOTHESIS_CORES:
        fam = classify_family(expr)
        assert fam in _NEW_FAMILIES, f"{expr} -> family {fam} không phải family mới"
        assert fam not in {"pv_reversal", "momentum", "fundamental"}
        seen.add(fam)
    # Cả 4 family mới đều có ít nhất 1 core (không lỡ quên 1 họ khi soạn HYPOTHESIS_CORES).
    assert seen == _NEW_FAMILIES


def test_co_it_nhat_5_core():
    assert len(HYPOTHESIS_CORES) >= 5


def test_khong_dung_group_neutralize_wrapper():
    """Core THUẦN (không bọc group_neutralize/scale/ts_decay_linear) — neutralization/decay áp
    qua sim settings ở refiner, không phải literal trong expression (khớp quy ước ALT_DATA_CORES
    /FUNDAMENTAL_CORES)."""
    for expr in HYPOTHESIS_CORES:
        assert "group_neutralize(" not in expr
        assert "scale(" not in expr
        assert "ts_decay_linear(" not in expr
