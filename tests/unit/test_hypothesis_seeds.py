"""HYPOTHESIS_CORES: core hypothesis-driven cho 4 HỌ NHÂN TỐ MỚI (analyst_revision/
short_interest/earnings_drift/value_quality) — mở khỏi 5 cụm đã bão hòa (pv_reversal/
momentum/fundamental ratio đơn/options_iv/news_social). Field analyst4 + short-interest ĐÃ
verify LIVE 2026-07-14 qua tools/verify_datasets.py (logs/verified_fields_20260714.json);
field short-interest suy đoán cũ (days_to_cover/shares_short) account KHÔNG có -> đã thay
bằng field securities-lending (shortinterest3) + SI-surprise (short_interest_pred) có thật."""

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


def test_seed_short_interest_dung_field_verify_live():
    """Field short-interest phải là field verify LIVE 14/07 (logs/verified_fields_20260714.json):
    loan_utilization_ratio + mean_loan_rate (shortinterest3, securities lending, coverage 1.0)
    và short_interest_surprise_ratio (short_interest_pred, coverage 0.9987). Tên suy đoán cũ
    days_to_cover/shares_short account KHÔNG có (field guard từng chặn cả họ) — cấm tái xuất."""
    joined = " ".join(HYPOTHESIS_CORES)
    assert "days_to_cover" not in joined
    assert "shares_short" not in joined
    assert "loan_utilization_ratio" in joined
    assert "mean_loan_rate" in joined
    assert "short_interest_surprise_ratio" in joined


def test_khong_dung_group_neutralize_wrapper():
    """Core THUẦN (không bọc group_neutralize/scale/ts_decay_linear) — neutralization/decay áp
    qua sim settings ở refiner, không phải literal trong expression (khớp quy ước ALT_DATA_CORES
    /FUNDAMENTAL_CORES)."""
    for expr in HYPOTHESIS_CORES:
        assert "group_neutralize(" not in expr
        assert "scale(" not in expr
        assert "ts_decay_linear(" not in expr
