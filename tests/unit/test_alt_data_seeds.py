"""Alt-data seed cores: biểu thức GAP/normalize/reversal trên dataset alt-data ĐÃ VERIFY
LIVE (option8 IV/HV, socialmedia8 sentiment) đi THẲNG tới Brain sim (không chấm local được
vì panel local chỉ có price/volume). Kèm helper map category dataset → neutralization theo
docs WQ (option→SECTOR, news/social→SUBINDUSTRY, analyst/fundamental→INDUSTRY)."""

from __future__ import annotations

import src.operators_local  # noqa: F401  # đăng ký operator (ts_backfill/ts_mean/subtract…)
from src.generation.alt_data_seeds import (
    ALT_DATA_CORES,
    neutralization_for_expr,
    pp_neut_candidates,
    pp_neutralization_for_expr,
)
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import FieldCollector

_PV_FIELDS = {"close", "open", "vwap", "volume", "high", "low", "returns"}

_ALLOWED = frozenset(
    {"SLOW", "FAST", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING"}
)


def test_moi_core_parse_duoc_qua_registry():
    # parse(validate=True) đòi mọi operator có trong registry — bắt lỗi operator lạ sớm.
    for expr in ALT_DATA_CORES:
        parse(expr)  # không raise là đạt


def test_moi_core_thuc_su_la_alt_data():
    # Mỗi core PHẢI dùng ít nhất 1 field NGOÀI price/volume (nếu không thì không phải alt-data
    # và sẽ đi nhầm đường local-tune thay vì sim-thẳng).
    reg = default_registry()
    for expr in ALT_DATA_CORES:
        fields = FieldCollector(reg).visit(parse(expr))
        assert not fields.issubset(_PV_FIELDS), expr


def test_neutralization_map_theo_category():
    # option (implied/historical_volatility) → SECTOR; social (snt_social_*) → SUBINDUSTRY.
    assert neutralization_for_expr(
        "subtract(ts_backfill(implied_volatility_put_30, 22), "
        "ts_backfill(implied_volatility_call_30, 22))"
    ) == "SECTOR"
    assert neutralization_for_expr("multiply(-1, ts_mean(snt_social_value, 5))") == "SUBINDUSTRY"
    # analyst/fundamental → INDUSTRY (dù chưa dùng trong core, helper phải map đúng).
    assert neutralization_for_expr("ts_backfill(anl4_fs_estimate_eps_mean, 66)") == "INDUSTRY"


def test_option_core_co_ts_backfill():
    # Field option sparse (coverage ~0.97) PHẢI backfill (cardinal rule #3) để không bị NaN.
    option_cores = [e for e in ALT_DATA_CORES if "implied_volatility" in e or "historical_volatility" in e]
    assert option_cores, "phải có core option"
    for expr in option_cores:
        assert "ts_backfill" in expr, expr


def test_co_da_dang_it_nhat_hai_dataset():
    # Đa dạng nguồn = giảm self-corr chéo: ít nhất option8 + socialmedia8.
    joined = " ".join(ALT_DATA_CORES)
    assert "implied_volatility" in joined
    assert "snt_social" in joined


def test_pp_neut_option_ra_statistical():
    expr = "ts_backfill(implied_volatility_call_30, 22)"
    assert pp_neutralization_for_expr(expr, _ALLOWED) == "STATISTICAL"


def test_pp_neut_social_ra_crowding():
    expr = "ts_mean(snt_social_value, 5)"
    assert pp_neutralization_for_expr(expr, _ALLOWED) == "CROWDING"


def test_pp_neut_fallback_khi_lua_chon_ngoai_allowed():
    # allowed KHÔNG có STATISTICAL -> rơi về phần tử đầu (sorted) của allowed
    allowed = frozenset({"CROWDING", "SLOW"})
    expr = "ts_backfill(implied_volatility_call_30, 22)"  # option -> muốn STATISTICAL
    assert pp_neutralization_for_expr(expr, allowed) == "CROWDING"  # sorted(["CROWDING","SLOW"])[0]


def test_pp_neut_allowed_rong_ra_statistical():
    # `allowed` rỗng -> luôn STATISTICAL (an toàn chung), kể cả expr social vốn map CROWDING.
    expr = "ts_mean(snt_social_value, 5)"  # social -> CROWDING nếu có allowed
    assert pp_neutralization_for_expr(expr, frozenset()) == "STATISTICAL"


def test_pp_neut_candidates_mac_dinh_1x_va_sweep():
    expr = "ts_mean(snt_social_value, 5)"
    assert pp_neut_candidates(expr, _ALLOWED) == ["CROWDING"]
    assert pp_neut_candidates(expr, _ALLOWED, sweep=True) == sorted(_ALLOWED)
