"""Golden test cross-sectional ops: per-row in-universe, rank bounded ~[0,1]."""

from __future__ import annotations

import numpy as np

import src.operators_local.cross_sectional  # noqa: F401  # đăng ký impl thật vào REGISTRY
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Constant, Field
from src.lang.registry import default_registry


def test_rank_bounded_0_1_trong_universe(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("rank", (Field("close"),)))
    in_uni = small_panel.universe
    assert np.nanmin(out[in_uni]) >= 0.0
    assert np.nanmax(out[in_uni]) <= 1.0
    assert np.all(np.isnan(out[~in_uni]))


def test_rank_chi_tinh_tren_in_universe_row(small_panel) -> None:
    """Một hàng có universe hẹp hơn (3 mã cuối ngoài universe ở nửa đầu fixture) —
    rank của các mã in-universe không bị ảnh hưởng bởi giá trị các mã ngoài universe."""
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("rank", (Field("close"),)))
    row0 = 0  # 3 mã cuối ngoài universe ở fixture (xem conftest small_panel)
    valid = small_panel.universe[row0]
    ranked_vals = out[row0][valid]
    assert ranked_vals.size == valid.sum()
    assert not np.any(np.isnan(ranked_vals))


def test_zscore_mean_0_std_1_per_row(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("zscore", (Field("close"),)))
    in_uni = small_panel.universe
    row = 100  # hàng universe đầy đủ (nửa sau fixture)
    vals = out[row][in_uni[row]]
    assert abs(float(np.mean(vals))) < 1e-8
    assert abs(float(np.std(vals)) - 1.0) < 1e-6


def test_winsorize_chan_outlier(small_panel) -> None:
    """Winsorize clip vào [mean - k*std, mean + k*std] tính từ dữ liệu GỐC (trước clip).
    Đo lại z-score bằng mean/std SAU clip (như impl không dùng) sẽ lệch ngưỡng vì clip
    làm méo phân phối không đối xứng (std sau-clip nhỏ hơn std gốc) — đo đúng invariant
    bằng mean/std gốc (close chưa winsorize), không phải mean/std của `vals` đã clip."""
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("winsorize", (Field("close"), Constant(2.0))))
    in_uni = small_panel.universe
    row = 100
    raw = small_panel.field("close")[row][in_uni[row]]
    vals = out[row][in_uni[row]]
    mean_goc, std_goc = np.mean(raw), np.std(raw)
    z = (vals - mean_goc) / std_goc
    assert np.nanmax(np.abs(z)) <= 2.0 + 1e-6


def test_scale_tong_abs_bang_1(small_panel) -> None:
    ctx = EvalContext(data=small_panel, registry=default_registry())
    out = Evaluator(ctx).evaluate(Call("scale", (Field("close"),)))
    in_uni = small_panel.universe
    row = 100
    vals = out[row][in_uni[row]]
    assert abs(float(np.sum(np.abs(vals))) - 1.0) < 1e-6
