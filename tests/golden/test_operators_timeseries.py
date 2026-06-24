"""Golden test time-series ops: giá trị đúng trên small_panel, KHÔNG look-ahead (thay đổi
rows > t không đổi kết quả tại row t), thiếu lịch sử -> NaN, ts_rank bounded."""

from __future__ import annotations

import numpy as np
import pytest

import src.operators_local.timeseries  # noqa: F401  # đăng ký impl thật vào REGISTRY
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.ast import Call, Constant, Field
from src.lang.registry import default_registry


def _eval(panel: MarketData, node) -> np.ndarray:
    return Evaluator(EvalContext(data=panel, registry=default_registry())).evaluate(node)


@pytest.mark.parametrize("op", ["ts_mean", "ts_std", "ts_zscore", "ts_decay_linear"])
def test_thieu_lich_su_la_nan(small_panel, op) -> None:
    out = _eval(small_panel, Call(op, (Field("close"), Constant(20))))
    assert np.all(np.isnan(out[:19]))  # < d-1=19 quan sát -> NaN


def test_ts_delay_khong_phai_ts_delta(small_panel) -> None:
    out_delay = _eval(small_panel, Call("ts_delay", (Field("close"), Constant(5))))
    out_delta = _eval(small_panel, Call("ts_delta", (Field("close"), Constant(5))))
    close = small_panel.field("close")
    row = 50
    in_uni = small_panel.universe[row]  # row 50 < 60 -> 3 mã cuối ngoài universe (mask)
    expected_delay = close[row - 5].copy()
    expected_delay[~in_uni] = np.nan
    expected_delta = (close[row] - close[row - 5]).copy()
    expected_delta[~in_uni] = np.nan
    np.testing.assert_allclose(out_delay[row], expected_delay, equal_nan=True)
    np.testing.assert_allclose(out_delta[row], expected_delta, equal_nan=True)
    assert not np.allclose(np.nan_to_num(out_delay[row]), np.nan_to_num(out_delta[row]))


def test_ts_mean_dung_gia_tri(small_panel) -> None:
    out = _eval(small_panel, Call("ts_mean", (Field("close"), Constant(10))))
    close = small_panel.field("close")
    row = 50
    in_uni = small_panel.universe[row]  # row 50 < 60 -> 3 mã cuối ngoài universe (mask)
    expected = np.nanmean(close[row - 9 : row + 1], axis=0)
    expected[~in_uni] = np.nan
    np.testing.assert_allclose(out[row], expected, equal_nan=True)


def test_ts_rank_bounded_0_1(small_panel) -> None:
    out = _eval(small_panel, Call("ts_rank", (Field("close"), Constant(20))))
    valid = ~np.isnan(out)
    assert np.nanmin(out[valid]) >= 0.0
    assert np.nanmax(out[valid]) <= 1.0


def test_no_look_ahead_doi_tuong_lai_khong_doi_qua_khu(small_panel) -> None:
    """Bất biến cốt lõi: sửa dữ liệu CHỈ ở rows > t không thay đổi kết quả tại row t,
    cho mọi op time-series (test chỉ chọn ts_mean làm đại diện + ts_corr)."""
    row_t = 60
    mutated_close = small_panel.field("close").copy()
    mutated_close[row_t + 1 :] += 999.0  # phá tương lai
    mutated = MarketData(
        dates=small_panel.dates, assets=small_panel.assets,
        fields={**small_panel.fields, "close": mutated_close},
        universe=small_panel.universe, returns=small_panel.returns,
        groups=small_panel.groups,
    )
    out_orig = _eval(small_panel, Call("ts_mean", (Field("close"), Constant(10))))
    out_mut = _eval(mutated, Call("ts_mean", (Field("close"), Constant(10))))
    np.testing.assert_allclose(out_orig[row_t], out_mut[row_t], equal_nan=True)


def test_ts_corr_no_look_ahead(small_panel) -> None:
    row_t = 60
    mutated_volume = small_panel.field("volume").copy()
    mutated_volume[row_t + 1 :] *= 5.0
    mutated = MarketData(
        dates=small_panel.dates, assets=small_panel.assets,
        fields={**small_panel.fields, "volume": mutated_volume},
        universe=small_panel.universe, returns=small_panel.returns,
        groups=small_panel.groups,
    )
    node = Call("ts_corr", (Field("close"), Field("volume"), Constant(15)))
    out_orig = _eval(small_panel, node)
    out_mut = _eval(mutated, node)
    np.testing.assert_allclose(out_orig[row_t], out_mut[row_t], equal_nan=True)


def test_ts_backfill_lap_nan_tu_qua_khu(small_panel) -> None:
    close = small_panel.field("close").copy()
    close[40, 0] = np.nan  # 1 NaN giữa dữ liệu hợp lệ ở cột 0
    mutated = MarketData(
        dates=small_panel.dates, assets=small_panel.assets,
        fields={**small_panel.fields, "close": close},
        universe=small_panel.universe, returns=small_panel.returns,
        groups=small_panel.groups,
    )
    out = _eval(mutated, Call("ts_backfill", (Field("close"), Constant(5))))
    assert not np.isnan(out[40, 0])
    np.testing.assert_allclose(out[40, 0], close[39, 0])
