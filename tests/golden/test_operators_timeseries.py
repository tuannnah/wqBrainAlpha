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


@pytest.mark.parametrize(
    "op", ["ts_mean", "ts_std", "ts_sum", "ts_std_dev", "ts_zscore", "ts_decay_linear"]
)
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


def test_ts_sum_dung_gia_tri(small_panel) -> None:
    out = _eval(small_panel, Call("ts_sum", (Field("close"), Constant(10))))
    close = small_panel.field("close")
    row = 50
    in_uni = small_panel.universe[row]  # row 50 < 60 -> 3 mã cuối ngoài universe (mask)
    expected = np.nansum(close[row - 9 : row + 1], axis=0)
    expected[~in_uni] = np.nan
    np.testing.assert_allclose(out[row], expected, equal_nan=True)


def test_ts_std_dev_dung_gia_tri(small_panel) -> None:
    out = _eval(small_panel, Call("ts_std_dev", (Field("close"), Constant(10))))
    close = small_panel.field("close")
    row = 50
    in_uni = small_panel.universe[row]  # row 50 < 60 -> 3 mã cuối ngoài universe (mask)
    expected = np.nanstd(close[row - 9 : row + 1], axis=0)
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


def _ref_ts_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Bản tham chiếu (vòng lặp thuần, chậm) — đặc tả ngữ nghĩa gốc của ts_corr để
    khóa hành vi trước khi vectorize: window [t-d+1, t], loại NaN theo cặp, valid<2
    hoặc một biến hằng (std=0) -> NaN, còn lại là Pearson corr."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        start = t - d + 1
        if start < 0:
            continue
        wx, wy = x[start : t + 1], y[start : t + 1]
        for col in range(x.shape[1]):
            sx, sy = wx[:, col], wy[:, col]
            valid = ~np.isnan(sx) & ~np.isnan(sy)
            if int(valid.sum()) < 2:
                continue
            sxv, syv = sx[valid], sy[valid]
            if np.std(sxv) == 0.0 or np.std(syv) == 0.0:
                continue
            out[t, col] = float(np.corrcoef(sxv, syv)[0, 1])
    return out


def test_ts_corr_vectorized_khop_reference() -> None:
    """ts_corr đã vectorize phải cho KẾT QUẢ y hệt bản tham chiếu vòng lặp trên dữ
    liệu có NaN rải rác, khoảng NaN dài, và cột hằng (std=0), qua nhiều window."""
    from src.operators_local.timeseries import ts_corr

    rng = np.random.default_rng(20260705)
    x = rng.standard_normal((90, 7))
    y = rng.standard_normal((90, 7))
    # NaN rải rác + một đoạn dài + đầu chuỗi
    x[5, 1] = np.nan
    y[10, 2] = np.nan
    x[20:28, 3] = np.nan
    y[0:15, 4] = np.nan
    y[:, 5] = 4.2  # cột hằng -> std=0 -> NaN theo đặc tả
    x[:, 6] = np.nan  # cột toàn NaN -> luôn NaN

    for d in (5, 15, 30):
        got = ts_corr(None, x, y, d)
        exp = _ref_ts_corr(x, y, d)
        np.testing.assert_allclose(got, exp, equal_nan=True, rtol=1e-9, atol=1e-12)


def _ref_ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    """Bản tham chiếu vòng lặp thuần của ts_rank: hạng của giá trị hiện tại trong
    cửa sổ [t-d+1, t] chuẩn hóa [0,1]; ô hiện tại NaN hoặc cửa sổ rỗng -> NaN."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        start = t - d + 1
        if start < 0:
            continue
        window = x[start : t + 1]
        for col in range(x.shape[1]):
            series = window[:, col]
            valid = ~np.isnan(series)
            n_valid = int(valid.sum())
            if n_valid == 0 or np.isnan(x[t, col]):
                continue
            vals = series[valid]
            denom = n_valid - 1 if n_valid > 1 else 1
            out[t, col] = float(np.sum(vals <= x[t, col]) - 1) / denom
    return out


def test_ts_rank_vectorized_khop_reference() -> None:
    """ts_rank đã vectorize phải cho kết quả y hệt bản tham chiếu vòng lặp, kể cả khi
    có NaN rải rác, đoạn NaN dài, giá trị trùng nhau, và ô hiện tại NaN."""
    from src.operators_local.timeseries import ts_rank

    rng = np.random.default_rng(20260705)
    x = rng.integers(0, 5, size=(90, 7)).astype(np.float64)  # nhiều giá trị trùng
    x[5, 1] = np.nan
    x[20:28, 3] = np.nan
    x[0:15, 4] = np.nan
    x[:, 6] = np.nan  # cột toàn NaN
    x[40, 0] = np.nan  # ô hiện tại NaN tại t=40

    for d in (5, 15, 30):
        got = ts_rank(None, x, d)
        exp = _ref_ts_rank(x, d)
        np.testing.assert_allclose(got, exp, equal_nan=True, rtol=1e-12, atol=1e-12)
