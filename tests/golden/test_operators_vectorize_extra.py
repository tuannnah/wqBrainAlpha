"""Characterization test cho các operator được vectorize thêm (conditional/group/
neutralization): impl mới phải khớp bản tham chiếu vòng lặp gốc trên dữ liệu có NaN,
group nhiều nhãn, cột hằng, giá trị lớn."""

from __future__ import annotations

import numpy as np

import src.operators_local.conditional  # noqa: F401
import src.operators_local.group  # noqa: F401
import src.operators_local.neutralization  # noqa: F401
from src.operators_local.conditional import hump
from src.operators_local.group import group_neutralize
from src.operators_local.neutralization import regression_neut, vector_neut


class _FakeCtx:
    def __init__(self, groups):
        self.data = type("D", (), {"groups": groups})()


def _sample(shape=(50, 14), scale=40.0, offset=100.0):
    rng = np.random.default_rng(20260705)
    x = rng.standard_normal(shape) * scale + offset
    x[3, 2] = np.nan
    x[10, :] = np.nan
    x[20, 5:] = np.nan
    return x


# ---------------- hump ----------------
def _ref_hump(x, thr):
    out = x.copy()
    for col in range(x.shape[1]):
        last = np.nan
        for t in range(x.shape[0]):
            cur = x[t, col]
            if np.isnan(cur):
                continue
            if np.isnan(last) or abs(cur - last) >= thr:
                last = cur
            out[t, col] = last
    return out


def test_hump_vectorized_khop_reference() -> None:
    x = _sample()
    for thr in (0.0, 5.0, 20.0):
        np.testing.assert_allclose(
            hump(None, x, thr), _ref_hump(x, thr), equal_nan=True, rtol=1e-12, atol=1e-12
        )


# ---------------- group_neutralize ----------------
def _ref_group_neut(x, groups):
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row, grp_row = x[t], groups[t]
        valid = ~np.isnan(row)
        if not np.any(valid):
            continue
        for g in np.unique(grp_row[valid]):
            mask = valid & (grp_row == g)
            if not np.any(mask):
                continue
            out[t][mask] = row[mask] - float(np.mean(row[mask]))
    return out


def test_group_neutralize_vectorized_khop_reference() -> None:
    x = _sample()
    rng = np.random.default_rng(7)
    groups = rng.integers(0, 4, size=x.shape)  # 4 sector, đổi theo ngày
    ctx = _FakeCtx({"sector": groups})
    got = group_neutralize(ctx, x, "sector")
    exp = _ref_group_neut(x, groups)
    np.testing.assert_allclose(got, exp, equal_nan=True, rtol=1e-9, atol=1e-9)


# ---------------- regression_neut ----------------
def _ref_regression_neut(y, x):
    out = np.full_like(y, np.nan, dtype=np.float64)
    for t in range(y.shape[0]):
        yr, xr = y[t], x[t]
        valid = ~np.isnan(yr) & ~np.isnan(xr)
        n = int(valid.sum())
        if n < 2:
            continue
        xv, yv = xr[valid], yr[valid]
        if np.std(xv) == 0.0:
            out[t][valid] = yv - float(np.mean(yv))
            continue
        design = np.column_stack([np.ones(n), xv])
        coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
        out[t][valid] = yv - design @ coef
    return out


def test_regression_neut_vectorized_khop_reference() -> None:
    y = _sample()
    x = _sample(offset=50.0, scale=30.0)[::-1]  # x khác y
    x[25, :3] = 7.0  # tạo hàng có std(x)=0 trên vài cột (không đủ) — kiểm nhánh
    got = regression_neut(None, y, x)
    exp = _ref_regression_neut(y, x)
    np.testing.assert_allclose(got, exp, equal_nan=True, rtol=1e-7, atol=1e-7)


# ---------------- vector_neut ----------------
def _ref_vector_neut(x, y):
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        xr, yr = x[t], y[t]
        valid = ~np.isnan(xr) & ~np.isnan(yr)
        if not np.any(valid):
            continue
        xv, yv = xr[valid], yr[valid]
        denom = float(np.dot(yv, yv))
        if denom == 0.0:
            out[t][valid] = xv
            continue
        out[t][valid] = xv - (float(np.dot(xv, yv)) / denom) * yv
    return out


def test_vector_neut_vectorized_khop_reference() -> None:
    x = _sample()
    y = _sample(offset=0.0, scale=1.0)[::-1]
    y[30, :] = 0.0  # hàng y toàn 0 -> denom=0 (giữ nguyên x)
    got = vector_neut(None, x, y)
    exp = _ref_vector_neut(x, y)
    np.testing.assert_allclose(got, exp, equal_nan=True, rtol=1e-9, atol=1e-9)
