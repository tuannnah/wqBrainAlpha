"""TDD cho combine_stage: điều phối chọn combo -> dựng -> chấm local -> giữ combo tốt.

Dùng scorer giả (không backtest thật) để test logic điều phối/lọc thuần.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import src.operators_local  # noqa: F401 — đăng ký operator cho parse/depth
from src.generation.combiner import SubSignal, build_combined_expression
from src.pipeline.combine_stage import combine_stage

DATES = np.arange(200)


def _sig(expr: str, pnl: np.ndarray, score: float) -> SubSignal:
    return SubSignal(expr=expr, pnl=pnl, dates=DATES.copy(), score=score)


@dataclass
class _FakeMetrics:
    fitness: float


@dataclass
class _FakeVerdict:
    passed: bool


@dataclass
class _FakeScore:
    metrics: _FakeMetrics
    verdict: _FakeVerdict
    pnl: np.ndarray
    dates: np.ndarray


def _scorer(fitness_map: dict[str, float], passed: bool = True):
    """Trả score_fn tra fitness theo expr; expr lạ -> fitness mặc định 0.5."""

    def score_fn(expr: str) -> _FakeScore:
        fit = fitness_map.get(expr, 0.5)
        return _FakeScore(_FakeMetrics(fit), _FakeVerdict(passed), np.zeros(200), DATES.copy())

    return score_fn


def _two_uncorrelated():
    rng = np.random.default_rng(1)
    a = _sig("-ts_mean(subtract(close, vwap), 10)", rng.normal(size=200), 1.0)
    b = _sig("-ts_mean(subtract(close, open), 5)", rng.normal(size=200), 0.9)
    return [a, b]


def test_giu_combo_vuot_tin_hieu_con_tot_nhat():
    sigs = _two_uncorrelated()
    combined_expr = build_combined_expression([s.expr for s in sigs]).expr
    # combo fitness 2.0 > mọi tín hiệu con (đều 0.5 dưới base cfg) -> giữ.
    score_fn = _scorer({combined_expr: 2.0})
    out = combine_stage(sigs, score_fn, tau=0.5, n_min=2, n_max=2, max_combos=1)
    assert len(out) == 1
    assert out[0].expr == combined_expr


def test_loai_combo_khong_vuot_tin_hieu_con():
    sigs = _two_uncorrelated()
    # combo fitness 0.4 < tín hiệu con 0.5 -> loại.
    score_fn = _scorer({})  # mọi expr -> 0.5; combo cũng 0.5, không > best -> loại
    out = combine_stage(sigs, score_fn, tau=0.5, n_min=2, n_max=2, max_combos=1)
    assert out == []


def test_loai_combo_khong_qua_gate_local():
    sigs = _two_uncorrelated()
    score_fn = _scorer({}, passed=False)
    out = combine_stage(sigs, score_fn, tau=0.5, n_min=2, n_max=2, max_combos=1)
    assert out == []


def test_khong_du_tin_hieu_khong_combo():
    rng = np.random.default_rng(2)
    one = [_sig("ts_delta(close, 5)", rng.normal(size=200), 1.0)]
    out = combine_stage(one, _scorer({}), tau=0.5, n_min=2, n_max=2, max_combos=1)
    assert out == []
