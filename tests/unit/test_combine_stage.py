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


# ------------------------- Fix 2: gate pool trung thực + tự-so -------------------------
# Diag đo được (logs/diag_combiner_20260712.md, 20260713.md): 3/3 combo vượt qua gate depth
# đều rớt vì self_corr >= 0.70 với `repo.load_pool()` (1321-1350 eval LOCAL bão hòa), trong
# khi self-corr THẬT của Brain cho vùng này đo được chỉ 0.40-0.46 -> proxy local giết oan.
# Fix: pool chấm gate = PnL local của CHÍNH các tín hiệu Brain-proven NGOÀI combo (không
# phải toàn bộ 1321 eval) -- loại thành phần combo khỏi pool cũng khử luôn tự-so.


def test_score_fn_factory_uu_tien_loai_thanh_vien_combo_khoi_pool():
    a, b = _two_uncorrelated()
    c = _sig("ts_rank(close, 20)", np.random.default_rng(3).normal(size=200), 0.1)
    all_sigs = [a, b, c]
    combined_expr = build_combined_expression([a.expr, b.expr]).expr

    received: list[list[SubSignal]] = []

    def factory(others: list[SubSignal]):
        received.append(others)

        def score_fn(expr: str) -> _FakeScore:
            fit = 2.0 if expr == combined_expr else 0.5
            return _FakeScore(_FakeMetrics(fit), _FakeVerdict(True), np.zeros(200), DATES.copy())

        return score_fn

    out = combine_stage(
        all_sigs, _scorer({}),  # score_fn cũ truyền vào nhưng KHÔNG được dùng vì có factory
        tau=0.5, n_min=2, n_max=2, max_combos=1, score_fn_factory=factory,
    )

    assert len(out) == 1  # factory trả fitness combo=2.0 > best_component=0.5 -> giữ
    assert len(received) == 1  # đúng 1 combo -> factory gọi đúng 1 lần
    assert {s.expr for s in received[0]} == {c.expr}  # pool CHỈ chứa C (loại A,B = combo)


# ------------------------- Fix 3: depth-aware pre-filter -------------------------
# Diag đo được (logs/diag_combiner_20260712.md: 3/5 combo, 20260713.md: 2/5 combo) chết vì
# component quá sâu -> build_combined_expression không lọt trần MAX_DEPTH sau khi bọc
# rank+add. Loại tín hiệu depth > COMBINER_MAX_COMPONENT_DEPTH NGAY TRƯỚC greedy (đo bằng
# DepthVisitor như `combiner._depth_of`) thay vì phát hiện muộn sau khi dựng thất bại.


def test_tin_hieu_qua_sau_bi_loai_truoc_greedy():
    """signal depth 6 (> COMBINER_MAX_COMPONENT_DEPTH=4) điểm CAO NHẤT — nếu không bị loại
    trước greedy sẽ luôn được chọn làm seed đầu tiên; phải KHÔNG BAO GIỜ có mặt trong combo."""
    rng = np.random.default_rng(5)
    deep = _sig(
        "rank(ts_rank(ts_mean(ts_std_dev(ts_delta(close, 1), 5), 5), 5))",  # depth 6
        rng.normal(size=200), 10.0,
    )
    a, b = _two_uncorrelated()  # depth 4 -> vào bình thường (== ngưỡng, không bị loại oan)
    combined_expr = build_combined_expression([a.expr, b.expr]).expr
    score_fn = _scorer({combined_expr: 2.0})

    out = combine_stage([deep, a, b], score_fn, tau=0.5, n_min=2, n_max=2, max_combos=1)

    assert len(out) == 1
    assert out[0].expr == combined_expr  # combo dựng từ a,b — "deep" KHÔNG tham gia


def test_score_fn_cu_van_dung_khi_khong_co_factory():
    """Tương thích ngược: không truyền score_fn_factory -> dùng score_fn cũ như trước Fix 2."""
    sigs = _two_uncorrelated()
    combined_expr = build_combined_expression([s.expr for s in sigs]).expr
    score_fn = _scorer({combined_expr: 2.0})
    out = combine_stage(sigs, score_fn, tau=0.5, n_min=2, n_max=2, max_combos=1)
    assert len(out) == 1
    assert out[0].expr == combined_expr
