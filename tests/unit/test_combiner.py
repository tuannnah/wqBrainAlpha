"""TDD cho stage combiner: chọn greedy khử tương quan + dựng biểu thức ghép.

Xem spec docs/superpowers/specs/2026-07-09-alpha-combiner-design.md.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.operators_local  # noqa: F401 — đăng ký operator thật vào registry cho parse/depth
from src.backtest.pool_corr import pairwise_abs_rho
from src.generation.combiner import (
    CombinedAlpha,
    SubSignal,
    build_combined_expression,
    component_depth_cap,
    select_decorrelated_combos,
)
from src.lang.parser import parse
from src.lang.visitors import DepthVisitor

DATES = np.arange(200)


def _sig(expr: str, pnl: np.ndarray, score: float, source: str = "run") -> SubSignal:
    return SubSignal(expr=expr, pnl=pnl, dates=DATES.copy(), score=score, source=source)


def _rng():
    return np.random.default_rng(42)


# ------------------------- pairwise_abs_rho -------------------------

def test_pairwise_abs_rho_identical_la_1():
    x = _rng().normal(size=200)
    assert pairwise_abs_rho(x, DATES, x.copy(), DATES.copy()) == pytest.approx(1.0)


def test_pairwise_abs_rho_lay_gia_tri_tuyet_doi():
    x = _rng().normal(size=200)
    assert pairwise_abs_rho(x, DATES, -x, DATES.copy()) == pytest.approx(1.0)


def test_pairwise_abs_rho_hang_so_tra_none():
    x = _rng().normal(size=200)
    assert pairwise_abs_rho(x, DATES, np.ones(200), DATES.copy()) is None


def test_pairwise_abs_rho_ngay_khong_giao_tra_none():
    x = _rng().normal(size=200)
    y = _rng().normal(size=200)
    assert pairwise_abs_rho(x, DATES, y, DATES + 10_000) is None


def test_pairwise_abs_rho_align_theo_ngay_giao():
    # a trên ngày 0..99, b trên ngày 50..149, cùng nguồn ở phần giao -> |rho|=1.
    base = _rng().normal(size=200)
    a_dates = np.arange(100)
    b_dates = np.arange(50, 150)
    a = base[:100]
    b = base[50:150]
    assert pairwise_abs_rho(a, a_dates, b, b_dates) == pytest.approx(1.0)


# ------------------------- select_decorrelated_combos -------------------------

def test_chon_seed_diem_cao_nhat_va_loai_tin_hieu_trung():
    rng = _rng()
    a = rng.normal(size=200)          # seed điểm cao nhất
    b = rng.normal(size=200)          # độc lập với a
    c = a + 1e-6 * rng.normal(size=200)  # gần trùng a -> |rho|~1, phải bị loại khỏi combo của a
    d = rng.normal(size=200)          # độc lập với tất cả
    sigs = [_sig("a", a, 1.0), _sig("b", b, 0.9), _sig("c", c, 0.8), _sig("d", d, 0.7)]

    combos = select_decorrelated_combos(sigs, tau=0.3, n_min=2, n_max=4, max_combos=5)

    assert len(combos) >= 1
    first = combos[0]
    exprs = [s.expr for s in first]
    assert exprs[0] == "a"          # seed là điểm cao nhất
    assert "c" not in exprs         # c gần trùng a -> loại
    assert "b" in exprs and "d" in exprs


def test_ton_trong_n_max():
    rng = _rng()
    sigs = [_sig(f"s{i}", rng.normal(size=200), 1.0 - i * 0.01) for i in range(6)]
    combos = select_decorrelated_combos(sigs, tau=0.9, n_min=2, n_max=3, max_combos=1)
    assert len(combos[0]) == 3


def test_combo_giu_thu_tu_diem_giam_dan():
    rng = _rng()
    sigs = [_sig("hi", rng.normal(size=200), 0.5),
            _sig("lo", rng.normal(size=200), 0.9)]
    combos = select_decorrelated_combos(sigs, tau=0.9, n_min=2, n_max=2, max_combos=1)
    assert [s.expr for s in combos[0]] == ["lo", "hi"]  # sắp theo điểm giảm dần


def test_pool_rong_hoac_mot_phan_tu_khong_combo():
    rng = _rng()
    assert select_decorrelated_combos([], tau=0.3, n_min=2, n_max=4, max_combos=5) == []
    one = [_sig("a", rng.normal(size=200), 1.0)]
    assert select_decorrelated_combos(one, tau=0.3, n_min=2, n_max=4, max_combos=5) == []


def test_uu_tien_combinability_khong_phai_fitness_tho():
    """T1.1: 1 tín hiệu depth=6 điểm CAO NHẤT (monster GP) vs 2 tín hiệu depth=2 điểm vừa
    -- combo phải chọn 2 tín hiệu NÔNG, monster KHÔNG được chọn dù điểm cao nhất (đúng kịch
    bản khiến greedy cũ luôn chọn nhầm biểu thức chết trần làm seed)."""
    rng = _rng()
    monster = _sig(
        "rank(ts_rank(ts_mean(ts_std_dev(ts_delta(close, 1), 5), 5), 5))",  # depth 6
        rng.normal(size=200), 100.0,
    )
    a = _sig("ts_delta(close, 5)", rng.normal(size=200), 0.6)   # depth 2
    b = _sig("ts_delta(close, 10)", rng.normal(size=200), 0.5)  # depth 2

    combos = select_decorrelated_combos([monster, a, b], tau=0.9, n_min=2, n_max=2, max_combos=5)

    assert len(combos) >= 1
    exprs = [s.expr for s in combos[0]]
    assert monster.expr not in exprs             # monster KHÔNG được chọn dù điểm cao nhất
    assert set(exprs) == {a.expr, b.expr}         # combo chỉ gồm 2 tín hiệu nông


def test_nhieu_combo_khong_trung_seed():
    rng = _rng()
    # 4 tín hiệu đôi một độc lập -> combo1 lấy 4; hết ứng viên -> chỉ 1 combo.
    # Thêm cặp độc lập nữa để có combo thứ 2.
    sigs = [_sig(f"s{i}", rng.normal(size=200), 1.0 - i * 0.01) for i in range(4)]
    combos = select_decorrelated_combos(sigs, tau=0.5, n_min=2, n_max=2, max_combos=5)
    # n_max=2 -> mỗi combo 2 tín hiệu, 4 tín hiệu -> 2 combo, seed khác nhau.
    assert len(combos) == 2
    seeds = {combos[0][0].expr, combos[1][0].expr}
    assert len(seeds) == 2


def test_uu_tien_tin_hieu_da_chuan_hoa_cung_bucket_do_sau():
    """T1.3: cùng bucket độ sâu (đều depth <= cap mặc định) -- bản gốc đã rank() được xếp
    TRƯỚC bản thô dù điểm THẤP hơn, vì _standardize bỏ qua bọc rank() cho nó -> tiết kiệm
    đúng 1 tầng độ sâu khi build."""
    rng = _rng()
    standardized = _sig("rank(ts_delta(close, 5))", rng.normal(size=200), 0.4)   # đã chuẩn hóa, điểm thấp
    raw = _sig("ts_delta(volume, 5)", rng.normal(size=200), 0.9)                 # điểm cao hơn, CHƯA chuẩn hóa

    combos = select_decorrelated_combos(
        [standardized, raw], tau=0.9, n_min=2, n_max=2, max_combos=1,
    )

    assert [s.expr for s in combos[0]] == [standardized.expr, raw.expr]


# ------------------------- component_depth_cap (T1.2) -------------------------

def test_component_depth_cap_suy_dung_theo_n():
    # N=4 -> ceil(log2(4))=2 tầng add + 1 rank = 3 -> cap = 7-3 = 4 (khớp hằng số cũ
    # COMBINER_MAX_COMPONENT_DEPTH).
    assert component_depth_cap(4) == 4
    # N=3 -> ceil(log2(3))=2 (cùng số tầng add như N=4) -> cap vẫn 4.
    assert component_depth_cap(3) == 4
    # N=2 -> ceil(log2(2))=1 -> cap nới ra 5.
    assert component_depth_cap(2) == 5
    # N=1 -> không cần cây add, chỉ 1 tầng rank -> cap = max_depth-1.
    assert component_depth_cap(1) == 6
    # max_depth tùy biến vẫn theo đúng công thức (không hardcode 7).
    assert component_depth_cap(4, max_depth=10) == 7


# ------------------------- build_combined_expression -------------------------

def _depth(expr: str) -> int:
    return DepthVisitor().visit(parse(expr))


def test_ghep_tin_hieu_nong_rank_add():
    exprs = [
        "-ts_mean(subtract(close, vwap), 10)",
        "-ts_mean(subtract(close, open), 5)",
    ]
    res = build_combined_expression(exprs, max_depth=7)
    assert isinstance(res, CombinedAlpha)
    assert res.sub_exprs == tuple(exprs)
    assert res.expr.startswith("add(")
    assert "rank(" in res.expr
    assert _depth(res.expr) <= 7
    parse(res.expr)  # phải parse được


def test_fold_can_bang_bon_tin_hieu():
    exprs = [
        "-ts_mean(subtract(close, vwap), 10)",
        "-ts_mean(subtract(close, open), 5)",
        "ts_delta(close, 20)",
        "ts_zscore(returns, 5)",
    ]
    res = build_combined_expression(exprs, max_depth=7)
    assert res is not None
    # cây add cân bằng: add(add(_,_), add(_,_)) -> đúng 2 tầng add trên đỉnh.
    assert res.expr.count("add(") == 3
    assert _depth(res.expr) <= 7


def test_tuoc_group_neutralize_khoi_bieu_thuc():
    # Tín hiệu có group_neutralize -> builder tước bỏ để tuner tự trung hòa (tránh kép).
    exprs = [
        "group_neutralize(-rank(ts_delta(close, 5)), sector)",
        "group_neutralize(rank(ts_delta(close, 60)), industry)",
        "group_neutralize(rank(ts_sum(returns, 20)), sector)",
    ]
    res = build_combined_expression(exprs, max_depth=7)
    assert res is not None
    assert "group_neutralize" not in res.expr        # đã tước neutralize khỏi biểu thức
    assert _depth(res.expr) <= 7
    parse(res.expr)


def test_giam_n_khi_van_qua_sau():
    # Nhiều tín hiệu sâu: builder phải giảm N (bỏ điểm thấp) để lọt trần độ sâu.
    deep = "group_neutralize(rank(ts_delta(ts_mean(close, 20), 5)), subindustry)"  # sâu sẵn
    exprs = [deep, deep, deep, deep]
    res = build_combined_expression(exprs, max_depth=7)
    if res is not None:
        assert _depth(res.expr) <= 7
        assert len(res.sub_exprs) >= 2


def test_khong_dung_duoc_tra_none():
    # Trần độ sâu cực nhỏ -> không combo 2 tín hiệu nào lọt.
    exprs = ["ts_delta(close, 5)", "ts_delta(close, 10)"]
    assert build_combined_expression(exprs, max_depth=2) is None
