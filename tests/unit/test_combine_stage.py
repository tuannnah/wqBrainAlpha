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
    sharpe: float = 0.5  # Fix 4: điểm-nộp cần cả sharpe; mặc định = fallback fitness cũ.


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
    """Trả score_fn tra `value` theo expr (expr lạ -> mặc định 0.5), gán CẢ fitness lẫn
    sharpe = value (Fix 4: điểm-nộp = min(sharpe/SUBMIT_SHARPE_REF, fitness/SUBMIT_FITNESS_REF)
    — đặt sharpe=fitness giữ nguyên thứ tự so sánh "value cao hơn -> điểm-nộp cao hơn" mà các
    test dominance cũ đã giả định, không cần sửa lại kỳ vọng của chúng)."""

    def score_fn(expr: str) -> _FakeScore:
        value = fitness_map.get(expr, 0.5)
        return _FakeScore(
            _FakeMetrics(fitness=value, sharpe=value), _FakeVerdict(passed), np.zeros(200),
            DATES.copy(),
        )

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
            # sharpe=fit (như `_scorer`): điểm-nộp (Fix 4) tỉ lệ thuận fitness ở đây, giữ
            # nguyên ý định gốc của test này (combo "mạnh hơn" component -> điểm-nộp cao hơn).
            return _FakeScore(
                _FakeMetrics(fitness=fit, sharpe=fit), _FakeVerdict(True), np.zeros(200),
                DATES.copy(),
            )

        return score_fn

    out = combine_stage(
        all_sigs, _scorer({}),  # score_fn cũ truyền vào nhưng KHÔNG được dùng vì có factory
        tau=0.5, n_min=2, n_max=2, max_combos=1, score_fn_factory=factory,
    )

    assert len(out) == 1  # factory trả fitness combo=2.0 > best_component=0.5 -> giữ
    assert len(received) == 1  # đúng 1 combo -> factory gọi đúng 1 lần
    assert {s.expr for s in received[0]} == {c.expr}  # pool CHỈ chứa C (loại A,B = combo)


def test_score_fn_factory_loai_ban_sao_expr_ngoai_combo_khoi_pool():
    """Finding #3 (Important): combine_stage loại thành viên combo khỏi pool bằng id() —
    nếu `signals` chứa CÙNG một expr 2 bản (vd 1 bản "run" của phiên hiện tại + 1 bản "db"
    nạp lại từ kho, cùng expr nhưng khác object) thì greedy KHÔNG cho 2 bản vào cùng combo
    (rho pnl ~1.0 >= tau -> tự loại), nhưng bản sao còn lại (id khác) vẫn lọt vào pool
    "others" của score_fn_factory với |rho|≈1 so với chính combo -> gate tự-so giết oan combo
    (đúng loại tự-so mà Fix 2 phải khử). Sau fix, others phải loại theo CHUỖI expr."""
    a, b = _two_uncorrelated()
    # a_dup: CÙNG expr + CÙNG pnl với a (mô phỏng bản ghi trùng từ nguồn "db"), khác object/id,
    # điểm hơi thấp hơn a để không đổi thứ tự seed nhưng vẫn cao hơn b (ranked: a > a_dup > b).
    a_dup = _sig(a.expr, a.pnl.copy(), score=a.score - 0.05)
    all_sigs = [a, b, a_dup]
    combined_expr = build_combined_expression([a.expr, b.expr]).expr

    received: list[list[SubSignal]] = []

    def factory(others: list[SubSignal]):
        received.append(others)

        def score_fn(expr: str) -> _FakeScore:
            fit = 2.0 if expr == combined_expr else 0.5
            return _FakeScore(
                _FakeMetrics(fitness=fit, sharpe=fit), _FakeVerdict(True), np.zeros(200),
                DATES.copy(),
            )

        return score_fn

    out = combine_stage(
        all_sigs, _scorer({}),
        tau=0.5, n_min=2, n_max=2, max_combos=1, score_fn_factory=factory,
    )

    assert len(out) == 1
    assert len(received) == 1
    pool_exprs = {s.expr for s in received[0]}
    # Bản sao a_dup (cùng expr với a, thành viên combo) KHÔNG được lọt vào pool.
    assert a.expr not in pool_exprs


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


# ------------------------- T1.2: thử N nhỏ hơn khi N lớn không lọt -------------------------
# 2 tín hiệu depth=5 KHÔNG lọt cap(n_max=4)=cap(n_max=3)=4 nhưng LỌT cap(n_max=2)=5
# (component_depth_cap) -- combine_stage phải tự thử lại greedy với n_max nhỏ hơn (4->3->2)
# thay vì chỉ bỏ tín hiệu điểm thấp cuối trong build_combined_expression.


def test_thu_n_max_nho_hon_khi_n_max_lon_khong_lot_tran():
    rng = np.random.default_rng(7)
    deep = [
        _sig(
            "ts_rank(ts_mean(ts_std_dev(ts_delta(close, 1), 5), 5), 5)",  # depth 5
            rng.normal(size=200), 1.0,
        ),
        _sig(
            "ts_rank(ts_mean(ts_std_dev(ts_delta(volume, 1), 5), 5), 5)",  # depth 5
            rng.normal(size=200), 0.9,
        ),
    ]
    combined_expr = build_combined_expression([s.expr for s in deep]).expr
    score_fn = _scorer({combined_expr: 2.0})

    out = combine_stage(deep, score_fn, tau=0.5, n_min=2, n_max=4, max_combos=1)

    assert len(out) == 1
    assert out[0].expr == combined_expr


def test_drop_stats_dem_greedy_empty_moi_lan_thu_n_max_khong_lot():
    """2 tín hiệu depth=5 -> greedy_empty ở CẢ n_max=4 lẫn n_max=3 (cap cùng =4), chỉ lọt ở
    n_max=2 (cap=5) -> drop_stats phải ghi nhận đúng 2 lần greedy_empty trước khi thành công."""
    rng = np.random.default_rng(7)
    deep = [
        _sig(
            "ts_rank(ts_mean(ts_std_dev(ts_delta(close, 1), 5), 5), 5)",
            rng.normal(size=200), 1.0,
        ),
        _sig(
            "ts_rank(ts_mean(ts_std_dev(ts_delta(volume, 1), 5), 5), 5)",
            rng.normal(size=200), 0.9,
        ),
    ]
    combined_expr = build_combined_expression([s.expr for s in deep]).expr
    score_fn = _scorer({combined_expr: 2.0})
    stats: dict[str, int] = {}

    out = combine_stage(
        deep, score_fn, tau=0.5, n_min=2, n_max=4, max_combos=1, drop_stats=stats,
    )

    assert len(out) == 1
    assert stats == {"greedy_empty": 2}


def test_score_fn_cu_van_dung_khi_khong_co_factory():
    """Tương thích ngược: không truyền score_fn_factory -> dùng score_fn cũ như trước Fix 2."""
    sigs = _two_uncorrelated()
    combined_expr = build_combined_expression([s.expr for s in sigs]).expr
    score_fn = _scorer({combined_expr: 2.0})
    out = combine_stage(sigs, score_fn, tau=0.5, n_min=2, n_max=2, max_combos=1)
    assert len(out) == 1
    assert out[0].expr == combined_expr


# ------------------------- Fix 4: instrument + điểm-nộp -------------------------
# drop_stats đếm tại 3 điểm `continue` (depth/gate/not_better) + `greedy_empty`. Tiêu chí
# vượt trội đổi từ so fitness thô sang điểm-nộp min(sharpe/SUBMIT_SHARPE_REF,
# fitness/SUBMIT_FITNESS_REF) — combo chỉ đáng giữ nếu nó tiến GẦN NGƯỠNG NỘP hơn thành
# phần mạnh nhất, không chỉ đơn thuần "fitness thô cao hơn".


def test_drop_stats_dem_dung_greedy_empty():
    stats: dict[str, int] = {}
    out = combine_stage([], _scorer({}), tau=0.5, n_min=2, n_max=2, max_combos=1, drop_stats=stats)
    assert out == []
    assert stats == {"greedy_empty": 1}


def test_drop_stats_dem_dung_depth():
    # Ép max_depth cực nhỏ (khác COMBINER_MAX_COMPONENT_DEPTH — đây là trần combo SAU khi
    # build, không phải trần từng component) để build_combined_expression chắc chắn thất bại.
    sigs = _two_uncorrelated()
    stats: dict[str, int] = {}
    out = combine_stage(
        sigs, _scorer({}), tau=0.5, n_min=2, n_max=2, max_combos=1, max_depth=1, drop_stats=stats,
    )
    assert out == []
    assert stats == {"depth": 1}


def test_drop_stats_dem_dung_gate():
    sigs = _two_uncorrelated()
    stats: dict[str, int] = {}
    out = combine_stage(
        sigs, _scorer({}, passed=False), tau=0.5, n_min=2, n_max=2, max_combos=1, drop_stats=stats,
    )
    assert out == []
    assert stats == {"gate": 1}


def test_drop_stats_dem_dung_not_better():
    sigs = _two_uncorrelated()
    stats: dict[str, int] = {}
    # mọi expr (kể cả combo) -> value=0.5 mặc định -> điểm-nộp combo == điểm-nộp component
    # mạnh nhất -> không vượt trội -> "not_better".
    out = combine_stage(
        sigs, _scorer({}), tau=0.5, n_min=2, n_max=2, max_combos=1, drop_stats=stats,
    )
    assert out == []
    assert stats == {"not_better": 1}


def test_diem_nop_thay_fitness_tho_khi_so_vuot_troi():
    """Combo fitness THÔ cao hơn component nhưng sharpe THẤP (điểm-nộp thấp hơn) -> vẫn bị
    loại — chứng minh tiêu chí đã đổi sang điểm-nộp, không còn so fitness thô đơn thuần."""
    sigs = _two_uncorrelated()
    combined_expr = build_combined_expression([s.expr for s in sigs]).expr

    def score_fn(expr: str) -> _FakeScore:
        if expr == combined_expr:
            # fitness thô 5.0 (>> 0.5 của component) nhưng sharpe chỉ 0.1 -> điểm-nộp
            # min(0.1/1.25, 5.0/1.0) = 0.08, THẤP hơn điểm-nộp component min(0.5/1.25,
            # 0.5/1.0)=0.4 -> KHÔNG vượt trội dù fitness thô cao hơn hẳn.
            return _FakeScore(
                _FakeMetrics(fitness=5.0, sharpe=0.1), _FakeVerdict(True), np.zeros(200),
                DATES.copy(),
            )
        return _FakeScore(
            _FakeMetrics(fitness=0.5, sharpe=0.5), _FakeVerdict(True), np.zeros(200), DATES.copy(),
        )

    stats: dict[str, int] = {}
    out = combine_stage(
        sigs, score_fn, tau=0.5, n_min=2, n_max=2, max_combos=1, drop_stats=stats,
    )
    assert out == []
    assert stats == {"not_better": 1}
