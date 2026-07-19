"""Test CalibrationHarness với fake LocalScorer xác định -> Spearman rho biết trước.

Không cần data/parser thật: tiêm fake scorer (expr -> LocalScore) để kiểm tra logic
gom mẫu, loại None, tính spearman/self_corr_agreement/decile_hit_rate/by_year.
"""

from __future__ import annotations

import math

import pytest

from src.calibration.harness import CalibrationHarness, LocalScore
from src.calibration.loader import BrainRecord


def _records() -> list[BrainRecord]:
    # 5 alpha: brain_sharpe tăng dần 1..5; local sẽ được fake scorer gán tương tự (rho~1.0).
    # alpha chẵn (2,4) có brain_self_corr=0.5; alpha lẻ có brain_self_corr=None (chưa nộp).
    return [
        BrainRecord(
            expr_string=f"e{i}",
            brain_sharpe=float(i),
            brain_fitness=float(i) * 0.8,
            brain_turnover=0.3,
            brain_self_corr=0.5 if i % 2 == 0 else None,
        )
        for i in range(1, 6)
    ]


def test_perfect_local_scorer_gives_rho_one() -> None:
    records = _records()

    def fake_scorer(expr: str) -> LocalScore | None:
        i = int(expr[1:])
        return LocalScore(
            sharpe=float(i), fitness=float(i) * 0.8, self_corr=0.5,
            per_year_sharpe={2023: float(i)},
        )

    report = CalibrationHarness(scorer=fake_scorer).run(records)
    assert report.n == 5
    assert report.spearman_sharpe == pytest.approx(1.0)
    assert report.spearman_fitness == pytest.approx(1.0)


def test_inverted_local_scorer_gives_rho_minus_one() -> None:
    records = _records()

    def fake_scorer(expr: str) -> LocalScore | None:
        i = int(expr[1:])
        return LocalScore(
            sharpe=float(6 - i), fitness=float(6 - i), self_corr=None, per_year_sharpe={},
        )

    report = CalibrationHarness(scorer=fake_scorer).run(records)
    assert report.spearman_sharpe == pytest.approx(-1.0)


def test_none_from_scorer_excludes_record_from_n() -> None:
    records = _records()

    def fake_scorer(expr: str) -> LocalScore | None:
        if expr == "e3":
            return None  # giả lập parse lỗi / không re-score được local
        i = int(expr[1:])
        return LocalScore(sharpe=float(i), fitness=float(i), self_corr=None, per_year_sharpe={})

    report = CalibrationHarness(scorer=fake_scorer).run(records)
    assert report.n == 4


def test_self_corr_agreement_counts_pairs_where_both_sides_agree_on_gate() -> None:
    records = _records()  # alpha 2,4 có brain_self_corr=0.5 (< 0.70 -> "qua gate")

    def fake_scorer(expr: str) -> LocalScore | None:
        i = int(expr[1:])
        # local_self_corr: đồng ý với brain cho alpha 2 (cùng < 0.70), bất đồng cho alpha 4
        local_corr = 0.9 if expr == "e4" else 0.5
        return LocalScore(sharpe=float(i), fitness=float(i), self_corr=local_corr, per_year_sharpe={})

    report = CalibrationHarness(scorer=fake_scorer).run(records)
    # chỉ alpha 2,4 có brain_self_corr khác None -> 2 cặp; alpha2 đồng ý (0.5<0.7 cả hai phía),
    # alpha4 bất đồng (local 0.9>=0.7 nhưng brain 0.5<0.7) -> agreement = 1/2.
    assert report.self_corr_agreement == pytest.approx(0.5)


def test_by_year_averages_local_per_year_sharpe() -> None:
    records = _records()

    def fake_scorer(expr: str) -> LocalScore | None:
        i = int(expr[1:])
        # mọi alpha có năm 2023 = giá trị riêng; alpha chẵn thêm năm 2024 = 10.0
        pys = {2023: float(i)}
        if i % 2 == 0:
            pys[2024] = 10.0
        return LocalScore(sharpe=float(i), fitness=float(i), self_corr=None, per_year_sharpe=pys)

    report = CalibrationHarness(scorer=fake_scorer).run(records)
    # 2023: mean(1,2,3,4,5)=3.0 ; 2024: chỉ alpha 2,4 -> mean(10,10)=10.0
    assert report.by_year[2023] == pytest.approx(3.0)
    assert report.by_year[2024] == pytest.approx(10.0)


def test_empty_records_returns_report_with_n_zero() -> None:
    report = CalibrationHarness(scorer=lambda expr: None).run([])
    assert report.n == 0
    assert math.isnan(report.spearman_sharpe)
    assert math.isnan(report.spearman_fitness)
    assert math.isnan(report.spearman_submit_score)
    assert math.isnan(report.self_corr_agreement)
    assert report.by_year == {}


def test_spearman_submit_score_bat_duoc_lech_ma_spearman_sharpe_bo_lo() -> None:
    # T4.1: brain sharpe VÀ fitness đều tăng dần đều (_records(): fitness = sharpe*0.8) ->
    # brain submit_score cũng tăng dần đều. Local khớp Y HỆT brain ở sharpe (rho_sharpe=1.0)
    # NHƯNG alpha 5 có fitness local SẬP xuống 0.1 (vd overfit turnover cao làm fitness tệ dù
    # sharpe cao) -> submit_score (min sharpe/REF, fitness/REF) của alpha 5 sập theo trong khi
    # sharpe thô không phản ánh gì -> spearman_submit_score PHẢI khác (thấp hơn) spearman_sharpe.
    records = _records()

    def fake_scorer(expr: str) -> LocalScore | None:
        i = int(expr[1:])
        fitness = 0.1 if i == 5 else float(i) * 0.8
        return LocalScore(sharpe=float(i), fitness=fitness, self_corr=None, per_year_sharpe={})

    report = CalibrationHarness(scorer=fake_scorer).run(records)
    assert report.spearman_sharpe == pytest.approx(1.0)
    assert report.spearman_submit_score < report.spearman_sharpe


def test_spearman_submit_score_hoan_hao_khi_local_khop_brain_ca_hai_truc() -> None:
    # Local khớp y hệt cả sharpe lẫn fitness Brain -> submit_score local == submit_score
    # Brain từng cặp -> rho = 1.0 (không chỉ rho_sharpe).
    records = _records()

    def fake_scorer(expr: str) -> LocalScore | None:
        i = int(expr[1:])
        return LocalScore(
            sharpe=float(i), fitness=float(i) * 0.8, self_corr=None, per_year_sharpe={},
        )

    report = CalibrationHarness(scorer=fake_scorer).run(records)
    assert report.spearman_submit_score == pytest.approx(1.0)


def test_spearman_submit_score_nan_khi_thieu_brain_sharpe_hoac_fitness() -> None:
    # Record có brain_sharpe=None -> submit_score Brain không tính được (None-safe: trả NaN,
    # KHÔNG dựa vào min() vốn lệ thuộc thứ tự tham số khi có NaN — loại khỏi mẫu rho thay vì
    # âm thầm giữ lại một số bịa).
    records = [
        BrainRecord(
            expr_string="e1", brain_sharpe=None, brain_fitness=0.5,
            brain_turnover=0.3, brain_self_corr=None,
        ),
        BrainRecord(
            expr_string="e2", brain_sharpe=1.0, brain_fitness=None,
            brain_turnover=0.3, brain_self_corr=None,
        ),
    ]

    def fake_scorer(expr: str) -> LocalScore | None:
        return LocalScore(sharpe=1.0, fitness=1.0, self_corr=None, per_year_sharpe={})

    report = CalibrationHarness(scorer=fake_scorer).run(records)
    assert report.n == 2
    assert math.isnan(report.spearman_submit_score)  # <2 cặp hợp lệ sau khi loại NaN
