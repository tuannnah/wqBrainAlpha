from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime

from src.reporting.run_alpha_log import COLUMNS, RunAlphaLogger, run_log_path


@dataclass
class _FakeOutcome:
    expr: str
    passed: bool
    sharpe: float | None = None      # brain sharpe
    fitness: float | None = None     # brain fitness
    turnover: float | None = None
    self_corr: float | None = None
    sims_used: int = 1
    stop_reason: str = ""
    power_pool_eligible: bool = False
    wq_alpha_id: str | None = None
    sim_settings: dict | None = None
    source: str | None = None
    # Pha 0
    stage_reached: str = ""
    fail_check: str = ""
    family: str = ""
    expr_depth: int | None = None
    gen_ms: float | None = None
    backtest_ms: float | None = None
    sim_ms: float | None = None
    dedup_key: str | None = None
    local_sharpe: float | None = None
    # Task 3 (spec C2): pre-sim reject trung thực
    presim_reason: str | None = None
    is_brain_sim: bool = True


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def test_header_ghi_ngay_khi_khoi_tao(tmp_path):
    p = tmp_path / "a.csv"
    RunAlphaLogger(p)
    rows = _read(p)
    assert rows[0] == COLUMNS


def test_schema_co_du_cot_pha0():
    """Schema cố định phải chứa các cột chẩn đoán Pha 0, tách local/brain."""
    for col in (
        "stage_reached", "fail_check", "family", "expr_depth",
        "gen_ms", "backtest_ms", "sim_ms", "dedup_key",
        "local_sharpe", "brain_sharpe", "brain_fitness",
    ):
        assert col in COLUMNS, f"thiếu cột {col}"
    # KHÔNG còn cột gộp mơ hồ tên "sharpe"/"fitness" (đã tách thành local_/brain_).
    assert "sharpe" not in COLUMNS
    assert "fitness" not in COLUMNS


def test_moi_dong_du_18_plus_cot(tmp_path):
    """Mọi dòng luôn có đúng len(COLUMNS) field, kể cả gate 0-sim (không lệch cột)."""
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(1, _FakeOutcome(expr="a", passed=True, sharpe=1.0, local_sharpe=0.9))
    lg.log(2, _FakeOutcome(expr="b", passed=False, sims_used=0, stop_reason="local_floor"))
    rows = _read(p)
    assert len(rows) == 3  # header + 2
    assert all(len(r) == len(COLUMNS) for r in rows)


def test_passed_ghi_du_metrics(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(1, _FakeOutcome(
        expr="rank(close)", passed=True, sharpe=1.5, fitness=1.1, turnover=0.3,
        self_corr=0.2, power_pool_eligible=True, wq_alpha_id="wq1", source="alt_data",
        stage_reached="passed", family="pv_reversal", expr_depth=3, local_sharpe=1.2,
        sim_settings={"region": "USA", "universe": "TOP1000", "delay": 1,
                      "neutralization": "STATISTICAL", "decay": 3, "truncation": 0.02},
    ))
    rows = _read(p)
    d = dict(zip(rows[0], rows[1]))
    assert d["status"] == "passed"
    assert d["brain_sharpe"] == "1.5" and d["brain_fitness"] == "1.1"
    assert d["local_sharpe"] == "1.2"
    assert d["universe"] == "TOP1000" and d["neutralization"] == "STATISTICAL"
    assert d["source"] == "alt_data" and d["expression"] == "rank(close)"
    assert d["stage_reached"] == "passed" and d["family"] == "pv_reversal"
    assert d["expr_depth"] == "3"


def test_failed_van_ghi_brain_metric(tmp_path):
    """Spec Pha 0: LUÔN điền đủ — sim đã chạy thì brain metric phải ghi dù passed=False,
    để phân biệt 'sim rồi trượt' với 'chưa sim' (mấu chốt log cũ nuốt sharpe khi failed)."""
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(2, _FakeOutcome(
        expr="rank(open)", passed=False, sharpe=1.04, fitness=0.71, turnover=0.28,
        self_corr=0.63, stop_reason="local_tuned", source="gp_local_tuner",
        stage_reached="simmed", fail_check="LOW_FITNESS", family="pv_reversal",
        sim_settings={"region": "USA", "universe": "TOP1000", "delay": 1,
                      "neutralization": "CROWDING", "decay": 0, "truncation": 0.08},
    ))
    d = dict(zip(*_read(p)))
    assert d["status"] == "failed"
    assert d["brain_sharpe"] == "1.04" and d["brain_fitness"] == "0.71"  # KHÔNG nuốt
    assert d["turnover"] == "0.28"
    assert d["neutralization"] == "CROWDING"
    assert d["fail_check"] == "LOW_FITNESS" and d["stage_reached"] == "simmed"


def test_gated_0sim_khong_co_setting(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(3, _FakeOutcome(
        expr="multiply(-1, ts_mean(close, 5))", passed=False, sims_used=0,
        stop_reason="local_floor", sim_settings=None, source=None,
        stage_reached="local_floor", fail_check="LOW_SHARPE", local_sharpe=0.31,
    ))
    d = dict(zip(*_read(p)))
    assert d["status"] == "failed"
    assert d["universe"] == "" and d["neutralization"] == ""
    assert d["brain_sharpe"] == ""          # chưa sim -> brain trống
    assert d["local_sharpe"] == "0.31"      # local có -> ghi
    assert d["expression"] == "multiply(-1, ts_mean(close, 5))"
    assert d["stop_reason"] == "local_floor" and d["fail_check"] == "LOW_SHARPE"


def test_run_log_path_theo_timestamp():
    p = run_log_path(datetime(2026, 7, 9, 16, 20))
    assert p.name == "alphas_2026-07-09_162000.csv"
    assert p.parent.name == "logs"


def test_schema_co_cot_presim_reason_va_is_brain_sim():
    """Task 3 (spec C2): CSV phải phân biệt được pre-sim reject (chưa chạm Brain) khỏi sim
    thật rớt — 2 cột mới presim_reason/is_brain_sim."""
    assert "presim_reason" in COLUMNS
    assert "is_brain_sim" in COLUMNS


def test_presim_reject_ghi_dung_cot(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(1, _FakeOutcome(
        expr="fake_op(close)", passed=False, sims_used=0, stop_reason="presim_reject",
        stage_reached="op_invalid", fail_check="OPERATOR_INVALID", family="other",
        presim_reason="Operator không tồn tại: fake_op", is_brain_sim=False,
    ))
    d = dict(zip(*_read(p)))
    assert d["presim_reason"] == "Operator không tồn tại: fake_op"
    assert d["is_brain_sim"] == "False"
    assert d["stage_reached"] == "op_invalid" and d["fail_check"] == "OPERATOR_INVALID"
    assert d["brain_sharpe"] == ""  # chưa sim -> vẫn trống


def test_sim_that_ghi_is_brain_sim_true(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(1, _FakeOutcome(
        expr="rank(close)", passed=True, sharpe=1.5, sims_used=1,
        stage_reached="passed", is_brain_sim=True,
    ))
    d = dict(zip(*_read(p)))
    assert d["is_brain_sim"] == "True"
