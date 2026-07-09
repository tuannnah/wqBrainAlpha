from __future__ import annotations

import csv
from dataclasses import dataclass

from src.reporting.run_alpha_log import COLUMNS, RunAlphaLogger


@dataclass
class _FakeOutcome:
    expr: str
    passed: bool
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    self_corr: float | None = None
    sims_used: int = 1
    stop_reason: str = ""
    power_pool_eligible: bool = False
    wq_alpha_id: str | None = None
    sim_settings: dict | None = None
    source: str | None = None


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def test_header_ghi_ngay_khi_khoi_tao(tmp_path):
    p = tmp_path / "a.csv"
    RunAlphaLogger(p)
    rows = _read(p)
    assert rows[0] == COLUMNS


def test_passed_ghi_du_metrics(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(1, _FakeOutcome(
        expr="rank(close)", passed=True, sharpe=1.5, fitness=1.1, turnover=0.3,
        self_corr=0.2, power_pool_eligible=True, wq_alpha_id="wq1", source="alt_data",
        sim_settings={"region": "USA", "universe": "TOP1000", "delay": 1,
                      "neutralization": "STATISTICAL", "decay": 3, "truncation": 0.02},
    ))
    rows = _read(p)
    d = dict(zip(rows[0], rows[1]))
    assert d["status"] == "passed"
    assert d["sharpe"] == "1.5" and d["fitness"] == "1.1"
    assert d["universe"] == "TOP1000" and d["neutralization"] == "STATISTICAL"
    assert d["source"] == "alt_data" and d["expression"] == "rank(close)"


def test_failed_de_trong_sharpe_fitness_nhung_giu_setting(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(2, _FakeOutcome(
        expr="rank(open)", passed=False, sharpe=1.04, fitness=0.71, turnover=0.28,
        stop_reason="alt_data_direct", source="alt_data",
        sim_settings={"region": "USA", "universe": "TOP1000", "delay": 1,
                      "neutralization": "CROWDING", "decay": 0, "truncation": 0.08},
    ))
    d = dict(zip(*_read(p)))
    assert d["status"] == "failed"
    assert d["sharpe"] == "" and d["fitness"] == ""       # để trống theo yêu cầu
    assert d["turnover"] == "0.28"                         # cột khác vẫn giữ
    assert d["neutralization"] == "CROWDING"


def test_gated_0sim_khong_co_setting(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(3, _FakeOutcome(
        expr="multiply(-1, ts_mean(close, 5))", passed=False, sims_used=0,
        stop_reason="local_floor", sim_settings=None, source=None,
    ))
    d = dict(zip(*_read(p)))
    assert d["status"] == "failed"
    assert d["universe"] == "" and d["neutralization"] == ""
    assert d["expression"] == "multiply(-1, ts_mean(close, 5))"
    assert d["stop_reason"] == "local_floor"


def test_append_nhieu_dong(tmp_path):
    p = tmp_path / "a.csv"
    lg = RunAlphaLogger(p)
    lg.log(1, _FakeOutcome(expr="a", passed=True, sharpe=1.0))
    lg.log(2, _FakeOutcome(expr="b", passed=False))
    rows = _read(p)
    assert len(rows) == 3  # header + 2
