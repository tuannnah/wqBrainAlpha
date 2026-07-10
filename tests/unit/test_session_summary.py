"""Pha 0: báo cáo cuối phiên (IMPROVEMENT_SPEC §3 Pha 0).

Acceptance: sau 1 phiên, summary trả lời được "ứng viên chết ở đâu (funnel theo
stage_reached), vì sao (phân bố fail_check), thuộc họ nào (phân bố family), tốn bao
lâu (median thời gian mỗi stage), và bao nhiêu ứng viên trùng bị chặn".
"""

from __future__ import annotations

from dataclasses import dataclass

from src.reporting.session_summary import SessionSummary


@dataclass
class _O:
    stage_reached: str = ""
    fail_check: str = ""
    family: str = ""
    passed: bool = False
    sims_used: int = 0
    gen_ms: float | None = None
    backtest_ms: float | None = None
    sim_ms: float | None = None


def test_funnel_dem_theo_stage():
    s = SessionSummary()
    s.record(_O(stage_reached="local_floor", fail_check="LOW_SHARPE", family="pv_reversal"))
    s.record(_O(stage_reached="local_floor", fail_check="LOW_SHARPE", family="pv_reversal"))
    s.record(_O(stage_reached="simmed", fail_check="SELF_CORR", family="momentum", sims_used=1))
    s.record(_O(stage_reached="passed", family="options_iv", passed=True, sims_used=1))
    d = s.as_dict()
    assert d["total"] == 4
    assert d["by_stage"]["local_floor"] == 2
    assert d["by_stage"]["simmed"] == 1
    assert d["by_stage"]["passed"] == 1


def test_phan_bo_fail_check_va_family():
    s = SessionSummary()
    s.record(_O(stage_reached="local_floor", fail_check="LOW_SHARPE", family="pv_reversal"))
    s.record(_O(stage_reached="simmed", fail_check="SELF_CORR", family="pv_reversal", sims_used=1))
    s.record(_O(stage_reached="passed", family="momentum", passed=True, sims_used=1))
    d = s.as_dict()
    assert d["by_fail_check"]["LOW_SHARPE"] == 1
    assert d["by_fail_check"]["SELF_CORR"] == 1
    assert d["by_family"]["pv_reversal"] == 2
    assert d["by_family"]["momentum"] == 1


def test_dup_chan_dem_rieng():
    s = SessionSummary()
    s.record_dup_blocked()
    s.record_dup_blocked()
    s.record(_O(stage_reached="passed", passed=True, sims_used=1))
    d = s.as_dict()
    assert d["dup_blocked"] == 2
    assert d["total"] == 1  # dup KHÔNG tính vào total ứng viên đã refine


def test_median_thoi_gian_moi_stage():
    s = SessionSummary()
    s.record(_O(stage_reached="passed", gen_ms=10.0, backtest_ms=100.0, sim_ms=8000.0, sims_used=1))
    s.record(_O(stage_reached="passed", gen_ms=30.0, backtest_ms=300.0, sim_ms=2000.0, sims_used=1))
    d = s.as_dict()
    assert d["median_ms"]["gen"] == 20.0
    assert d["median_ms"]["backtest"] == 200.0
    assert d["median_ms"]["sim"] == 5000.0


def test_render_markdown_co_cac_muc():
    s = SessionSummary()
    s.record(_O(stage_reached="passed", family="momentum", passed=True, sims_used=1))
    s.record_dup_blocked()
    md = s.render_markdown()
    assert "# Tóm tắt phiên" in md
    assert "Funnel" in md
    assert "passed" in md
    assert "momentum" in md


def test_ghi_file(tmp_path):
    s = SessionSummary()
    s.record(_O(stage_reached="passed", passed=True, sims_used=1))
    p = tmp_path / "session_summary.md"
    s.write(p)
    assert p.exists()
    assert "Tóm tắt phiên" in p.read_text(encoding="utf-8")


def test_funnel_dem_bucket_op_invalid_va_field_invalid():
    """Task 3 (spec C2): pre-sim reject giờ có stage_reached riêng (op_invalid/field_invalid)
    thay vì lẫn vào 'simmed' -> funnel phải đếm được 2 bucket này."""
    s = SessionSummary()
    s.record(_O(stage_reached="op_invalid", fail_check="OPERATOR_INVALID", family="other"))
    s.record(_O(stage_reached="op_invalid", fail_check="OPERATOR_INVALID", family="other"))
    s.record(_O(stage_reached="field_invalid", fail_check="FIELD_INVALID", family="other"))
    d = s.as_dict()
    assert d["by_stage"]["op_invalid"] == 2
    assert d["by_stage"]["field_invalid"] == 1
    assert d["by_fail_check"]["OPERATOR_INVALID"] == 2
    assert d["by_fail_check"]["FIELD_INVALID"] == 1


def test_summary_rong_khong_vo():
    """Phiên 0 ứng viên (log 230943 thực tế) vẫn render được, không chia cho 0."""
    s = SessionSummary()
    d = s.as_dict()
    assert d["total"] == 0
    md = s.render_markdown()
    assert "Tóm tắt phiên" in md
