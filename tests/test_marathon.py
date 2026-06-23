"""Driver marathon: lặp nhiều hướng nghiên cứu cho tới khi hết quota. Lỗi tạm thời
-> retry rồi bỏ hướng; QuotaExhaustedError -> dừng hẳn."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm.errors import QuotaExhaustedError
from src.llm.marathon import MarathonReport, run_marathon


def _result(sims=3, zoo=1):
    return SimpleNamespace(sims_used=sims, zoo_added=zoo, stop_reason="patience")


class _Provider:
    """Trả lần lượt direction; nếu phần tử là Exception thì raise."""

    def __init__(self, items):
        self.items = list(items)
        self.i = 0

    def __call__(self):
        item = self.items[self.i] if self.i < len(self.items) else "fallback"
        self.i += 1
        if isinstance(item, Exception):
            raise item
        return item


class _Runner:
    """run_direction giả: mỗi lần gọi trả/raise theo hàng đợi outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.i = 0
        self.directions = []

    def __call__(self, direction):
        self.directions.append(direction)
        outcome = self.outcomes[self.i] if self.i < len(self.outcomes) else _result()
        self.i += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# ------------------------------------------------------------------- dừng quota
def test_marathon_provider_het_quota_dung_ngay():
    provider = _Provider([QuotaExhaustedError("hết quota")])
    runner = _Runner([])
    report = run_marathon(provider, runner)
    assert report.stop_reason == "quota"
    assert report.directions_completed == 0
    assert runner.directions == []  # chưa chạy hướng nào


def test_marathon_run_direction_het_quota_dung():
    provider = _Provider(["d1"])
    runner = _Runner([QuotaExhaustedError("hết quota giữa chừng")])
    report = run_marathon(provider, runner)
    assert report.stop_reason == "quota"
    assert report.directions_completed == 0


# --------------------------------------------------------------- nhiều hướng
def test_marathon_chay_nhieu_huong_roi_het_quota():
    provider = _Provider(["d1", "d2", "d3"])
    runner = _Runner([_result(sims=3, zoo=1), _result(sims=5, zoo=2),
                      QuotaExhaustedError("hết quota")])
    report = run_marathon(provider, runner)
    assert report.directions_completed == 2
    assert report.total_sims == 8
    assert report.total_zoo_added == 3
    assert report.stop_reason == "quota"


# ---------------------------------------------------------- retry lỗi tạm thời
def test_marathon_retry_loi_tam_roi_thanh_cong():
    provider = _Provider(["d1", QuotaExhaustedError("stop")])
    # d1: lỗi tạm 2 lần rồi thành công (max_retries=2 cho phép)
    runner = _Runner([RuntimeError("timeout"), RuntimeError("timeout"), _result(sims=4, zoo=1)])
    report = run_marathon(provider, runner, max_retries=2)
    assert report.directions_completed == 1
    assert report.directions_skipped == 0
    assert report.total_sims == 4


def test_marathon_loi_tam_vuot_retry_thi_bo_huong():
    provider = _Provider(["d1", "d2", QuotaExhaustedError("stop")])
    # d1: lỗi tạm 3 lần (vượt max_retries=2) -> bỏ; d2 thành công
    runner = _Runner([
        RuntimeError("e"), RuntimeError("e"), RuntimeError("e"),
        _result(sims=2, zoo=1),
    ])
    report = run_marathon(provider, runner, max_retries=2)
    assert report.directions_skipped == 1
    assert report.directions_completed == 1
    assert report.stop_reason == "quota"
    assert runner.directions[-1] == "d2"


# ------------------------------------------------------------------- on_event
def test_marathon_phat_su_kien():
    provider = _Provider(["d1", QuotaExhaustedError("stop")])
    runner = _Runner([_result()])
    events = []
    run_marathon(provider, runner, on_event=lambda kind, direction, payload: events.append(kind))
    assert "done" in events


def test_report_mac_dinh_rong():
    r = MarathonReport()
    assert r.directions_completed == 0 and r.total_sims == 0 and r.stop_reason == ""


def test_marathon_loi_khac_runtimeerror_thi_propagate():
    """Lỗi không phải RuntimeError (vd bug lập trình) -> không nuốt, ném ra ngoài."""
    provider = _Provider(["d1"])
    runner = _Runner([ValueError("bug")])
    with pytest.raises(ValueError):
        run_marathon(provider, runner)
