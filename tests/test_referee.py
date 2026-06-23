"""Trọng tài LLM: sau mỗi sim quyết refine_formula | tune_config | abandon, và
ConfigTuner đề xuất decay/truncation/neutralization mới (chốt về giá trị hợp lệ)."""

from __future__ import annotations

from src.llm.referee import (
    ABANDON,
    REFINE_FORMULA,
    TUNE_CONFIG,
    ConfigTuner,
    Referee,
    Verdict,
)
from src.simulation.config import SimConfig
from tests.fakes import FakeDeepSeek


def _history():
    return [
        {"expression": "rank(close)", "total": 0.5},
        {"expression": "rank(ts_mean(close, 5))", "total": 0.52},
    ]


def _metrics():
    return {"sharpe": 1.3, "fitness": 0.9, "turnover": 0.8, "drawdown": 0.1}


# --------------------------------------------------------------------- Referee
def test_referee_doc_action_abandon():
    ds = FakeDeepSeek(['{"action": "abandon", "reason": "hướng cạn ý"}'])
    v = Referee(ds).judge("mean-reversion", _history(), _metrics())
    assert v.action == ABANDON
    assert "cạn" in v.reason


def test_referee_doc_action_tune_config():
    ds = FakeDeepSeek(['{"action": "tune_config", "reason": "turnover cao, tăng decay"}'])
    v = Referee(ds).judge("x", _history(), _metrics())
    assert v.action == TUNE_CONFIG


def test_referee_action_la_thi_default_refine_formula():
    """Action không hợp lệ -> mặc định an toàn refine_formula (trần cứng vẫn chặn loop)."""
    ds = FakeDeepSeek(['{"action": "explode", "reason": "?"}'])
    v = Referee(ds).judge("x", _history(), _metrics())
    assert v.action == REFINE_FORMULA


def test_referee_khong_parse_duoc_thi_refine_formula():
    ds = FakeDeepSeek(["xin chào không phải json"])
    v = Referee(ds).judge("x", _history(), _metrics())
    assert v.action == REFINE_FORMULA


def test_referee_chiu_duoc_json_ban():
    ds = FakeDeepSeek(['Kết luận:\n```json\n{"action": "abandon", "reason": "x"}\n```'])
    v = Referee(ds).judge("x", _history(), _metrics())
    assert v.action == ABANDON


def test_referee_dung_task_referee():
    ds = FakeDeepSeek(['{"action": "refine_formula", "reason": "còn dư địa"}'])
    Referee(ds).judge("x", _history(), _metrics())
    assert ds.tasks == ["referee"]


# ------------------------------------------------------------------ ConfigTuner
def test_config_tuner_ap_dung_gia_tri_hop_le():
    base = SimConfig(region="USA", universe="TOP3000", delay=1,
                     decay=4, truncation=0.01, neutralization="MARKET")
    ds = FakeDeepSeek(['{"decay": 10, "truncation": 0.05, "neutralization": "INDUSTRY"}'])
    out = ConfigTuner(ds).tune(base, _metrics(), "turnover cao")
    assert out.decay == 10
    assert out.truncation == 0.05
    assert out.neutralization == "INDUSTRY"
    # giữ nguyên scope
    assert out.region == "USA" and out.universe == "TOP3000" and out.delay == 1


def test_config_tuner_decay_ngoai_khoang_giu_gia_tri_cu():
    base = SimConfig(decay=4, truncation=0.01, neutralization="MARKET")
    ds = FakeDeepSeek(['{"decay": 9999, "truncation": 0.05, "neutralization": "MARKET"}'])
    out = ConfigTuner(ds).tune(base, _metrics(), "x")
    assert out.decay == 4          # 9999 > 512 -> giữ cũ
    assert out.truncation == 0.05  # hợp lệ -> áp dụng


def test_config_tuner_neutralization_la_giu_cu():
    base = SimConfig(decay=4, truncation=0.01, neutralization="MARKET")
    ds = FakeDeepSeek(['{"decay": 4, "truncation": 0.01, "neutralization": "KHONG_CO"}'])
    out = ConfigTuner(ds).tune(base, _metrics(), "x")
    assert out.neutralization == "MARKET"


def test_config_tuner_truncation_ngoai_khoang_giu_cu():
    base = SimConfig(decay=4, truncation=0.01, neutralization="MARKET")
    ds = FakeDeepSeek(['{"decay": 8, "truncation": 0.9, "neutralization": "MARKET"}'])
    out = ConfigTuner(ds).tune(base, _metrics(), "x")
    assert out.truncation == 0.01  # 0.9 > 0.5 -> giữ cũ
    assert out.decay == 8


def test_config_tuner_khong_parse_duoc_giu_nguyen_config():
    base = SimConfig(decay=4, truncation=0.01, neutralization="MARKET")
    ds = FakeDeepSeek(["không phải json"])
    out = ConfigTuner(ds).tune(base, _metrics(), "x")
    assert out.key() == base.key()


def test_verdict_la_dataclass_co_reason_mac_dinh():
    v = Verdict(REFINE_FORMULA)
    assert v.reason == ""
