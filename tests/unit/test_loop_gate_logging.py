"""Log trạng thái local gate phải ĐÚNG THỜI ĐIỂM: không cảnh báo 'gate tắt' ngay lúc
__init__ (vì closed-loop set market_data SAU khi dựng loop -> cảnh báo lúc init là báo
động giả, gây chẩn nhầm 'gate đang tắt'). Thay vào đó log MỘT LẦN lúc bắt đầu run:
cảnh báo nếu thật sự thiếu market_data, báo INFO 'gate BẬT' nếu đã có.
"""

from __future__ import annotations

from loguru import logger

from src.backtest.config import PortfolioConfig
from src.llm.loop import RefinementLoop
from src.simulation.config import SimConfig


class _Repo:
    def recent_failures(self, n):
        return []


def _make_loop(market_data):
    return RefinementLoop(
        hypothesis_gen=None, translator=None, refiner=None,
        simulator=None, prefilter=None, repo=_Repo(),
        region="USA", universe="TOP3000", delay=1,
        sim_config=SimConfig.default(region="USA", universe="TOP3000", delay=1),
        market_data=market_data, local_gate_cfg=PortfolioConfig(),
    )


def _capture(fn, level="DEBUG"):
    msgs: list[str] = []
    hid = logger.add(lambda m: msgs.append(m.record["message"]), level=level)
    try:
        fn()
    finally:
        logger.remove(hid)
    return msgs


def test_khong_canh_bao_gate_luc_khoi_tao():
    msgs = _capture(lambda: _make_loop(market_data=None), level="WARNING")
    assert not any("gate" in m.lower() for m in msgs)


def test_canh_bao_gate_tat_khi_run_neu_thieu_market_data():
    loop = _make_loop(market_data=None)
    msgs = _capture(loop._ensure_gate_status_logged, level="WARNING")
    assert any("gate" in m.lower() and "tắt" in m.lower() for m in msgs)


def test_bao_gate_bat_khi_co_market_data():
    loop = _make_loop(market_data=object())
    msgs = _capture(loop._ensure_gate_status_logged, level="DEBUG")
    assert any("gate" in m.lower() and "bật" in m.lower() for m in msgs)
    assert not any("tắt" in m.lower() for m in msgs)


def test_chi_log_mot_lan():
    loop = _make_loop(market_data=None)

    def run_twice():
        loop._ensure_gate_status_logged()
        loop._ensure_gate_status_logged()

    msgs = _capture(run_twice, level="WARNING")
    assert len([m for m in msgs if "gate" in m.lower()]) == 1
