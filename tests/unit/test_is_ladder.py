"""IS-Ladder Sharpe robustness gate (docs consultant-submission-tests).

Logic ladder: xét Sharpe cửa sổ trượt N=2..10 năm GẦN NHẤT; N=2 dùng ngưỡng chặt nhất.
- Sharpe[N] < FAIL(1.58) -> FAIL ngay (alpha suy thoái gần đây).
- Sharpe[N] >= PASS[N] -> PASS ngay (PASS: 2-5=2.38, 6=2.22, 7=2.06, 8=1.90, 9=1.74, 10=1.59).
- ở giữa -> tăng N. Turnover < 30% -> nhân PASS ×0.85 (FAIL không nhân).
Tách `ladder_decision` (quyết định thuần, test chính xác) khỏi `is_ladder_verdict`
(end-to-end trên daily_pnl + years()).
"""

from __future__ import annotations

import numpy as np

from config.thresholds import IS_LADDER_PASS
from src.backtest.is_ladder import (
    LadderVerdict,
    is_ladder_verdict,
    ladder_decision,
    pass_threshold,
)


class _FakeData:
    """Chỉ cần .years() -> {year: slice} liên tục, tăng dần — như MarketData.years()."""

    def __init__(self, days_per_year: dict[int, int]):
        self._years: dict[int, slice] = {}
        start = 0
        for y, n in days_per_year.items():
            self._years[y] = slice(start, start + n)
            start += n
        self._total = start

    def years(self) -> dict[int, slice]:
        return self._years

    @property
    def total(self) -> int:
        return self._total


# --------------------------------------------------------------- pass_threshold
def test_pass_threshold_theo_thang_tai_lieu():
    assert pass_threshold(2, low_turnover=False) == IS_LADDER_PASS[2] == 2.38
    assert pass_threshold(5, low_turnover=False) == 2.38
    assert pass_threshold(6, low_turnover=False) == 2.22
    assert pass_threshold(10, low_turnover=False) == 1.59


def test_pass_threshold_clamp_qua_10_nam():
    # Cửa sổ > 10 năm dùng ngưỡng năm 10 (thấp nhất), không KeyError.
    assert pass_threshold(15, low_turnover=False) == IS_LADDER_PASS[10]


def test_low_turnover_nhan_0_85_chi_len_pass():
    assert pass_threshold(2, low_turnover=True) == 2.38 * 0.85


# --------------------------------------------------------------- ladder_decision
def test_fail_ngay_khi_2y_sharpe_duoi_nguong_fail():
    v = ladder_decision({2: 1.20}, low_turnover=False)
    assert not v.passed
    assert v.fail_window == 2
    assert "FAIL" in v.detail


def test_pass_ngay_khi_2y_vuot_nguong_pass():
    v = ladder_decision({2: 2.50}, low_turnover=False)
    assert v.passed
    assert v.fail_window is None


def test_leo_thang_toi_nam_6_moi_pass():
    # N=2..5 Sharpe 2.0: không FAIL (>=1.58) nhưng chưa PASS (<2.38) -> leo tiếp;
    # N=6: 2.30 >= 2.22 -> PASS ở cửa sổ 6 năm.
    windows = {2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0, 6: 2.30}
    v = ladder_decision(windows, low_turnover=False)
    assert v.passed
    assert v.pass_window == 6


def test_low_turnover_cuu_alpha_o_bien():
    # 2.10: bình thường < 2.38 (không pass, chỉ 1 cửa sổ -> borderline); TO thấp -> 2.023 -> PASS.
    assert ladder_decision({2: 2.10}, low_turnover=True).passed
    v = ladder_decision({2: 2.10}, low_turnover=False)
    assert v.passed  # borderline vẫn passed=True (không suy thoái)
    assert v.pass_window is None  # nhưng KHÔNG đạt PASS tường minh


def test_fail_o_cua_so_giua_bat_suy_thoai_som():
    # 2Y ổn (2.5 -> lẽ ra PASS ngay). Đảo lại: 2Y yếu phải FAIL trước khi xét xa hơn.
    windows = {2: 1.0, 3: 3.0, 4: 3.0}
    v = ladder_decision(windows, low_turnover=False)
    assert not v.passed and v.fail_window == 2


# --------------------------------------------------------------- is_ladder_verdict
def _pnl_for_sharpe(target_sharpe: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = (x - x.mean()) / x.std()  # mean 0, std 1
    # mean/std * sqrt(252) = target -> mean = target/sqrt(252) (std giữ ~1)
    return x + target_sharpe / np.sqrt(252)


def test_end_to_end_fail_khi_2_nam_gan_nhat_yeu():
    # 4 năm: 2 năm đầu mạnh, 2 năm gần nhất yếu -> ladder xét recent-2Y -> FAIL.
    data = _FakeData({2020: 252, 2021: 252, 2022: 252, 2023: 252})
    strong = _pnl_for_sharpe(3.0, 504, seed=1)
    weak = _pnl_for_sharpe(0.5, 504, seed=2)
    pnl = np.concatenate([strong, weak])
    v = is_ladder_verdict(pnl, data, turnover=0.4)
    assert isinstance(v, LadderVerdict)
    assert not v.passed
    assert v.fail_window == 2


def test_end_to_end_pass_khi_gan_nhat_manh():
    data = _FakeData({2020: 252, 2021: 252, 2022: 252, 2023: 252})
    pnl = _pnl_for_sharpe(3.0, 1008, seed=3)
    v = is_ladder_verdict(pnl, data, turnover=0.4)
    assert v.passed
    assert v.fail_window is None


def test_it_hon_2_nam_thi_bo_qua_ladder():
    data = _FakeData({2023: 200})
    pnl = _pnl_for_sharpe(0.1, 200, seed=4)
    v = is_ladder_verdict(pnl, data, turnover=0.4)
    assert v.passed  # không đủ dữ liệu -> không chặn


def test_turnover_thap_lam_de_pass_hon():
    # Sharpe ~2.1 mọi cửa sổ: TO cao -> borderline (pass_window None); TO thấp -> PASS tường minh.
    data = _FakeData({2020: 252, 2021: 252})
    pnl = _pnl_for_sharpe(2.1, 504, seed=5)
    hi = is_ladder_verdict(pnl, data, turnover=0.5)
    lo = is_ladder_verdict(pnl, data, turnover=0.2)
    assert hi.pass_window is None
    assert lo.pass_window == 2
