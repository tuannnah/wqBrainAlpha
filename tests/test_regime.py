"""Test đo độ ổn định theo regime (Sharpe theo năm) — review 3."""

from __future__ import annotations

from src.scoring.regime import min_annual_sharpe, regime_fit, yearly_sharpe


def test_yearly_sharpe_tach_theo_nam():
    # 2020 dương ổn định; 2021 dao động mạnh quanh 0 -> sharpe năm thấp hơn.
    series = [
        ("2020-01-01", 1.0), ("2020-06-01", 1.2), ("2020-12-01", 0.8),
        ("2021-01-01", 1.0), ("2021-06-01", -1.0), ("2021-12-01", 0.5),
    ]
    ys = yearly_sharpe(series)
    assert set(ys) == {2020, 2021}
    assert ys[2020] > ys[2021]


def test_yearly_sharpe_nhan_year_dang_int():
    series = [(2019, 1.0), (2019, 1.1), (2019, 0.9)]
    ys = yearly_sharpe(series)
    assert set(ys) == {2019}


def test_min_annual_sharpe():
    assert min_annual_sharpe({2020: 1.5, 2021: 0.3}) == 0.3
    assert min_annual_sharpe({}) == 0.0


def test_regime_fit_phat_nam_yeu():
    strong = regime_fit({2020: 1.5, 2021: 1.2}, target=1.0)   # min 1.2 -> clamp 1.0
    fragile = regime_fit({2020: 1.5, 2021: 0.2}, target=1.0)  # min 0.2
    assert strong == 1.0
    assert fragile < strong
    # năm lỗ -> regime_fit = 0
    assert regime_fit({2020: 1.5, 2021: -0.5}, target=1.0) == 0.0
    # không đo được -> không phạt (1.0)
    assert regime_fit({}, target=1.0) == 1.0
