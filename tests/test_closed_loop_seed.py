"""base_seed cho vòng kín: hardcode 42 khiến GP luôn sinh CÙNG quần thể mỗi lần chạy
tiến trình mới -> khi pool tích lũy, quần thể đó bị decorrelate loại hết -> no_more_ideas
(0 ý tưởng). `_resolve_base_seed` cho phép ngẫu nhiên hóa (None/0) để mỗi lần chạy khác
nhau, nhưng vẫn tái lập được nếu chỉ định số cụ thể."""

from __future__ import annotations

import main


def test_giu_nguyen_seed_cu_the_de_tai_lap() -> None:
    assert main._resolve_base_seed(42) == 42
    assert main._resolve_base_seed(1234) == 1234


def test_0_hoac_none_cho_seed_ngau_nhien_hop_le() -> None:
    for val in (0, None):
        s = main._resolve_base_seed(val)
        assert isinstance(s, int)
        assert 1 <= s < 2**31  # nằm trong khoảng seed hợp lệ cho numpy/GP


def test_ngau_nhien_khac_nhau_qua_nhieu_lan() -> None:
    # Xác suất 20 lần ra cùng một giá trị ~0 -> phải có >1 giá trị phân biệt.
    vals = {main._resolve_base_seed(0) for _ in range(20)}
    assert len(vals) > 1


def test_closed_loop_configs_khop_local_va_brain() -> None:
    """cfg (local gate) và sim_config (Brain sim) phải dùng CHUNG một bộ
    neutralization/decay/truncation — tránh mismatch làm local gate lọc sai."""
    from src.backtest.config import Neutralization

    cfg, sim = main._closed_loop_configs("SUBINDUSTRY", 4, 0.08, 1, "USA", "TOP3000")
    # local gate (PortfolioConfig, neutralization là enum)
    assert cfg.neutralization == Neutralization.SUBINDUSTRY
    assert cfg.decay == 4
    assert cfg.truncation == 0.08
    # Brain sim (SimConfig, neutralization là str đã normalize)
    assert sim.neutralization == "SUBINDUSTRY"
    assert sim.decay == 4
    assert sim.truncation == 0.08
    assert sim.region == "USA" and sim.universe == "TOP3000" and sim.delay == 1


def test_closed_loop_defaults_la_bo_config_thong_nhat() -> None:
    """Default closed-loop phải là bộ thống nhất SUBINDUSTRY/4/0.08 (đổi mặc định engine)."""
    import inspect

    sig = inspect.signature(main._run_closed_loop_session)
    assert sig.parameters["neutralization"].default == "SUBINDUSTRY"
    assert sig.parameters["decay"].default == 4
    assert sig.parameters["truncation"].default == 0.08
