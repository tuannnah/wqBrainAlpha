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


def test_local_neutralization_ha_cap_theo_nhom_panel() -> None:
    """Local gate hạ cấp neutralization về nhóm panel CÓ (tránh KeyError). Panel local
    chỉ có 'sector' -> SUBINDUSTRY/INDUSTRY hạ về SECTOR; nếu không có sector -> NONE."""
    assert main._local_neutralization("SUBINDUSTRY", {"sector"}) == "SECTOR"
    assert main._local_neutralization("INDUSTRY", {"sector"}) == "SECTOR"
    assert main._local_neutralization("SUBINDUSTRY", {"subindustry", "sector"}) == "SUBINDUSTRY"
    assert main._local_neutralization("SECTOR", {"sector"}) == "SECTOR"
    assert main._local_neutralization("SUBINDUSTRY", set()) == "NONE"
    assert main._local_neutralization("NONE", {"sector"}) == "NONE"
    assert main._local_neutralization("MARKET", set()) == "MARKET"


def test_closed_loop_configs_local_ha_cap_brain_day_du() -> None:
    """Brain sim giữ SUBINDUSTRY đầy đủ; local gate hạ về SECTOR khi panel chỉ có
    'sector' — nhưng decay/truncation khớp nhau ở cả hai."""
    from src.backtest.config import Neutralization

    cfg, sim = main._closed_loop_configs(
        "SUBINDUSTRY", 4, 0.08, 1, "USA", "TOP3000", {"sector"},
    )
    # local gate hạ cấp (panel chỉ có sector)
    assert cfg.neutralization == Neutralization.SECTOR
    assert cfg.decay == 4 and cfg.truncation == 0.08
    # Brain sim giữ neutralization đầy đủ
    assert sim.neutralization == "SUBINDUSTRY"
    assert sim.decay == 4 and sim.truncation == 0.08
    assert sim.region == "USA" and sim.universe == "TOP3000" and sim.delay == 1


def test_closed_loop_defaults_la_bo_config_thong_nhat() -> None:
    """Default closed-loop phải là bộ thống nhất MARKET/4/0.08 (Task 5: đổi mặc định
    neutralization SUBINDUSTRY -> MARKET — docs Brain khuyến nghị MARKET/SECTOR cho alpha
    price/volume, và sweep neutralization (Task 1) sẽ tự chọn lại MARKET/SECTOR cho từng
    ứng viên nên default chỉ là điểm khởi đầu)."""
    import inspect

    sig = inspect.signature(main._run_closed_loop_session)
    assert sig.parameters["neutralization"].default == "MARKET"
    assert sig.parameters["decay"].default == 4
    assert sig.parameters["truncation"].default == 0.08
