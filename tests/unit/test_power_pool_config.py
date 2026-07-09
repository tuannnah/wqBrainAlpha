from datetime import date

from src.app.power_pool_config import resolve_theme_sim_config
from src.simulation.config import SimConfig


def test_resolve_co_theme_override_region_universe_delay():
    base = SimConfig.default(region="USA", universe="TOP3000", delay=1)
    res = resolve_theme_sim_config(base, date(2026, 7, 9))
    assert res.theme is not None
    assert res.sim_config.universe == "TOP1000"
    assert res.sim_config.region == "USA"
    assert res.sim_config.delay == 1
    assert res.region == "USA"
    assert res.universe == "TOP1000"
    assert "STATISTICAL" in res.allowed_neutralizations
    assert res.warning is None


def test_resolve_khong_theme_giu_nguyen_va_canh_bao():
    base = SimConfig.default(region="USA", universe="TOP3000", delay=1)
    res = resolve_theme_sim_config(base, date(2026, 8, 15))  # ngoài lịch
    assert res.theme is None
    assert res.sim_config.universe == "TOP3000"  # giữ nguyên
    assert res.allowed_neutralizations == frozenset()
    assert res.warning is not None
