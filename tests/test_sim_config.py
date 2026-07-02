"""Test SimConfig: tách không gian cấu hình khỏi không gian biểu thức (GĐ5: T5.1, T5.2)."""

from __future__ import annotations

import pytest

from src.simulation.config import SimConfig


def test_default_co_dinh_cau_hinh_hop_ly():
    c = SimConfig.default()
    assert c.neutralization == "SUBINDUSTRY"
    assert c.truncation == 0.08
    assert c.decay == 0
    assert c.delay == 1


def test_to_settings_tra_dict_truyen_simulator():
    c = SimConfig(region="USA", universe="TOP3000", delay=1, neutralization="INDUSTRY", decay=6, truncation=0.05)
    s = c.to_settings()
    assert s["region"] == "USA"
    assert s["universe"] == "TOP3000"
    assert s["neutralization"] == "INDUSTRY"
    assert s["decay"] == 6
    assert s["truncation"] == 0.05
    assert s["delay"] == 1


def test_key_on_dinh_va_phan_biet_theo_cau_hinh():
    a = SimConfig.default()
    b = SimConfig.default()
    assert a.key() == b.key()  # cùng cấu hình -> cùng key
    c = SimConfig.default().with_overrides(decay=10)
    assert c.key() != a.key()  # khác decay -> key khác


def test_with_overrides_khong_doi_ban_goc():
    base = SimConfig.default()
    derived = base.with_overrides(truncation=0.02, neutralization="MARKET")
    assert derived.truncation == 0.02
    assert derived.neutralization == "MARKET"
    # bản gốc giữ nguyên (immutable-friendly)
    assert base.truncation == 0.08
    assert base.neutralization == "SUBINDUSTRY"


def test_key_doc_duoc_chua_cac_chieu_chinh():
    c = SimConfig(region="USA", universe="TOP3000", delay=1, neutralization="SECTOR", decay=4, truncation=0.1)
    key = c.key()
    assert "SECTOR" in key
    assert "decay=4" in key


@pytest.mark.parametrize("decay", [-1, 513])
def test_decay_must_be_valid_range(decay):
    with pytest.raises(ValueError, match="decay"):
        SimConfig(decay=decay)


@pytest.mark.parametrize("truncation", [0, -0.1, 0.51])
def test_truncation_must_be_valid_range(truncation):
    with pytest.raises(ValueError, match="truncation"):
        SimConfig(truncation=truncation)


def test_neutralization_is_normalized_to_uppercase():
    assert SimConfig(neutralization="market").neutralization == "MARKET"


def test_neutralization_must_be_valid():
    with pytest.raises(ValueError, match="neutralization"):
        SimConfig(neutralization="BAD_GROUP")


@pytest.mark.parametrize("neutralization", [None, 123])
def test_neutralization_must_be_string(neutralization):
    with pytest.raises(ValueError, match="neutralization"):
        SimConfig(neutralization=neutralization)


def test_test_period_max_trade_max_position_mac_dinh():
    c = SimConfig.default()
    assert c.test_period == "P0Y0M"
    assert c.max_trade == "OFF"
    assert c.max_position == "OFF"


def test_to_settings_co_test_period_max_trade_max_position():
    c = SimConfig.default(region="USA").with_overrides(max_position="ON")
    s = c.to_settings()
    assert s["testPeriod"] == "P0Y0M"
    assert s["maxTrade"] == "OFF"
    assert s["maxPosition"] == "ON"


def test_max_trade_normalize_hoa_thuong():
    assert SimConfig(max_trade="on").max_trade == "ON"


def test_max_trade_gia_tri_khong_hop_le_raise():
    with pytest.raises(ValueError, match="max_trade"):
        SimConfig(max_trade="MAYBE")


def test_max_position_gia_tri_khong_hop_le_raise():
    with pytest.raises(ValueError, match="max_position"):
        SimConfig(max_position="MAYBE")


def test_key_phan_biet_theo_max_trade():
    a = SimConfig.default()
    b = SimConfig.default().with_overrides(max_trade="ON")
    assert a.key() != b.key()


@pytest.mark.parametrize("region", ["ASI", "JPN", "HKG", "KOR", "TWN", "asi"])
def test_default_tu_bat_max_trade_cho_region_bat_buoc(region):
    c = SimConfig.default(region=region)
    assert c.max_trade == "ON"


@pytest.mark.parametrize("region", ["USA", "EUR", "GLB", "CHN", "AMR"])
def test_default_khong_bat_max_trade_cho_region_khac(region):
    c = SimConfig.default(region=region)
    assert c.max_trade == "OFF"
