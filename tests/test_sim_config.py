"""Test SimConfig: tách không gian cấu hình khỏi không gian biểu thức (GĐ5: T5.1, T5.2)."""

from __future__ import annotations

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
