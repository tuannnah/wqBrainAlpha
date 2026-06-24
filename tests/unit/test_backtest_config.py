"""Test PortfolioConfig: stage-separation config (neut/decay/trunc/scale/delay)."""

from __future__ import annotations

import pytest

from src.backtest.config import Neutralization, PortfolioConfig


def test_neutralization_has_five_members():
    names = {m.name for m in Neutralization}
    assert names == {"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"}


def test_default_config_matches_master_spec():
    cfg = PortfolioConfig()
    assert cfg.neutralization is Neutralization.SECTOR
    assert cfg.decay == 0
    assert cfg.truncation == pytest.approx(0.10)
    assert cfg.scale_book == pytest.approx(1.0)
    assert cfg.delay == 1


def test_config_is_frozen():
    cfg = PortfolioConfig()
    with pytest.raises(AttributeError):
        cfg.decay = 5  # type: ignore[misc]


def test_config_is_hashable_for_cache_key():
    cfg1 = PortfolioConfig(decay=10)
    cfg2 = PortfolioConfig(decay=10)
    assert hash(cfg1) == hash(cfg2)
    assert cfg1 == cfg2


def test_config_custom_values():
    cfg = PortfolioConfig(
        neutralization=Neutralization.MARKET, decay=5, truncation=0.05,
        scale_book=2.0, delay=2,
    )
    assert cfg.neutralization is Neutralization.MARKET
    assert cfg.decay == 5
    assert cfg.truncation == pytest.approx(0.05)
    assert cfg.scale_book == pytest.approx(2.0)
    assert cfg.delay == 2
