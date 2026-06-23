from config.settings import Settings


def test_market_settings_defaults():
    s = Settings()
    assert s.market_data_dir == "data/market"
    assert s.global_seed == 42
