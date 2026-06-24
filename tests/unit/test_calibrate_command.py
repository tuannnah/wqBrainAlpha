"""Test lệnh CLI `calibrate`: re-score local trên panel parquet thật, in báo cáo Spearman ρ.

Dùng DB tạm + panel parquet tạm (KHÔNG đụng DB/data thật của user). Chứng minh đường ống
CLI: load_brain_records -> make_local_scorer(MarketData) -> CalibrationHarness -> in ρ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from main import app
from src.data.adapters.parquet_source import save
from src.data.market_panel import MarketData
from src.storage.models import AlphaModel, Base, SimulationModel

runner = CliRunner()


def _make_panel() -> MarketData:
    rng = np.random.default_rng(7)
    t, n = 60, 12
    dates = (np.datetime64("2021-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array([f"A{i:02d}" for i in range(n)], dtype=np.str_)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(t, n)), axis=0))
    volume = rng.uniform(1e5, 1e6, size=(t, n))
    universe = np.ones((t, n), dtype=bool)
    prev = np.empty_like(close)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = (close - prev) / prev
    sector = np.tile(np.arange(n) % 3, (t, 1)).astype(np.int64)
    return MarketData(dates=dates, assets=assets, fields={"close": close, "volume": volume},
                      universe=universe, returns=returns, groups={"sector": sector})


def _seed_db(db_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    exprs = ["rank(close)", "ts_mean(close, 5)", "rank(volume)",
             "rank(divide(close, volume))", "ts_std_dev(close, 10)"]
    for i, expr in enumerate(exprs, start=1):
        session.add(AlphaModel(id=f"a{i}", expression=expr, source="manual"))
        session.add(SimulationModel(
            id=f"s{i}", alpha_id=f"a{i}", region="USA", universe="TOP3000",
            sharpe=float(i), fitness=float(i) * 0.8, turnover=0.3, status="passed",
        ))
    session.commit()
    session.close()


def test_calibrate_runs_against_temp_panel_and_prints_rho(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "calib_test.db"
    _seed_db(db_path)
    panel_dir = tmp_path / "panel"
    save(_make_panel(), str(panel_dir))

    monkeypatch.setenv("WQ_NO_FILE_LOG", "1")
    result = runner.invoke(app, [
        "calibrate", "--db-url", f"sqlite:///{db_path}", "--market-data-dir", str(panel_dir),
    ])
    assert result.exit_code == 0, result.stdout
    assert "spearman" in result.stdout.lower() or "ρ" in result.stdout


def test_calibrate_errors_clearly_when_no_market_data_dir(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "calib_test.db"
    _seed_db(db_path)
    monkeypatch.setenv("WQ_NO_FILE_LOG", "1")
    result = runner.invoke(app, ["calibrate", "--db-url", f"sqlite:///{db_path}"])
    # thiếu --market-data-dir -> lỗi rõ ràng, KHÔNG in báo cáo giả
    assert result.exit_code == 1
    assert "market-data-dir" in result.stdout


def test_calibrate_exits_zero_when_no_records(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "empty.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    monkeypatch.setenv("WQ_NO_FILE_LOG", "1")
    result = runner.invoke(app, ["calibrate", "--db-url", f"sqlite:///{db_path}"])
    assert result.exit_code == 0
    assert "không có brainrecord" in result.stdout.lower()


def test_calibrate_on_fresh_db_creates_tables_and_exits_zero(tmp_path, monkeypatch) -> None:
    # DB chưa từng tồn tại (không có bảng) -> init_db tạo bảng -> 0 record -> exit 0 nhẹ nhàng,
    # KHÔNG ném OperationalError 'no such table'.
    db_path = tmp_path / "never_created.db"
    monkeypatch.setenv("WQ_NO_FILE_LOG", "1")
    result = runner.invoke(app, ["calibrate", "--db-url", f"sqlite:///{db_path}"])
    assert result.exit_code == 0, result.stdout
    assert "không có brainrecord" in result.stdout.lower()
