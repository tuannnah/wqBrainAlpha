"""Test CLI score-one/generate: dùng CliRunner, KHÔNG mạng/sim Brain. Ghi MarketData fake ra
parquet tạm để CLI đọc lại qua --market-data-dir."""

from __future__ import annotations

from typer.testing import CliRunner

from main import app

runner = CliRunner()


def _write_panel(data_dir, panel) -> None:  # noqa: ANN001
    """Ghi small_panel ra layout ParquetSource.load() đọc được.

    ParquetSource có module-level `save(md, root)` — gọi trực tiếp.
    Layout: root/axes_assets.parquet, root/fields/<name>.parquet,
    root/universe.parquet, root/returns.parquet, root/groups/<g>.parquet.
    """
    from src.data.adapters.parquet_source import save

    save(panel, str(data_dir))


def test_score_one_missing_market_data_dir_fails_clearly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["score-one", "close", "--market-data-dir", str(tmp_path / "nope")],
    )
    assert result.exit_code == 1


def test_score_one_real_panel_prints_metrics(tmp_path, small_panel) -> None:  # noqa: ANN001
    data_dir = tmp_path / "panel"
    _write_panel(data_dir, small_panel)
    result = runner.invoke(app, ["score-one", "close", "--market-data-dir", str(data_dir), "--no-pool"])
    assert result.exit_code == 0
    assert "sharpe" in result.stdout.lower()


def test_score_one_invalid_expr_exits_zero_prints_fail(tmp_path, small_panel) -> None:  # noqa: ANN001
    data_dir = tmp_path / "panel2"
    _write_panel(data_dir, small_panel)
    result = runner.invoke(
        app, ["score-one", "not_a_real_op(close,", "--market-data-dir", str(data_dir), "--no-pool"],
    )
    assert result.exit_code == 0  # CLI không crash; in verdict fail
    assert "false" in result.stdout.lower() or "fail" in result.stdout.lower()


def test_generate_missing_market_data_dir_fails_clearly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["generate", "--market-data-dir", str(tmp_path / "nope"), "--count", "4"],
    )
    assert result.exit_code == 1
