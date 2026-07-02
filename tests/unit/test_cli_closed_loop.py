"""Smoke test CLI closed-loop: KHÔNG login/sim Brain thật (chỉ kiểm parse + lỗi input rõ)."""

from __future__ import annotations

from typer.testing import CliRunner

from main import app

runner = CliRunner()


def test_closed_loop_missing_market_data_dir_fails_clearly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["closed-loop", "--market-data-dir", str(tmp_path / "nope")],
    )
    assert result.exit_code == 1


def test_closed_loop_help_lists_options() -> None:
    result = runner.invoke(app, ["closed-loop", "--help"])
    assert result.exit_code == 0
    assert "market-data-dir" in result.stdout
    assert "patience" in result.stdout
