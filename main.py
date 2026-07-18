"""CLI entry cho WorldQuant Brain Auto-Alpha Tool."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

# Đảm bảo in được tiếng Việt trên console Windows (cp1252) khi output bị pipe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from src.app.cli import auth as cli_auth
from src.app.cli import fields as cli_fields
from src.app.cli import simulate as cli_simulate
from src.app.cli import generate as cli_generate
from src.app.cli import submit as cli_submit
from src.app.cli import report as cli_report
from src.app.cli import migrate as cli_migrate
from src.app.cli import llm as cli_llm
from src.app.cli import research as cli_research
from src.app.cli import closed_loop as cli_closed_loop
from src.app.cli import marathon as cli_marathon
from src.app import menu as cli_menu

app = typer.Typer(help="WorldQuant Brain Auto-Alpha Tool")
app.command()(cli_auth.login)
app.command("probe-fields")(cli_fields.probe_fields)
app.command("warm-cache")(cli_fields.warm_cache_cmd)
app.command("fetch-fields")(cli_fields.fetch_fields)
app.command("cache-status")(cli_fields.cache_status)
app.command("fetch-operators")(cli_fields.fetch_operators)
app.command("list-fields")(cli_fields.list_fields)
app.command()(cli_simulate.simulate)
app.command("sweep-config")(cli_simulate.sweep_config)
app.command()(cli_generate.generate)
app.command("score-one")(cli_generate.score_one_cmd)
app.command()(cli_submit.submit)
app.command()(cli_report.top)
app.command()(cli_report.originality)
app.command("genius-report")(cli_report.genius_report_cmd)
app.command("migrate-sqlite")(cli_migrate.migrate_sqlite)
app.command("calibrate")(cli_migrate.calibrate)
app.command("check-deepseek")(cli_llm.check_deepseek)
app.command("llm-generate")(cli_llm.llm_generate)
app.command("llm-ideas")(cli_llm.llm_ideas)
app.command()(cli_research.research)
app.command("closed-loop")(cli_closed_loop.closed_loop_cmd)
app.command()(cli_marathon.marathon)
app.command()(cli_menu.start)
console = Console()

LOG_DIR = Path("logs")


def _setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # WQ_NO_FILE_LOG: bỏ file sink (conftest đặt khi chạy test) để không ghi
    # đè log production bằng nhiễu fixture.
    if os.environ.get("WQ_NO_FILE_LOG"):
        return
    LOG_DIR.mkdir(exist_ok=True)
    logger.add(LOG_DIR / "wq_alpha_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="14 days")


if __name__ == "__main__":
    app()
