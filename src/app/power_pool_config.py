"""Biến Power Pool Theme của một ngày thành SimConfig (override region/delay/universe) + tập
neutralization cho phép. Đọc theme làm MẶC ĐỊNH: có theme -> áp ràng buộc; không có -> giữ
config gốc + cảnh báo (đường Regular). Thuần logic, dễ test — wiring nằm ở main.py."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.scoring.power_pool_theme import PowerPoolThemeWeek, theme_for_date
from src.simulation.config import SimConfig


@dataclass(frozen=True)
class ThemeResolution:
    sim_config: SimConfig
    allowed_neutralizations: frozenset[str]
    theme: PowerPoolThemeWeek | None
    region: str
    universe: str
    warning: str | None


def resolve_theme_sim_config(
    base: SimConfig, on_date: date, calendar: list[PowerPoolThemeWeek] | None = None
) -> ThemeResolution:
    """Có theme cho `on_date` -> override region/delay/universe của `base` theo theme + trả tập
    allowed_neutralizations. Không có theme -> trả `base` nguyên vẹn, allowed rỗng, kèm warning."""
    week = theme_for_date(on_date, calendar)
    if week is None:
        return ThemeResolution(
            sim_config=base, allowed_neutralizations=frozenset(), theme=None,
            region=base.region, universe=base.universe,
            warning=(
                f"Không có Power Pool Theme cho {on_date} trong lịch — giữ config Regular "
                f"({base.region}/{base.universe}/delay={base.delay}). Cập nhật lịch nếu muốn "
                f"nộp Pure Power Pool (xem docstring src/scoring/power_pool_theme.py)."
            ),
        )
    overrides: dict = {}
    if week.region is not None:
        overrides["region"] = week.region
    if week.universe is not None:
        overrides["universe"] = week.universe
    if week.delay is not None:
        overrides["delay"] = week.delay
    sim_config = base.with_overrides(**overrides) if overrides else base
    return ThemeResolution(
        sim_config=sim_config,
        allowed_neutralizations=week.allowed_neutralizations,
        theme=week,
        region=sim_config.region,
        universe=sim_config.universe,
        warning=None,
    )
