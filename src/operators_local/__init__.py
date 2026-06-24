"""Side-effect import: mỗi submodule tự @register() operator vào REGISTRY toàn cục khi
import. Import package này (hoặc bất kỳ submodule) là đủ để có toàn bộ operator Phase 2."""

from __future__ import annotations

from src.operators_local import (  # noqa: F401
    arithmetic,
    conditional,
    cross_sectional,
    group,
    neutralization,
    timeseries,
)
