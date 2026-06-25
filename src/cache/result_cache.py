# src/cache/result_cache.py
"""ResultCache — lớp DB-backed (B12 tier 3): canonical_hash+config+window -> AlphaMetrics.

Bọc MiniBrainRepository để GP/pipeline (Phase 7/8) có một điểm truy cập cache duy nhất,
không gọi trực tiếp repository — cho phép đổi backend cache (vd thêm LRU phía trước) mà
không sửa call site. Re-scoring một expression đã biết là MIỄN PHÍ khi cache hit.
"""

from __future__ import annotations

from src.backtest.metrics_local import AlphaMetrics
from src.storage.repository import MiniBrainRepository


class ResultCache:
    """Cache kết quả backtest theo khóa (canonical_hash, config_json, data_window)."""

    def __init__(self, repo: MiniBrainRepository) -> None:
        self.repo = repo

    def get(
        self, canonical_hash: str, config_json: str, data_window: str,
    ) -> AlphaMetrics | None:
        return self.repo.result_cache_get(canonical_hash, config_json, data_window)

    def put(
        self, canonical_hash: str, expr_string: str, depth: int, complexity: int,
        fields: set[str], config_json: str, data_window: str, metrics: AlphaMetrics,
        seed: int | None = None,
    ) -> None:
        self.repo.result_cache_put(
            canonical_hash, expr_string, depth, complexity, fields, config_json,
            data_window, metrics, seed,
        )
