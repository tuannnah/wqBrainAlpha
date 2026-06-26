"""GPSeedGenerator — adapter mỏng cho ``RefinementLoop.idea_generator``: sinh ``n`` core
seed (Phase 7.3) serialize thành chuỗi FASTEXPR. RefinementLoop dùng các chuỗi này làm hạt
giống direction mới khi reseed kích hoạt.

Adapter sống ở ``src/gp/`` để giữ chiều phụ thuộc một chiều (dependency rule B1): ``src.llm``
consume Protocol ``idea_generator`` (đã có sẵn), còn ``src.gp.seed_adapter`` chỉ implement
Protocol đó — ``src.gp`` KHÔNG import ``src.llm``.
"""

from __future__ import annotations

import numpy as np

from src.gp.seeds import all_seed_cores
from src.lang.visitors import Serializer


class GPSeedGenerator:
    """Trả ``n`` core seed serialize từ pool families + novel ideas (LLM tùy chọn)."""

    def __init__(
        self,
        *,
        with_llm: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.with_llm = with_llm
        self.rng = rng if rng is not None else np.random.default_rng()
        # Cache pool một lần — seed cores tĩnh trong phạm vi adapter.
        self._pool_strings: list[str] | None = None

    def _ensure_pool(self) -> list[str]:
        if self._pool_strings is None:
            cores = all_seed_cores(with_llm=self.with_llm)
            self._pool_strings = [c.accept(Serializer()) for c in cores]
        return self._pool_strings

    def generate_ideas(self, n: int) -> list[str]:
        """Trả ``n`` chuỗi seed; pool < n → lấy with-replacement để đủ, ngược lại không lặp."""
        pool = self._ensure_pool()
        if not pool or n <= 0:
            return []
        replace = n > len(pool)
        indices = self.rng.choice(len(pool), size=n, replace=replace)
        return [pool[int(i)] for i in indices]
