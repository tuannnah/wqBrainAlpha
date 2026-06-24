"""Cache LRU sub-expression theo canonical hash — chia sẻ panel đã eval giữa các node
AST trùng nhau trong cùng một cây hoặc giữa các cá thể GP (B6: throughput win chính
trước khi tối ưu numba)."""

from __future__ import annotations

from collections import OrderedDict

from src.local_types import Panel


class SubexprCache:
    """LRU cache key=canonical hash (str) -> Panel (T,N) đã eval."""

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize phải dương")
        self._maxsize = maxsize
        self._store: OrderedDict[str, Panel] = OrderedDict()

    def get(self, key: str) -> Panel | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: Panel) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)
