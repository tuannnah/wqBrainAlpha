"""Test SubexprCache: LRU theo key string (canonical hash), giữ panel (T,N)."""

from __future__ import annotations

import numpy as np

from src.engine.subexpr_cache import SubexprCache


def test_miss_tra_none() -> None:
    cache = SubexprCache(maxsize=4)
    assert cache.get("hash-a") is None


def test_put_get_hit_tra_dung_panel() -> None:
    cache = SubexprCache(maxsize=4)
    panel = np.array([[1.0, 2.0], [3.0, 4.0]])
    cache.put("hash-a", panel)
    out = cache.get("hash-a")
    assert out is not None
    np.testing.assert_array_equal(out, panel)


def test_vuot_maxsize_evict_key_cu_nhat() -> None:
    cache = SubexprCache(maxsize=2)
    cache.put("a", np.zeros((1, 1)))
    cache.put("b", np.zeros((1, 1)))
    cache.put("c", np.zeros((1, 1)))  # evict "a" (cũ nhất, chưa được get lại)
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert len(cache) == 2


def test_get_lam_moi_thu_tu_lru() -> None:
    cache = SubexprCache(maxsize=2)
    cache.put("a", np.zeros((1, 1)))
    cache.put("b", np.zeros((1, 1)))
    cache.get("a")  # "a" vừa được dùng -> không còn là cũ nhất
    cache.put("c", np.zeros((1, 1)))  # evict "b" thay vì "a"
    assert cache.get("b") is None
    assert cache.get("a") is not None
