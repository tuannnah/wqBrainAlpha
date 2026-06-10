"""Alias tương thích ngược cho tên lớp cũ `BrainBatchAlpha`.

Toàn bộ logic đăng nhập và API đã chuyển sang `WorldQuantClient`. Lớp này chỉ
giữ lại tên cũ cho các import bên ngoài; pipeline sinh Alpha hard-code đã bị
loại bỏ và thay bằng `research_engine`.
"""

from worldquant_client import AuthenticationError, WorldQuantClient

__all__ = ["AuthenticationError", "BrainBatchAlpha"]


class BrainBatchAlpha(WorldQuantClient):
    """Tên lớp tương thích ngược cho các import cũ."""
