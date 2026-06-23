"""Port (Protocol) nguồn dữ liệu thị trường — MiniBrain phụ thuộc cái NÀY, không phụ thuộc
feed cụ thể. Adapter chịu trách nhiệm PIT correctness, lịch sử universe, quy ước delay."""

from __future__ import annotations

from typing import Protocol

from src.data.market_panel import MarketData


class MarketDataSource(Protocol):
    def load(self, start: str, end: str, universe: str = "TOP3000") -> MarketData: ...

    def available_fields(self) -> list[str]: ...
