"""Zoo tham chiếu để đo độ độc đáo của alpha (T3.3, T3.4).

Gồm (a) tập Alpha101 đã dịch FASTEXPR, (b) các alpha đã nộp (bổ sung dần). Parse
sẵn thành AST. Điểm độc đáo = 1 - tương đồng cao nhất so với toàn bộ zoo.
"""

from __future__ import annotations

from loguru import logger

from src.decorrelation.alpha101 import ALPHA101_FASTEXPR
from src.decorrelation.similarity import similarity_ratio
from src.generation.ast_utils import parse_expression


class ReferenceZoo:
    def __init__(self, expressions=None):
        self._entries: list[tuple[str, object]] = []  # (expr, node)
        for expr in expressions or []:
            self.add(expr)

    @classmethod
    def default(cls, extra=None) -> "ReferenceZoo":
        return cls(list(ALPHA101_FASTEXPR) + list(extra or []))

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, expression: str) -> bool:
        """Thêm alpha vào zoo (parse sẵn). Trả False nếu parse lỗi (bỏ qua)."""
        try:
            node = parse_expression(expression)
        except ValueError as exc:
            logger.debug("Zoo bỏ qua biểu thức parse lỗi: {} — {}", expression, exc)
            return False
        self._entries.append((expression, node))
        return True

    def most_similar(self, expression: str) -> tuple[str | None, float]:
        """(biểu thức gần nhất trong zoo, tỉ lệ tương đồng). Zoo rỗng -> (None, 0)."""
        try:
            target = parse_expression(expression)
        except ValueError:
            return (None, 1.0)  # parse lỗi -> coi như không độc đáo (an toàn)
        best_expr, best_ratio = None, 0.0
        for expr, node in self._entries:
            # field_aware: dataset/field thay thế -> tính là độc đáo (không gộp về "F").
            r = similarity_ratio(target, node, field_aware=True)
            if r > best_ratio:
                best_expr, best_ratio = expr, r
        return (best_expr, best_ratio)

    def originality(self, expression: str) -> float:
        """1 - tương đồng cao nhất so với zoo. Zoo rỗng -> 1.0 (hoàn toàn độc đáo)."""
        if not self._entries:
            return 1.0
        _, ratio = self.most_similar(expression)
        return 1.0 - ratio
