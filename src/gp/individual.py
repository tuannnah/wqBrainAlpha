"""Individual — bọc một Node (signal core AST, Phase 1) cùng FitnessVector đã cache (nếu
đã eval) và số thế hệ sinh ra. Đây là đơn vị quần thể của GPEngine (Task 7.7); mọi biến đổi
(crossover/mutation, Task 7.5) tạo Individual MỚI từ Node mới, không sửa expr tại chỗ —
giữ tính bất biến của AST (Phase 1, frozen+slots) lan ra cả tầng GP.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.lang.ast import Node
from src.lang.visitors import CanonicalHasher, ComplexityVisitor, DepthVisitor


@dataclass(slots=True)
class Individual:
    """Một cá thể quần thể GP: signal core (`expr`) + fitness đã eval (nếu có)."""

    expr: Node
    # NOTE: kiểu thật là ``FitnessVector | None`` (Task 7.2). Tại Task 7.1 module
    # ``src.gp.fitness_vec`` CHƯA tồn tại, nên dùng ``object | None`` để mypy --strict
    # sạch (không import module thiếu, không cần ``type: ignore`` — vốn sẽ tự thành lỗi
    # "unused" khi 7.2 thêm import). Hành vi trùng test: fitness chỉ được gán placeholder
    # ``object()``. Khi 7.2 xong, siết lại thành ``FitnessVector | None``.
    fitness: object | None = None
    generation: int = 0

    def canonical_hash(self) -> str:
        return CanonicalHasher().visit(self.expr)

    def depth(self) -> int:
        return DepthVisitor().visit(self.expr)

    def complexity(self) -> int:
        return ComplexityVisitor().visit(self.expr)

    def is_evaluated(self) -> bool:
        return self.fitness is not None
