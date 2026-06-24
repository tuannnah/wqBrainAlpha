"""Cây AST sealed hierarchy cho FASTEXPR-subset.

Ba loại node: `Constant` (literal số), `Field` (tên cột dữ liệu, vd "close"), `Call`
(operator/hàm với danh sách args). Mỗi node bất biến (frozen+slots) để an toàn dùng làm
khóa cache/hash. Visitor pattern (`NodeVisitor` Protocol) tách mọi phân tích (depth, hash,
serialize, eval...) ra khỏi node — open/closed: thêm phân tích mới = thêm visitor, không
sửa node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, TypeVar

T = TypeVar("T")
# T_co covariant: NodeVisitor chỉ dùng T ở vị trí RETURN (output của visit_*), nên
# theo luật variance phải khai báo covariant — mypy --strict đòi điều này, và nó cho
# phép NodeVisitor[set[str]] được coi là NodeVisitor[AbstractSet[str]] (cần cho `|=`).
T_co = TypeVar("T_co", covariant=True)


class NodeVisitor(Protocol[T_co]):
    """Hợp đồng visitor: một phương thức `visit_*` cho mỗi loại node cụ thể."""

    def visit_constant(self, node: Constant) -> T_co: ...

    def visit_field(self, node: Field) -> T_co: ...

    def visit_call(self, node: Call) -> T_co: ...


class Node(ABC):
    """Nút trừu tượng của AST. Không tự sealed bằng cú pháp Python, nhưng chỉ
    `Constant/Field/Call` được định nghĩa trong module này — coi là sealed theo quy ước."""

    @abstractmethod
    def accept(self, v: NodeVisitor[T]) -> T:
        """Gọi đúng phương thức `visit_*` tương ứng loại node cụ thể (double dispatch)."""
        raise NotImplementedError

    @abstractmethod
    def children(self) -> tuple[Node, ...]:
        """Các node con trực tiếp; rỗng cho leaf (`Constant`/`Field`)."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Constant(Node):
    """Literal số (window int hoặc threshold float) — leaf, không có con."""

    value: float

    def accept(self, v: NodeVisitor[T]) -> T:
        return v.visit_constant(self)

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class Field(Node):
    """Tham chiếu tới một cột dữ liệu thị trường theo tên (vd "close", "volume") — leaf."""

    name: str

    def accept(self, v: NodeVisitor[T]) -> T:
        return v.visit_field(self)

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class Call(Node):
    """Lời gọi operator/hàm: `op` phải tồn tại trong OperatorRegistry; `args` định vị,
    có thể trộn sub-expression (Field/Call) và literal (Constant)."""

    op: str
    args: tuple[Node, ...]

    def accept(self, v: NodeVisitor[T]) -> T:
        return v.visit_call(self)

    def children(self) -> tuple[Node, ...]:
        return self.args
