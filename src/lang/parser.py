"""Parser FASTEXPR-subset: chuỗi -> AST (Node), validate operator/arity qua OperatorRegistry.

Dùng `lark` với grammar `grammar.lark` (Task 1.4) để parse cú pháp; một `lark.Transformer`
biến `lark.Tree` thành `Node` (Constant/Field/Call). Toán tử nhị phân `+ - * /` map sang
operator registry `add/subtract/multiply/divide` để mọi computation đi qua MỘT con đường
(Call) — không có node BinOp riêng.

Chạy độc lập: `python -m src.lang.parser "rank(close)"`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lark import Lark, Token, Transformer
from lark.exceptions import UnexpectedInput, VisitError

from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import OperatorRegistry, default_registry

_GRAMMAR_PATH = Path(__file__).resolve().parent / "grammar.lark"
_GRAMMAR_TEXT = _GRAMMAR_PATH.read_text(encoding="utf-8")

_BINARY_OP_NAME = {
    "add_expr": "add",
    "sub_expr": "subtract",
    "mul_expr": "multiply",
    "div_expr": "divide",
}


class ParseError(ValueError):
    """Lỗi parse: cú pháp sai, operator không tồn tại, hoặc sai số lượng đối số."""


class _ToAst(Transformer[Token, Node]):
    """Biến lark.Tree (theo grammar.lark) thành cây Node, validate qua registry khi gặp Call."""

    def __init__(self, registry: OperatorRegistry) -> None:
        super().__init__()
        self._registry = registry

    def start(self, children: list[Node]) -> Node:
        return children[0]

    def number_atom(self, children: list[Token]) -> Constant:
        return Constant(float(children[0]))

    def field(self, children: list[Token]) -> Field:
        return Field(str(children[0]))

    def call(self, children: list[object]) -> Call:
        name = str(children[0])
        args = tuple(c for c in children[1:] if isinstance(c, Node))
        self._validate(name, len(args))
        return Call(op=name, args=args)

    def add_expr(self, children: list[Node]) -> Call:
        return self._binary("add_expr", children)

    def sub_expr(self, children: list[Node]) -> Call:
        return self._binary("sub_expr", children)

    def mul_expr(self, children: list[Node]) -> Call:
        return self._binary("mul_expr", children)

    def div_expr(self, children: list[Node]) -> Call:
        return self._binary("div_expr", children)

    def _binary(self, rule_name: str, children: list[Node]) -> Call:
        op_name = _BINARY_OP_NAME[rule_name]
        left, right = children
        self._validate(op_name, 2)
        return Call(op=op_name, args=(left, right))

    def _validate(self, name: str, n_args: int) -> None:
        try:
            spec = self._registry.get(name)
        except KeyError as exc:
            raise ParseError(f"operator không tồn tại trong registry: {name!r}") from exc
        if len(spec.signature) != n_args:
            raise ParseError(
                f"sai arity cho operator {name!r}: cần {len(spec.signature)} đối số "
                f"({[k.name for k in spec.signature]}), nhận {n_args}"
            )


def _build_lark(start: str = "start") -> Lark:
    return Lark(_GRAMMAR_TEXT, parser="lalr", start=start)


_LARK = _build_lark()


def parse(text: str, registry: OperatorRegistry | None = None) -> Node:
    """Parse chuỗi FASTEXPR-subset thành AST; raise ParseError nếu cú pháp/operator/arity sai."""
    reg = registry if registry is not None else default_registry()
    try:
        tree = _LARK.parse(text)
    except UnexpectedInput as exc:
        raise ParseError(f"cú pháp không hợp lệ tại: {text!r} ({exc})") from exc
    try:
        result = _ToAst(reg).transform(tree)
    except VisitError as exc:
        # Lark bọc ParseError (raise từ trong _validate) bằng VisitError; gỡ lớp bọc.
        if isinstance(exc.orig_exc, ParseError):
            raise exc.orig_exc from None
        raise
    if not isinstance(result, Node):
        raise ParseError(f"không parse được thành AST hợp lệ: {text!r}")
    return result


if __name__ == "__main__":
    from src.lang.visitors import Serializer

    expr = sys.argv[1] if len(sys.argv) > 1 else "rank(close)"
    node = parse(expr)
    print(Serializer().visit(node))
