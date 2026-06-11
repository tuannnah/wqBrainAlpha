"""Biểu diễn FASTEXPR dưới dạng cây AST để GA thao tác.

Cây gồm `Node` (operator/hàm hoặc toán tử nhị phân) và `Leaf` (field hoặc số).
Hỗ trợ parse chuỗi → cây và render cây → chuỗi (round-trip đủ để sinh alpha hợp lệ).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BINARY_OPS = {"+", "-", "*", "/"}


@dataclass
class Leaf:
    value: str | int | float

    def copy(self) -> "Leaf":
        return Leaf(self.value)


@dataclass
class Node:
    op: str  # tên hàm ("rank", "ts_delta"), toán tử ("+","-",...) hoặc "neg"
    children: list = field(default_factory=list)

    def copy(self) -> "Node":
        return Node(self.op, [c.copy() for c in self.children])


# --------------------------------------------------------------------- render
def to_expression(node) -> str:
    if isinstance(node, Leaf):
        return str(node.value)
    if node.op == "neg":
        return f"-{to_expression(node.children[0])}"
    if node.op in BINARY_OPS and len(node.children) == 2:
        left, right = node.children
        return f"({to_expression(left)} {node.op} {to_expression(right)})"
    args = ", ".join(to_expression(c) for c in node.children)
    return f"{node.op}({args})"


# ---------------------------------------------------------------------- parse
_TOKEN_RE = re.compile(r"\s*([A-Za-z_]\w*|\d+\.\d+|\d+|[()+\-*/,])")


def _tokenize(expr: str) -> list[str]:
    tokens, pos = [], 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            if expr[pos].isspace():
                pos += 1
                continue
            raise ValueError(f"Ký tự không hợp lệ tại {pos}: {expr[pos]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def next(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, tok):
        if self.peek() != tok:
            raise ValueError(f"Mong đợi {tok!r}, gặp {self.peek()!r}")
        return self.next()

    def parse(self):
        node = self.expression()
        if self.peek() is not None:
            raise ValueError(f"Token thừa: {self.peek()!r}")
        return node

    def expression(self):
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.next()
            node = Node(op, [node, self.term()])
        return node

    def term(self):
        node = self.factor()
        while self.peek() in ("*", "/"):
            op = self.next()
            node = Node(op, [node, self.factor()])
        return node

    def factor(self):
        if self.peek() == "-":
            self.next()
            return Node("neg", [self.factor()])
        return self.primary()

    def primary(self):
        tok = self.peek()
        if tok is None:
            raise ValueError("Hết token bất ngờ")
        if tok == "(":
            self.next()
            node = self.expression()
            self.expect(")")
            return node
        self.next()
        if re.fullmatch(r"\d+\.\d+|\d+", tok):
            value = float(tok) if "." in tok else int(tok)
            return Leaf(value)
        # identifier: có thể là hàm nếu theo sau là '('
        if self.peek() == "(":
            self.next()
            args = []
            if self.peek() != ")":
                args.append(self.expression())
                while self.peek() == ",":
                    self.next()
                    args.append(self.expression())
            self.expect(")")
            return Node(tok, args)
        return Leaf(tok)


def parse_expression(expr: str) -> Node | Leaf:
    return _Parser(_tokenize(expr)).parse()


# ------------------------------------------------------------------ utilities
def tree_depth(node) -> int:
    if isinstance(node, Leaf) or not node.children:
        return 1
    return 1 + max(tree_depth(c) for c in node.children)


def node_count(node) -> int:
    if isinstance(node, Leaf):
        return 1
    return 1 + sum(node_count(c) for c in node.children)


def all_subtrees(node) -> list:
    """Trả mọi sub-node (gồm cả Leaf) — phục vụ chọn điểm crossover."""
    result = [node]
    if isinstance(node, Node):
        for c in node.children:
            result.extend(all_subtrees(c))
    return result


def iter_leaves(node):
    if isinstance(node, Leaf):
        yield node
    else:
        for c in node.children:
            yield from iter_leaves(c)
