"""Parser tập con FASTEXPR và tạo structural fingerprint cho Alpha."""

import hashlib
from dataclasses import dataclass
from typing import List, Set

from lark import Lark, Token, Tree
from lark.exceptions import LarkError


class ExpressionSyntaxError(ValueError):
    """Biểu thức Alpha sai cú pháp."""


GRAMMAR = r"""
?start: expr
?expr: logical_or
?logical_or: logical_and ("||" logical_and)*
?logical_and: comparison ("&&" comparison)*
?comparison: sum (COMP_OP sum)*
?sum: product (ADD_OP product)*
?product: power (MUL_OP power)*
?power: unary ("^" unary)*
?unary: ("+" | "-" | "!") unary | atom
?atom: function_call | NAME | NUMBER | STRING | "(" expr ")"
function_call: NAME "(" [argument ("," argument)*] ")"
?argument: NAME "=" expr -> keyword_argument
         | expr
COMP_OP: "<=" | ">=" | "==" | "!=" | "<" | ">"
ADD_OP: "+" | "-"
MUL_OP: "*" | "/"
NAME: /[A-Za-z_][A-Za-z0-9_]*/
NUMBER: /\d+(\.\d+)?/
STRING: /'([^'\\]|\\.)*'/ | /"([^"\\]|\\.)*"/
%import common.WS
%ignore WS
"""

_PARSER = Lark(GRAMMAR, parser="lalr")


@dataclass(frozen=True)
class ParsedExpression:
    expression: str
    normalized_expression: str
    expression_hash: str
    fingerprint: str
    operator_names: Set[str]
    identifiers: Set[str]
    tokens: List[str]


def _tokens(expression):
    try:
        return list(_PARSER.lex(expression))
    except LarkError as exc:
        raise ExpressionSyntaxError(str(exc)) from exc


def _tree(expression):
    try:
        return _PARSER.parse(expression)
    except LarkError as exc:
        raise ExpressionSyntaxError(str(exc)) from exc


def _collect(node, operators, keyword_names):
    if isinstance(node, Token):
        return
    if not isinstance(node, Tree):
        return
    if node.data == "function_call":
        operators.add(str(node.children[0]))
        for child in node.children[1:]:
            _collect(child, operators, keyword_names)
        return
    if node.data == "keyword_argument":
        keyword_names.add(str(node.children[0]))
        for child in node.children[1:]:
            _collect(child, operators, keyword_names)
        return
    for child in node.children:
        _collect(child, operators, keyword_names)


def _normalize(tokens):
    return "".join(str(token) for token in tokens)


def parse_expression(expression):
    tokens = _tokens(expression)
    tree = _tree(expression)

    operators = set()
    keyword_names = set()
    _collect(tree, operators, keyword_names)

    all_names = {str(token) for token in tokens if token.type == "NAME"}
    identifiers = all_names - operators - keyword_names

    normalized = _normalize(tokens)
    expression_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    structural = _fingerprint_from_tokens(tokens, identifiers)

    return ParsedExpression(
        expression=expression,
        normalized_expression=normalized,
        expression_hash=expression_hash,
        fingerprint=structural,
        operator_names=operators,
        identifiers=identifiers,
        tokens=[str(token) for token in tokens],
    )


def _fingerprint_from_tokens(tokens, field_ids):
    parts = []
    for token in tokens:
        if token.type == "NUMBER":
            parts.append("$NUMBER")
        elif token.type == "NAME" and str(token) in field_ids:
            parts.append("$FIELD")
        else:
            parts.append(str(token))
    return "".join(parts)


def fingerprint(expression, field_ids):
    """Fingerprint thay field đã biết bằng $FIELD và số bằng $NUMBER."""

    tokens = _tokens(expression)
    return _fingerprint_from_tokens(tokens, set(field_ids))


@dataclass(frozen=True)
class FieldContext:
    name: str
    ancestors: List[str]
    parent_operator: str


def iter_field_contexts(expression):
    """Liệt kê mỗi identifier kèm chuỗi operator tổ tiên và operator cha trực tiếp."""

    tree = _tree(expression)
    results = []
    _walk_contexts(tree, [], results)
    return results


def _walk_contexts(node, ancestors, results):
    if isinstance(node, Token):
        return
    if not isinstance(node, Tree):
        return

    if node.data == "function_call":
        operator = str(node.children[0])
        inner_ancestors = ancestors + [operator]
        for child in node.children[1:]:
            _walk_child(child, inner_ancestors, operator, results)
        return

    if node.data == "keyword_argument":
        for child in node.children[1:]:
            _walk_child(child, ancestors, ancestors[-1] if ancestors else None, results)
        return

    parent = ancestors[-1] if ancestors else None
    for child in node.children:
        _walk_child(child, ancestors, parent, results)


def _walk_child(child, ancestors, parent_operator, results):
    if isinstance(child, Token):
        if child.type == "NAME":
            results.append(FieldContext(str(child), list(ancestors), parent_operator))
        return
    _walk_contexts(child, ancestors, results)
