"""Test parse/render và tiện ích cây AST."""

from __future__ import annotations

import pytest

from src.generation.ast_utils import (
    Leaf,
    Node,
    all_subtrees,
    node_count,
    parse_expression,
    to_expression,
    tree_depth,
)


@pytest.mark.parametrize(
    "expr",
    [
        "rank(close)",
        "rank(ts_delta(close, 5))",
        "group_neutralize(rank(close), sector)",
        "rank(ts_corr(close, open, 20))",
        "-rank(ts_zscore(volume, 10))",
    ],
)
def test_round_trip_parse_render(expr):
    tree = parse_expression(expr)
    rendered = to_expression(tree)
    # Render lại rồi parse lần nữa phải cho cùng cấu trúc.
    assert to_expression(parse_expression(rendered)) == rendered
    assert parse_expression(rendered) == tree


def test_binary_subtraction_parsed():
    tree = parse_expression("ts_mean(close, 5) - ts_mean(close, 20)")
    assert isinstance(tree, Node) and tree.op == "-"
    assert len(tree.children) == 2


def test_tree_depth_and_node_count():
    tree = parse_expression("rank(ts_delta(close, 5))")
    # rank -> ts_delta -> [close, 5]
    assert tree_depth(tree) == 3
    assert node_count(tree) == 4  # rank, ts_delta, close, 5


def test_all_subtrees_includes_leaves():
    tree = Node("rank", [Leaf("close")])
    subs = all_subtrees(tree)
    assert tree in subs
    assert any(isinstance(s, Leaf) for s in subs)
