"""Test grammar.lark load được và parse cú pháp FASTEXPR-subset thành Tree thô."""

from __future__ import annotations

from pathlib import Path

import pytest
from lark import Lark
from lark.exceptions import UnexpectedInput

GRAMMAR_PATH = Path(__file__).resolve().parents[2] / "src" / "lang" / "grammar.lark"


@pytest.fixture(scope="module")
def lark_parser() -> Lark:
    text = GRAMMAR_PATH.read_text(encoding="utf-8")
    return Lark(text, parser="lalr", start="start")


def test_grammar_file_exists():
    assert GRAMMAR_PATH.is_file()


@pytest.mark.parametrize(
    "expr",
    [
        "close",
        "5",
        "5.5",
        "rank(close)",
        "ts_mean(close, 20)",
        "add(close, open)",
        "close + open",
        "close - open",
        "close * 2",
        "close / 2",
        "rank(ts_mean(close, 20))",
        "rank(close) + rank(open)",
    ],
)
def test_grammar_parses_valid_expressions(lark_parser: Lark, expr: str):
    tree = lark_parser.parse(expr)
    assert tree is not None


@pytest.mark.parametrize(
    "expr",
    [
        "",
        "rank(",
        "close +",
        "rank(close,)",
        "@close",
    ],
)
def test_grammar_rejects_invalid_syntax(lark_parser: Lark, expr: str):
    with pytest.raises(UnexpectedInput):
        lark_parser.parse(expr)
