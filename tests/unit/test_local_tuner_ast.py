# tests/unit/test_local_tuner_ast.py
from __future__ import annotations

from src.backtest.local_tuner import iter_constants, set_constant
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import Serializer


def test_iter_constants_danh_dau_window():
    import src.operators_local  # noqa: F401  (side-effect: nạp ts_delta vào registry)

    node = parse("add(multiply(2, ts_mean(close, 10)), ts_delta(close, 5))")
    reg = default_registry()
    slots = iter_constants(node, reg)
    values = {round(v, 3): is_win for _, v, is_win in slots}
    assert values[10.0] is True   # window của ts_mean
    assert values[5.0] is True    # window của ts_delta
    assert values[2.0] is False   # hệ số của multiply (vị trí PANEL, không phải WINDOW)


def test_set_constant_thay_dung_o():
    node = parse("ts_mean(close, 10)")
    reg = default_registry()
    (path, value, is_win) = [s for s in iter_constants(node, reg) if s[2]][0]
    assert value == 10.0 and is_win
    new_node = set_constant(node, path, 20)
    assert Serializer().visit(new_node) == "ts_mean(close, 20)"
    # node gốc bất biến
    assert Serializer().visit(node) == "ts_mean(close, 10)"
