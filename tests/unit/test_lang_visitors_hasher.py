"""Test CanonicalHasher: hash ổn định, sort args commutative, normalize literal."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field
from src.lang.registry import ArgKind, OpCategory, OperatorRegistry, OperatorSpec
from src.lang.visitors import CanonicalHasher


def _placeholder(*_a: object) -> object:
    raise NotImplementedError


def _registry() -> OperatorRegistry:
    reg = OperatorRegistry()
    reg.register(OperatorSpec(
        name="add", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_placeholder, bounded=False,
        commutative=True,
    ))
    reg.register(OperatorSpec(
        name="subtract", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_placeholder, bounded=False,
        commutative=False,
    ))
    return reg


def test_hash_is_deterministic_for_same_tree():
    reg = _registry()
    tree = Call(op="add", args=(Field("close"), Field("open")))
    h1 = CanonicalHasher(reg).visit(tree)
    h2 = CanonicalHasher(reg).visit(tree)
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64  # sha256 hex digest


def test_hash_differs_for_different_trees():
    reg = _registry()
    t1 = Call(op="add", args=(Field("close"), Field("open")))
    t2 = Call(op="add", args=(Field("close"), Field("volume")))
    assert CanonicalHasher(reg).visit(t1) != CanonicalHasher(reg).visit(t2)


def test_hash_same_for_commutative_args_swapped():
    reg = _registry()
    t1 = Call(op="add", args=(Field("close"), Field("open")))
    t2 = Call(op="add", args=(Field("open"), Field("close")))
    assert CanonicalHasher(reg).visit(t1) == CanonicalHasher(reg).visit(t2)


def test_hash_differs_for_non_commutative_args_swapped():
    reg = _registry()
    t1 = Call(op="subtract", args=(Field("close"), Field("open")))
    t2 = Call(op="subtract", args=(Field("open"), Field("close")))
    assert CanonicalHasher(reg).visit(t1) != CanonicalHasher(reg).visit(t2)


def test_hash_normalizes_literal_representation():
    reg = _registry()
    # Constant sinh từ "20" hay từ float(20) đều cùng giá trị 20.0 -> cùng hash.
    t1 = Call(op="add", args=(Field("close"), Constant(20.0)))
    t2 = Call(op="add", args=(Field("close"), Constant(float("20"))))
    assert CanonicalHasher(reg).visit(t1) == CanonicalHasher(reg).visit(t2)


def test_hash_unknown_op_falls_back_non_commutative():
    reg = _registry()  # "mystery_op" không đăng ký
    t1 = Call(op="mystery_op", args=(Field("close"), Field("open")))
    t2 = Call(op="mystery_op", args=(Field("open"), Field("close")))
    assert CanonicalHasher(reg).visit(t1) != CanonicalHasher(reg).visit(t2)
