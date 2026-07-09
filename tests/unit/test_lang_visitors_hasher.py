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
    reg.register(OperatorSpec(
        name="multiply", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_placeholder, bounded=False,
        commutative=True,
    ))
    reg.register(OperatorSpec(
        name="divide", category=OpCategory.ARITHMETIC,
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


# --- Pha 1.1: fold scale DƯƠNG Ở GỐC (whole-alpha) ---
# WQ rank/normalize book ở tầng alpha -> nhân cả tín hiệu với hằng dương KHÔNG đổi positions
# -> cùng dedup_key. Chỉ áp ở GỐC; scale chôn trong add/subtract là TRỌNG SỐ TƯƠNG ĐỐI, KHÔNG
# bất biến -> KHÔNG fold (tránh gộp nhầm 2 alpha khác nhau).

def test_fold_multiply_hang_duong_o_goc_cung_hash():
    reg = _registry()
    t1 = Call(op="multiply", args=(Constant(4.0), Field("close")))
    t2 = Call(op="multiply", args=(Constant(2.0), Field("close")))
    t_plain = Field("close")
    h = CanonicalHasher(reg)
    assert h.visit(t1) == h.visit(t2) == h.visit(t_plain)


def test_fold_multiply_hang_duong_ca_hai_ben():
    """multiply(X, 3) cũng fold như multiply(3, X) (giao hoán) và như X."""
    reg = _registry()
    t1 = Call(op="multiply", args=(Field("close"), Constant(3.0)))
    assert CanonicalHasher(reg).visit(t1) == CanonicalHasher(reg).visit(Field("close"))


def test_fold_divide_hang_duong_o_goc():
    """divide(X, 2) = scale dương 0.5 toàn tín hiệu -> bất biến -> fold về X."""
    reg = _registry()
    t1 = Call(op="divide", args=(Field("close"), Constant(2.0)))
    assert CanonicalHasher(reg).visit(t1) == CanonicalHasher(reg).visit(Field("close"))


def test_KHONG_fold_hang_am():
    """multiply(-1, X) đảo dấu = alpha KHÁC -> phải khác hash với X."""
    reg = _registry()
    t1 = Call(op="multiply", args=(Constant(-1.0), Field("close")))
    assert CanonicalHasher(reg).visit(t1) != CanonicalHasher(reg).visit(Field("close"))


def test_KHONG_fold_scale_chon_trong_add():
    """add(multiply(2,A), B) != add(multiply(1,A), B): trọng số tương đối khác = alpha khác.
    Đây là 2 VERIFIED_CORES thật -> KHÔNG được gộp nhầm."""
    reg = _registry()
    a2 = Call(op="add", args=(Call(op="multiply", args=(Constant(2.0), Field("close"))),
                              Field("open")))
    a1 = Call(op="add", args=(Call(op="multiply", args=(Constant(1.0), Field("close"))),
                              Field("open")))
    assert CanonicalHasher(reg).visit(a2) != CanonicalHasher(reg).visit(a1)


def test_KHONG_fold_divide_hang_o_tu_so():
    """divide(2, X) = 2/X KHÔNG phải scale dương của X (phi tuyến) -> không fold."""
    reg = _registry()
    t1 = Call(op="divide", args=(Constant(2.0), Field("close")))
    assert CanonicalHasher(reg).visit(t1) != CanonicalHasher(reg).visit(Field("close"))
