"""OperatorRegistry — nguồn sự thật duy nhất về operator FASTEXPR-subset.

Parser (Phase 1) dùng registry để validate operator tồn tại + arity. Evaluator (Phase 2)
dùng để dispatch impl thật. GP (Phase 7) dùng `gp_function_set()` để xây function set
(chỉ operator lõi, loại các wrapper config như group_neutralize/scale).

Ranh giới Phase 1: registry chỉ đăng ký SPEC (khai báo), `impl` của operator còn thiếu
trong Phase 1 là placeholder raise NotImplementedError — Phase 2 mới nạp impl thật qua
decorator `@register(...)` đặt lên hàm trong `src/operators_local/*.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ArgKind(Enum):
    """Loại đối số dương vị trí của một operator."""

    PANEL = auto()  # sub-expression bay hơi thành (T, N)
    WINDOW = auto()  # số nguyên dương lookback (vd 5, 10, 20)
    SCALAR = auto()  # literal float (ngưỡng, hệ số scale...)
    GROUP = auto()  # tên group key (vd "sector")


class OpCategory(Enum):
    """Phân nhóm operator — dùng để lọc function set GP và áp luật stage-separation."""

    ARITHMETIC = auto()
    CROSS_SECTIONAL = auto()
    TIME_SERIES = auto()
    GROUP = auto()
    NEUTRALIZATION = auto()
    CONDITIONAL = auto()
    SCALING = auto()


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Khai báo đầy đủ một operator: tên, nhóm, chữ ký đối số, impl, và các cờ cho GP/gate."""

    name: str
    category: OpCategory
    signature: tuple[ArgKind, ...]
    impl: Callable[..., Any]
    bounded: bool
    depth_cost: int = 1
    gp_usable: bool = True
    window_choices: tuple[int, ...] = (5, 10, 20, 60, 120)
    commutative: bool = False


class OperatorRegistry:
    """Bảng tra operator theo tên; nguồn sự thật cho parser/evaluator/GP."""

    def __init__(self) -> None:
        self._ops: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        """Đăng ký một OperatorSpec; ghi đè nếu tên đã tồn tại (cho phép redefinition test)."""
        self._ops[spec.name] = spec

    def get(self, name: str) -> OperatorSpec:
        """Trả OperatorSpec theo tên; raise KeyError với thông điệp rõ nếu không tồn tại."""
        try:
            return self._ops[name]
        except KeyError as exc:
            raise KeyError(f"operator không tồn tại trong registry: {name!r}") from exc

    def by_category(self, c: OpCategory) -> list[OperatorSpec]:
        """Mọi OperatorSpec thuộc category `c`, theo thứ tự đăng ký."""
        return [spec for spec in self._ops.values() if spec.category is c]

    def gp_function_set(self) -> list[OperatorSpec]:
        """Operator lõi dùng được cho GP (gp_usable=True) — loại wrapper config."""
        return [spec for spec in self._ops.values() if spec.gp_usable]


REGISTRY = OperatorRegistry()


def register(
    name: str,
    category: OpCategory,
    signature: tuple[ArgKind, ...],
    bounded: bool,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: đăng ký hàm bên dưới làm impl của operator `name` vào REGISTRY toàn cục,
    trả lại hàm gốc không đổi (để vẫn gọi/test được trực tiếp)."""

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY.register(
            OperatorSpec(
                name=name, category=category, signature=signature,
                impl=fn, bounded=bounded, **kwargs,
            )
        )
        return fn

    return _wrap


def _not_implemented(*_args: Any, **_kwargs: Any) -> Any:
    """Impl placeholder cho operator Phase 1 chưa có logic thật (đó là việc của Phase 2)."""
    raise NotImplementedError("Impl operator thuộc Phase 2 (Operator Engine)")


def _register_phase1_minimal_ops() -> None:
    """Đăng ký tập operator tối thiểu cần để Parser (Phase 1) validate được arity/tồn tại.

    Đây KHÔNG phải danh sách operator đầy đủ của MiniBrain — chỉ đủ cho test parse của
    Phase 1 (rank/ts_mean/4 phép số học nhị phân). Phase 2 đăng ký toàn bộ operator thật
    qua `src/operators_local/*.py` (ghi đè placeholder này bằng impl thật cùng tên).
    """
    REGISTRY.register(OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_not_implemented, bounded=True,
    ))
    REGISTRY.register(OperatorSpec(
        name="ts_mean", category=OpCategory.TIME_SERIES,
        signature=(ArgKind.PANEL, ArgKind.WINDOW), impl=_not_implemented, bounded=False,
    ))
    REGISTRY.register(OperatorSpec(
        name="add", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_not_implemented, bounded=False,
        commutative=True,
    ))
    REGISTRY.register(OperatorSpec(
        name="subtract", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_not_implemented, bounded=False,
        commutative=False,
    ))
    REGISTRY.register(OperatorSpec(
        name="multiply", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_not_implemented, bounded=False,
        commutative=True,
    ))
    REGISTRY.register(OperatorSpec(
        name="divide", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_not_implemented, bounded=False,
        commutative=False,
    ))


_register_phase1_minimal_ops()


def default_registry() -> OperatorRegistry:
    """Registry toàn cục với tập operator tối thiểu Phase 1 đã đăng ký sẵn."""
    return REGISTRY
