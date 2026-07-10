"""OperatorRegistry — nguồn sự thật duy nhất về operator FASTEXPR-subset.

Parser (Phase 1) dùng registry để validate operator tồn tại + arity. Evaluator (Phase 2)
dùng để dispatch impl thật. GP (Phase 7) dùng `gp_function_set()` để xây function set
(chỉ operator lõi, loại các wrapper config như group_neutralize/scale).

Ranh giới Phase 1: registry chỉ đăng ký SPEC (khai báo), `impl` của operator còn thiếu
trong Phase 1 là placeholder raise NotImplementedError — Phase 2 mới nạp impl thật qua
decorator `@register(...)` đặt lên hàm trong `src/operators_local/*.py`.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from loguru import logger


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

    def all_specs(self) -> dict[str, OperatorSpec]:
        """Bản sao {tên -> OperatorSpec} của MỌI op đã đăng ký — dùng cho các thao tác
        cần chụp toàn bộ trạng thái registry (vd baseline của guard catalog), không cho
        code ngoài class đụng trực tiếp vào `_ops`."""
        return dict(self._ops)


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


# Baseline PRISTINE gp_usable của từng registry (chụp NGAY LẦN GỌI GUARD ĐẦU TIÊN có
# catalog thật, trước khi guard mutate bất cứ spec nào). WeakKeyDictionary để không giữ
# registry (vd instance tạo riêng trong test) sống mãi sau khi nó hết được tham chiếu, và
# để mỗi registry (kể cả nhiều instance test khác nhau) có baseline độc lập của riêng nó.
_gp_usable_baseline: "weakref.WeakKeyDictionary[OperatorRegistry, dict[str, bool]]" = (
    weakref.WeakKeyDictionary()
)
# Kết quả offending của lần gọi guard gần nhất theo registry — chỉ để tránh log cảnh báo
# lặp lại khi gọi lại với catalog giống hệt lần trước (không ảnh hưởng tính đúng đắn).
_last_offending: "weakref.WeakKeyDictionary[OperatorRegistry, frozenset[str]]" = (
    weakref.WeakKeyDictionary()
)


def enforce_gp_vocab_against_catalog(
    registry: OperatorRegistry, catalog_names: Iterable[str] | None,
) -> list[str]:
    """Lưới chặn TỔNG QUÁT: đối chiếu vocab GP hiện có với catalog operator THẬT của tài
    khoản Brain (nạp qua ``OperatorRepository.load_cached()`` ở main.py). Operator nào GP
    có thể emit nhưng KHÔNG có trong catalog live sẽ luôn bị Brain từ chối khi sim (tốn phí
    pre-sim vô ích, vd bug ``ts_std`` vs ``ts_std_dev`` đã gặp) — hàm này đánh
    ``gp_usable=False`` NGAY TRÊN ``registry`` cho các op lệch (loại khỏi vocab GP từ lần
    gọi ``gp_function_set()`` kế tiếp) và log cảnh báo nêu tên op lệch. Gọi hàm này mỗi khi
    catalog live sẵn sàng (vd đầu closed-loop, sau khi nạp ``operators`` từ
    ``_cached_symbols``) — trong menu tương tác của main.py (`while True`), hàm này có thể
    được gọi NHIỀU LẦN trên CÙNG MỘT registry singleton (`default_registry()`) qua các
    vòng lặp session khác nhau, với catalog có thể khác nhau mỗi lần (vd catalog chưa tải
    xong ở lần đầu, đầy đủ hơn ở lần sau).

    QUAN TRỌNG — IDEMPOTENT/RE-EVALUABLE (fix follow-up commit d3ad0e3): hàm KHÔNG được
    dồn tích (compound) qua nhiều lần gọi. Nếu tính offending dựa trên
    ``registry.gp_function_set()`` HIỆN TẠI (đã có thể bị chính lần gọi trước đó mutate),
    một catalog thoáng qua/thiếu sót ở lần gọi đầu sẽ loại oan một op, và lần gọi sau dù
    catalog đã đầy đủ hơn cũng KHÔNG BAO GIỜ phục hồi được op đó (registry là singleton
    module-level, sống suốt vòng đời process, re-import là no-op) — mất vocab GP vĩnh
    viễn, chỉ khắc phục được bằng cách khởi động lại process. Để tránh việc này, hàm chụp
    một baseline PRISTINE (gp_usable gốc, độc lập catalog) của MỌI op trong registry ở lần
    gọi có-catalog đầu tiên, và MỌI lần gọi sau đều tính lại
    ``effective_gp_usable = baseline_gp_usable AND (tên op có trong catalog hiện tại)``
    từ baseline đó — không phải từ state đã mutate. Nhờ vậy một op bị loại ở lần gọi trước
    (do catalog lúc đó thiếu nó) sẽ được ĐƯA TRỞ LẠI vocab GP ngay khi một catalog đầy đủ
    hơn xuất hiện; và gọi lặp lại với catalog giống hệt là no-op ổn định.

    ``ts_std`` (và các op khác đăng ký sẵn ``gp_usable=False``, vd wrapper stage-separation
    như ``regression_neut``) không bị ảnh hưởng bởi cơ chế này: baseline của chúng vốn đã
    là ``False`` (đây là quyết định thiết kế tại nơi ĐĂNG KÝ op, độc lập catalog), nên
    ``effective_gp_usable`` luôn ``False`` bất kể catalog — chúng không bao giờ được "phục
    hồi" vào vocab GP qua guard này.

    ``catalog_names`` rỗng/``None`` (chưa đăng nhập/chưa tải catalog/test offline) ->
    KHÔNG làm gì và trả ``[]`` — hàm KHÔNG BAO GIỜ crash toàn app khi thiếu DB/catalog, để
    test và chạy offline vẫn hoạt động bình thường. Một lời gọi catalog rỗng/None cũng
    KHÔNG chụp baseline (baseline chỉ chụp khi có catalog thật để đối chiếu).

    Trả về danh sách tên operator (baseline gp_usable=True) hiện KHÔNG có trong catalog —
    tức đang bị loại khỏi vocab GP theo catalog lần gọi này (rỗng nếu vocab đã khớp)."""
    if not catalog_names:
        return []
    catalog = set(catalog_names)

    baseline = _gp_usable_baseline.get(registry)
    if baseline is None:
        # Lần gọi guard đầu tiên (có catalog) trên registry này -> chụp baseline PRISTINE
        # trước khi mutate bất cứ gì.
        baseline = {name: spec.gp_usable for name, spec in registry.all_specs().items()}
        _gp_usable_baseline[registry] = baseline
    else:
        # Op mới xuất hiện trong registry sau lần chụp đầu (vd operators_local nạp thêm
        # op chưa từng thấy) -> bổ sung baseline cho op đó bằng gp_usable hiện tại của nó.
        # KHÔNG ghi đè baseline của op đã có sẵn trong dict — nếu ghi đè, baseline sẽ bị
        # "nhiễm" bởi chính mutation của guard ở lần gọi trước, quay lại đúng cái bug đang
        # sửa (dồn tích qua nhiều lần gọi).
        for name, spec in registry.all_specs().items():
            baseline.setdefault(name, spec.gp_usable)

    offending = sorted(
        name for name, was_gp_usable in baseline.items()
        if was_gp_usable and name not in catalog
    )

    # Áp lại effective gp_usable = baseline AND (có trong catalog) cho MỌI op có baseline
    # True — kể cả op đã bị lần gọi TRƯỚC loại nhưng nay lại có mặt trong catalog (đây là
    # bước "re-include" giải quyết bug review nêu).
    for name, was_gp_usable in baseline.items():
        if not was_gp_usable:
            continue
        desired_gp_usable = name in catalog
        spec = registry.get(name)
        if spec.gp_usable != desired_gp_usable:
            registry.register(dataclasses.replace(spec, gp_usable=desired_gp_usable))

    offending_key = frozenset(offending)
    if offending_key and offending_key != _last_offending.get(registry):
        logger.warning(
            "GP vocab lệch catalog Brain live -> loại khỏi vocab GP (Brain sẽ từ chối "
            "operator này khi sim): {}",
            ", ".join(offending),
        )
    _last_offending[registry] = offending_key

    return offending
