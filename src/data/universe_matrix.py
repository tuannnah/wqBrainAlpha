"""Ma trận region/universe/delay đã biết của WQB + sinh tổ hợp để warm-cache.

Bảng hằng là nguồn sự thật để DUYỆT; tổ hợp tài khoản không có quyền sẽ được
phát hiện qua probe lúc fetch (warm_cache đánh dấu no_access). Vì vậy bảng không
cần khớp tuyệt đối với quyền tài khoản — chỉ cần bao phủ rộng. Bổ sung/sửa khi
WQB thay đổi danh mục universe.
"""

from __future__ import annotations

from typing import Iterator

# region -> universes hay dùng + các delay khả dụng.
WQB_MATRIX: dict[str, dict] = {
    "USA": {"universes": ["TOP3000", "TOP1000", "TOP500", "TOP200"], "delays": [0, 1]},
    "EUR": {"universes": ["TOP2500", "TOP1200", "TOP800", "TOP400"], "delays": [0, 1]},
    "GLB": {"universes": ["TOP3000", "MINVOL1M"], "delays": [1]},
    "ASI": {"universes": ["MINVOL1M", "ILLIQUID_MINVOL1M"], "delays": [1]},
    "CHN": {"universes": ["TOP2000U"], "delays": [0, 1]},
    "JPN": {"universes": ["TOP1600", "TOP1200", "TOP800"], "delays": [0, 1]},
    "KOR": {"universes": ["TOP600"], "delays": [0, 1]},
    "TWN": {"universes": ["TOP500"], "delays": [0, 1]},
    "HKG": {"universes": ["TOP800", "TOP500"], "delays": [0, 1]},
    "AMR": {"universes": ["TOP600"], "delays": [1]},
}


def iter_scopes(
    regions: list[str] | None = None,
    delays: list[int] | None = None,
) -> Iterator[tuple[str, str, int]]:
    """Sinh (region, universe, delay) cho mọi tổ hợp trong WQB_MATRIX.

    regions: lọc theo danh sách region (không phân biệt hoa/thường); None = tất cả.
    delays:  lọc theo danh sách delay; None = tất cả delay của từng region.
    """
    region_filter = {r.upper() for r in regions} if regions else None
    delay_filter = set(delays) if delays is not None else None
    for region, cfg in WQB_MATRIX.items():
        if region_filter is not None and region not in region_filter:
            continue
        for universe in cfg["universes"]:
            for delay in cfg["delays"]:
                if delay_filter is not None and delay not in delay_filter:
                    continue
                yield (region, universe, delay)
