"""base_seed cho vòng kín: hardcode 42 khiến GP luôn sinh CÙNG quần thể mỗi lần chạy
tiến trình mới -> khi pool tích lũy, quần thể đó bị decorrelate loại hết -> no_more_ideas
(0 ý tưởng). `_resolve_base_seed` cho phép ngẫu nhiên hóa (None/0) để mỗi lần chạy khác
nhau, nhưng vẫn tái lập được nếu chỉ định số cụ thể."""

from __future__ import annotations

import main


def test_giu_nguyen_seed_cu_the_de_tai_lap() -> None:
    assert main._resolve_base_seed(42) == 42
    assert main._resolve_base_seed(1234) == 1234


def test_0_hoac_none_cho_seed_ngau_nhien_hop_le() -> None:
    for val in (0, None):
        s = main._resolve_base_seed(val)
        assert isinstance(s, int)
        assert 1 <= s < 2**31  # nằm trong khoảng seed hợp lệ cho numpy/GP


def test_ngau_nhien_khac_nhau_qua_nhieu_lan() -> None:
    # Xác suất 20 lần ra cùng một giá trị ~0 -> phải có >1 giá trị phân biệt.
    vals = {main._resolve_base_seed(0) for _ in range(20)}
    assert len(vals) > 1
