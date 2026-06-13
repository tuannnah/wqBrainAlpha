"""Test ModelRouter: định tuyến tác vụ rẻ/mạnh, gộp usage (GĐ6: T6.3)."""

from __future__ import annotations

from src.llm.router import ModelRouter
from tests.fakes import FakeDeepSeek


def test_route_tac_vu_manh_dung_model_manh():
    cheap = FakeDeepSeek(["rẻ"])
    strong = FakeDeepSeek(["mạnh"])
    router = ModelRouter(cheap=cheap, strong=strong)
    out = router.complete("sys", "user", task="hypothesis")
    assert out == "mạnh"
    assert len(strong.calls) == 1
    assert len(cheap.calls) == 0


def test_route_tac_vu_re_dung_model_re():
    cheap = FakeDeepSeek(["rẻ"])
    strong = FakeDeepSeek(["mạnh"])
    router = ModelRouter(cheap=cheap, strong=strong)
    out = router.complete("sys", "user", task="mutate")
    assert out == "rẻ"
    assert len(cheap.calls) == 1
    assert len(strong.calls) == 0


def test_task_khong_biet_mac_dinh_dung_re():
    cheap = FakeDeepSeek(["rẻ"])
    strong = FakeDeepSeek(["mạnh"])
    router = ModelRouter(cheap=cheap, strong=strong)
    assert router.complete("s", "u", task="khong_ro") == "rẻ"


def test_khong_truyen_task_dung_model_mac_dinh():
    """complete() không task -> tương thích interface cũ (dùng model mặc định = strong)."""
    cheap = FakeDeepSeek(["rẻ"])
    strong = FakeDeepSeek(["mạnh"])
    router = ModelRouter(cheap=cheap, strong=strong, default="strong")
    assert router.complete("s", "u") == "mạnh"


def test_chi_mot_model_thi_route_ve_no():
    """Không có model mạnh riêng -> mọi tác vụ về model rẻ (cấu hình tối thiểu)."""
    cheap = FakeDeepSeek(["rẻ", "rẻ2"])
    router = ModelRouter(cheap=cheap)
    assert router.complete("s", "u", task="hypothesis") == "rẻ"
    assert router.complete("s", "u", task="mutate") == "rẻ2"


def test_usage_gop_tu_cac_model_con():
    cheap = FakeDeepSeek(["rẻ"])
    strong = FakeDeepSeek(["mạnh"])
    cheap.usage.prompt_tokens = 100
    cheap.usage.completion_tokens = 50
    strong.usage.prompt_tokens = 200
    strong.usage.completion_tokens = 80
    router = ModelRouter(cheap=cheap, strong=strong)
    assert router.usage.total_tokens == 430  # 150 + 280


def test_set_route_cau_hinh_duoc():
    cheap = FakeDeepSeek(["rẻ"])
    strong = FakeDeepSeek(["mạnh"])
    router = ModelRouter(cheap=cheap, strong=strong)
    router.set_route("mutate", "strong")  # ép mutate dùng model mạnh
    assert router.complete("s", "u", task="mutate") == "mạnh"
