"""Menu 5 / closed-loop không trần max_ideas: `no_more_ideas` (cạn ý tưởng TẠM THỜI —
batch GP rỗng sau dedup/family-closed) không được kết thúc phiên như trước; phải reseed
GP rồi dựng lại vòng và chạy tiếp. Phiên chỉ dừng thật khi hết quota Brain
(stop_reason="quota" hoặc QuotaExhausted lan ra) hoặc người dùng Ctrl+C."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import main
from src.pipeline.closed_loop import QuotaExhausted


class _VongGia:
    """Vòng kín giả: run() trả report theo kịch bản định sẵn (hoặc ném exception)."""

    def __init__(self, ket_qua):
        self._ket_qua = ket_qua

    def run(self):
        if isinstance(self._ket_qua, Exception):
            raise self._ket_qua
        return self._ket_qua


def _report(stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason)


def test_no_more_ideas_reseed_va_chay_tiep() -> None:
    """Cạn ý tưởng 2 lần liên tiếp -> dựng lại vòng với seed MỚI mỗi lần, chạy tiếp;
    trả về report cuối cùng khi stop_reason khác no_more_ideas."""
    kich_ban = [_report("no_more_ideas"), _report("no_more_ideas"), _report("quota")]
    seeds_da_dung: list[int] = []

    def build_loop(seed: int) -> _VongGia:
        seeds_da_dung.append(seed)
        return _VongGia(kich_ban[len(seeds_da_dung) - 1])

    reseeds = iter([111, 222])
    report = main._run_reseed_until_quota(
        build_loop, first_seed=42, reseed_fn=lambda _: next(reseeds),
    )

    assert report.stop_reason == "quota"
    assert seeds_da_dung == [42, 111, 222]


def test_stop_reason_khac_khong_reseed() -> None:
    """stop_reason không phải no_more_ideas (vd quota ngay lần đầu) -> trả về luôn,
    không dựng lại vòng lần nào nữa."""
    so_lan_dung = 0

    def build_loop(seed: int) -> _VongGia:
        nonlocal so_lan_dung
        so_lan_dung += 1
        return _VongGia(_report("quota"))

    report = main._run_reseed_until_quota(build_loop, first_seed=7)

    assert report.stop_reason == "quota"
    assert so_lan_dung == 1


def test_quota_exhausted_lan_ra_ngoai() -> None:
    """QuotaExhausted từ run() phải lan ra cho caller (nơi in 'Hết quota Brain'),
    không bị helper nuốt rồi reseed nhầm."""
    def build_loop(seed: int) -> _VongGia:
        return _VongGia(QuotaExhausted("hết quota"))

    with pytest.raises(QuotaExhausted):
        main._run_reseed_until_quota(build_loop, first_seed=1)
