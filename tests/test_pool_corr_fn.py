"""Test helper nối WQ self-correlation vào RefinementLoop (corr first-class 3/3)."""

from __future__ import annotations

from main import _make_pool_corr_fn
from tests.fakes import FakeClient, FakeResponse


def test_make_pool_corr_fn_tra_max_self_corr():
    """Đóng gói CorrelationChecker -> callback trả self-corr lớn nhất từ WQ."""
    client = FakeClient()
    client.queue_get(FakeResponse(200, {"max": 0.83}))
    fn = _make_pool_corr_fn(client)
    assert fn("wq-123") == 0.83
    # gọi đúng endpoint self-correlation của WQ.
    assert client.calls[0][0] == "GET"
    assert "/alphas/wq-123/correlations/self" in client.calls[0][1]


def test_make_pool_corr_fn_loi_mang_tra_none():
    """Lỗi hạ tầng (exception) -> None để loop không chặn nhầm trên trục trặc tạm thời."""

    class _Boom:
        def get(self, *a, **k):
            raise RuntimeError("net down")

    fn = _make_pool_corr_fn(_Boom())
    assert fn("wq-1") is None
