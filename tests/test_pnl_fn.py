"""Test helper lấy daily PnL từ WQ recordset cho regime gate (review 3)."""

from __future__ import annotations

from src.app.cli.research import _make_pnl_fn
from tests.fakes import FakeClient, FakeResponse


def test_make_pnl_fn_diff_cumulative_thanh_daily():
    """WQ trả PnL tích luỹ -> helper diff thành gia số ngày (bỏ điểm gốc)."""
    client = FakeClient()
    client.queue_get(
        FakeResponse(
            200,
            {
                "schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
                "records": [["2020-01-01", 0.0], ["2020-01-02", 1.0], ["2020-01-03", 1.5]],
            },
        )
    )
    fn = _make_pnl_fn(client)
    series = fn("wq-1")
    assert series == [("2020-01-02", 1.0), ("2020-01-03", 0.5)]
    assert "/alphas/wq-1/recordsets/pnl" in client.calls[0][1]


def test_make_pnl_fn_loi_tra_none():
    client = FakeClient()
    client.queue_get(FakeResponse(500, {}))
    fn = _make_pnl_fn(client)
    assert fn("wq-1") is None
