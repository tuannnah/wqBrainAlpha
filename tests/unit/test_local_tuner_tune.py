"""Test cho `tune`/`local_sharpe`/`TuneResult` — coordinate descent quanh MỘT biểu thức.

`eval_fn` inject được để test hành vi coordinate descent (chọn window/hệ số/config tốt
hơn, giữ gốc làm cận dưới, bỏ qua biến thể eval lỗi) mà KHÔNG cần chạy backtest thật.
Test cuối (`test_local_sharpe_ra_so_huu_han_tren_panel_that`) dùng `local_sharpe` thật
trên panel nhỏ tự tạo để xác nhận đường backtest MiniBrain chạy hết không NaN/lỗi.
"""

from __future__ import annotations

from src.backtest.config import PortfolioConfig
from src.backtest.local_tuner import tune


def _cfg() -> PortfolioConfig:
    return PortfolioConfig(decay=4, truncation=0.08)


def test_tune_chon_window_tot_hon():
    # eval_fn: expr chứa "20" tốt hơn; các thứ khác thấp
    def eval_fn(node, config):
        from src.lang.visitors import Serializer
        return 2.0 if "20" in Serializer().visit(node) else 0.5
    res = tune("ts_mean(close, 10)", _cfg(), data=None, budget=40, eval_fn=eval_fn)
    assert "20" in res.best_expr
    assert res.local_sharpe == 2.0


def test_tune_chon_config_tot_hon():
    # mọi expr như nhau (0.5), nhưng truncation=0.02 cho điểm cao hơn
    def eval_fn(node, config):
        return 1.5 if abs(config.truncation - 0.02) < 1e-9 else 0.5
    res = tune("rank(close)", _cfg(), data=None, budget=40, eval_fn=eval_fn)
    assert abs(res.best_config.truncation - 0.02) < 1e-9
    assert res.local_sharpe == 1.5


def test_tune_bat_bien_don_dieu_khong_te_hon_goc():
    # mọi biến thể tệ hơn gốc -> giữ nguyên gốc
    def eval_fn(node, config):
        from src.lang.visitors import Serializer
        return 1.0 if Serializer().visit(node) == "ts_mean(close, 10)" and config.decay == 4 else -5.0
    res = tune("ts_mean(close, 10)", _cfg(), data=None, budget=40, eval_fn=eval_fn)
    assert res.best_expr == "ts_mean(close, 10)"
    assert res.local_sharpe == 1.0


def test_tune_khong_o_chinh_tra_goc():
    def eval_fn(node, config):
        return 0.9
    res = tune("close", _cfg(), data=None, budget=40, eval_fn=eval_fn)
    assert res.best_expr == "close"


def test_tune_bien_the_eval_loi_bi_bo_qua():
    def eval_fn(node, config):
        from src.lang.visitors import Serializer
        s = Serializer().visit(node)
        if "20" in s:
            raise ValueError("giả lập eval lỗi")
        return 1.0
    # không sập; trả gốc (mọi biến thể khác không tốt hơn 1.0)
    res = tune("ts_mean(close, 10)", _cfg(), data=None, budget=40, eval_fn=eval_fn)
    assert res.local_sharpe == 1.0


def test_local_sharpe_ra_so_huu_han_tren_panel_that():
    import numpy as np

    import src.operators_local  # noqa: F401  side-effect: đăng ký operator thật vào registry
    from src.backtest.config import PortfolioConfig as _PortfolioConfig
    from src.backtest.local_tuner import local_sharpe
    from src.data.market_panel import MarketData
    from src.lang.parser import parse
    from src.lang.registry import default_registry

    rng = np.random.default_rng(0)
    t, n = 60, 8
    dates = np.arange("2020-01-01", "2020-03-01", dtype="datetime64[D]")[:t].astype("datetime64[ns]")
    close = 100 + np.cumsum(rng.normal(0, 1, (t, n)), axis=0)
    fields = {name: close.copy() for name in ("close", "open", "high", "low", "vwap")}
    fields["volume"] = np.abs(rng.normal(1e6, 1e5, (t, n)))
    data = MarketData(
        dates=dates, assets=np.array([f"S{i}" for i in range(n)]), fields=fields,
        universe=np.ones((t, n), dtype=bool),
        returns=np.vstack([np.zeros((1, n)), np.diff(close, axis=0) / close[:-1]]),
        groups={"sector": (np.arange(n) % 2).reshape(1, n).repeat(t, axis=0)},
    )
    s = local_sharpe(
        parse("rank(ts_delta(close, 5))"),
        _PortfolioConfig(decay=4, truncation=0.08),
        data,
        default_registry(),
    )
    assert np.isfinite(s)
