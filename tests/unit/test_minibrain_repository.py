# tests/unit/test_minibrain_repository.py
"""Test MiniBrainRepository: upsert_expression dedup, record_evaluation (pass&fail),
load_pool/save_pool_pnl round-trip, dead_field, result_cache hit/miss, top_n."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.metrics_local import AlphaMetrics
from src.simulation.simulator import SimulationResult
from src.storage.db import init_db
from src.storage.models import SubmissionModel
from src.storage.repository import AlphaRepository, MiniBrainRepository


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    session_factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(session_factory)


def _metrics(sharpe=1.5) -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=sharpe, annual_return=0.1, turnover=0.2, max_drawdown=-0.05,
        fitness=2.0, per_year_sharpe={2021: 1.2, 2022: 1.8}, weight_concentration=0.05,
    )


def _cfg_json() -> str:
    cfg = PortfolioConfig(neutralization=Neutralization.SECTOR)
    return json.dumps({"neutralization": cfg.neutralization.name, "decay": cfg.decay,
                        "truncation": cfg.truncation, "scale_book": cfg.scale_book,
                        "delay": cfg.delay})


def test_upsert_expression_dedups_by_canonical_hash(repo):
    id1 = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    id2 = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    assert id1 == id2


def test_upsert_expression_distinct_hash_creates_new_row(repo):
    id1 = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    id2 = repo.upsert_expression("open", "hash2", depth=1, complexity=1, fields={"open"})
    assert id1 != id2


def test_record_evaluation_passed_stores_full_metrics(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    eval_id = repo.record_evaluation(
        expr_id, _cfg_json(), "2020..2021", _metrics(), self_corr_max=0.1,
        status="passed", fail_reasons=[], seed=42,
    )
    assert isinstance(eval_id, int)


def test_record_evaluation_failed_stores_reasons_without_metrics(repo):
    expr_id = repo.upsert_expression("bad(", "hash_bad", depth=0, complexity=0, fields=set())
    eval_id = repo.record_evaluation(
        expr_id, _cfg_json(), "2020..2021", metrics=None, self_corr_max=None,
        status="invalid", fail_reasons=["parse lỗi: unexpected token"], seed=None,
    )
    assert isinstance(eval_id, int)


def test_record_evaluation_upsert_same_key_updates_not_duplicates(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(1.0), 0.1, "passed", [], 1)
    repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(2.0), 0.1, "passed", [], 1)
    cached = repo.result_cache_get("hash1", _cfg_json(), "w1")
    assert cached is not None
    assert cached.sharpe == pytest.approx(2.0)  # ghi đè, không nhân đôi


def test_save_and_load_pool_pnl_roundtrip(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    eval_id = repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(), 0.1, "passed", [], 1)
    dates = np.array(["2021-01-01", "2021-01-02", "2021-01-03"], dtype="datetime64[D]")
    pnl = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    repo.save_pool_pnl(eval_id, dates, pnl)
    pool = repo.load_pool()
    assert eval_id in pool
    loaded_dates, loaded_pnl = pool[eval_id]
    assert np.array_equal(loaded_dates, dates)
    np.testing.assert_allclose(loaded_pnl, pnl)


def test_load_pool_returns_writeable_array_for_inplace_ops(repo):
    """Phase 6 (max_corr trên pool) cần thao tác in-place (vd demean `arr -= mean`) trên
    mảng pnl trả về từ load_pool; np.frombuffer() trần trả mảng read-only và sẽ raise
    ValueError khi bị trừ in-place — load_pool phải trả bản pnl ghi-được (.copy())."""
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    eval_id = repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(), 0.1, "passed", [], 1)
    dates = np.array(["2021-01-01", "2021-01-02", "2021-01-03"], dtype="datetime64[D]")
    pnl = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    repo.save_pool_pnl(eval_id, dates, pnl)

    pool = repo.load_pool()

    _, loaded_pnl = pool[eval_id]
    assert loaded_pnl.flags.writeable is True
    loaded_pnl -= 1.0  # thao tác in-place kiểu Phase 6 max_corr; không được raise
    np.testing.assert_allclose(loaded_pnl, pnl - 1.0)


def test_load_pool_roundtrips_dates_and_pnl_as_tuple(repo):
    """Khóa hành vi: load_pool trả {evaluation_id: (dates, pnl)} đúng format mà
    PoolCorrelation.__init__ (Task 6.1) mong đợi — kể cả khi 2 alpha trong pool có lịch
    sử dài-ngắn khác nhau (alpha A 3 ngày, alpha B 10 ngày, mốc thời gian khác nhau)."""
    expr_a = repo.upsert_expression("close", "hash_a", depth=1, complexity=1, fields={"close"})
    eval_a = repo.record_evaluation(expr_a, _cfg_json(), "w1", _metrics(), 0.1, "passed", [], 1)
    dates_a = np.array(["2021-01-01", "2021-01-02", "2021-01-03"], dtype="datetime64[D]")
    pnl_a = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    repo.save_pool_pnl(eval_a, dates_a, pnl_a)

    expr_b = repo.upsert_expression("open", "hash_b", depth=1, complexity=1, fields={"open"})
    eval_b = repo.record_evaluation(expr_b, _cfg_json(), "w1", _metrics(), 0.2, "passed", [], 2)
    dates_b = np.array(
        ["2022-06-01", "2022-06-02", "2022-06-03", "2022-06-04", "2022-06-05",
         "2022-06-06", "2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10"],
        dtype="datetime64[D]",
    )
    pnl_b = np.arange(10, dtype=np.float64) * 0.001
    repo.save_pool_pnl(eval_b, dates_b, pnl_b)

    pool = repo.load_pool()

    assert set(pool.keys()) == {eval_a, eval_b}

    loaded_dates_a, loaded_pnl_a = pool[eval_a]
    assert loaded_dates_a.shape == (3,)
    assert loaded_pnl_a.shape == (3,)
    assert np.array_equal(loaded_dates_a, dates_a)
    np.testing.assert_allclose(loaded_pnl_a, pnl_a)

    loaded_dates_b, loaded_pnl_b = pool[eval_b]
    assert loaded_dates_b.shape == (10,)
    assert loaded_pnl_b.shape == (10,)
    assert np.array_equal(loaded_dates_b, dates_b)
    np.testing.assert_allclose(loaded_pnl_b, pnl_b)


def test_dead_field_add_and_check(repo):
    assert repo.is_dead_field("bad_field") is False
    repo.add_dead_field("bad_field", reason="brain rejected")
    assert repo.is_dead_field("bad_field") is True


def test_dead_field_add_is_idempotent(repo):
    repo.add_dead_field("bad_field", reason="r1")
    repo.add_dead_field("bad_field", reason="r2")  # ghi đè, không lỗi PK trùng
    assert repo.is_dead_field("bad_field") is True


def test_result_cache_miss_returns_none(repo):
    assert repo.result_cache_get("never_seen_hash", _cfg_json(), "w1") is None


def test_result_cache_hit_after_passed_evaluation(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    repo.record_evaluation(expr_id, _cfg_json(), "w1", _metrics(1.7), 0.1, "passed", [], 9)
    cached = repo.result_cache_get("hash1", _cfg_json(), "w1")
    assert cached is not None
    assert cached.sharpe == pytest.approx(1.7)
    assert cached.per_year_sharpe == {2021: 1.2, 2022: 1.8}


def test_result_cache_no_hit_for_failed_evaluation(repo):
    expr_id = repo.upsert_expression("close", "hash1", depth=1, complexity=1, fields={"close"})
    repo.record_evaluation(expr_id, _cfg_json(), "w1", None, None, "invalid", ["x"], None)
    assert repo.result_cache_get("hash1", _cfg_json(), "w1") is None


def test_result_cache_put_then_get(repo):
    m = _metrics(2.5)
    repo.result_cache_put(
        "hash_new", "ts_mean(close, 5)", depth=2, complexity=3, fields={"close"},
        config_json=_cfg_json(), data_window="w1", metrics=m, seed=7,
    )
    cached = repo.result_cache_get("hash_new", _cfg_json(), "w1")
    assert cached is not None
    assert cached.sharpe == pytest.approx(2.5)


def test_top_n_orders_by_sharpe_desc_passed_only(repo):
    id_a = repo.upsert_expression("a", "ha", depth=1, complexity=1, fields=set())
    id_b = repo.upsert_expression("b", "hb", depth=1, complexity=1, fields=set())
    repo.record_evaluation(id_a, _cfg_json(), "w1", _metrics(1.0), 0.1, "passed", [], 1)
    repo.record_evaluation(id_b, _cfg_json(), "w1", _metrics(3.0), 0.1, "passed", [], 1)
    top = repo.top_n(5)
    assert top[0][0] == "b"
    assert top[0][1] == pytest.approx(3.0)


# --- Task 8: submit_ready_alphas() — khối "SẴN SÀNG NỘP" cuối phiên ---------------------


def _seed_passed_sim(
    repo, alpha_repo, *, expr: str, wq_alpha_id: str, sharpe: float, failed_checks: list[str],
) -> str:
    """Ghi 1 alpha + simulation status='passed' (đúng đường AlphaRepository thật dùng bởi
    lệnh `submit`) — trả alpha_id (khoá để gắn SubmissionModel/BrainSimLinkModel nếu cần)."""
    alpha_id = alpha_repo.save_alpha(expr, source="test")
    result = SimulationResult(
        expression=expr, alpha_id=wq_alpha_id, status="passed", sharpe=sharpe,
        fitness=1.0, turnover=0.3, failed_checks=failed_checks,
    )
    alpha_repo.save_simulation(result, region="USA", universe="TOP3000", alpha_id=alpha_id)
    return alpha_id


def test_submit_ready_alphas_bao_gom_alpha_dat_ca_ba_dieu_kien(repo):
    """status=passed + failed_checks=[] + self_corr<ngưỡng (đã verify qua BrainSimLinkModel,
    tra theo wq_alpha_id) -> có mặt trong danh sách sẵn sàng nộp."""
    alpha_repo = AlphaRepository(repo.session_factory)
    _seed_passed_sim(
        repo, alpha_repo, expr="close - open", wq_alpha_id="rKlkG9O8", sharpe=1.57,
        failed_checks=[],
    )
    repo.record_brain_sim(
        canonical_hash="h1", expr_string="close - open", wq_alpha_id="rKlkG9O8",
        region="USA", universe="TOP3000", sharpe=1.57, fitness=0.73, turnover=0.55,
        self_corr=0.49, status="passed",
    )
    ready = repo.submit_ready_alphas(self_corr_max=0.70)
    assert len(ready) == 1
    assert ready[0].wq_alpha_id == "rKlkG9O8"
    assert ready[0].sharpe == pytest.approx(1.57)
    assert ready[0].self_corr == pytest.approx(0.49)


def test_submit_ready_alphas_rong_khi_khong_co_alpha_dat_chuan(repo):
    """DB trống hoàn toàn -> danh sách rỗng (không lỗi, không giả vờ có gì)."""
    assert repo.submit_ready_alphas(self_corr_max=0.70) == []


def test_submit_ready_alphas_loai_alpha_co_failed_checks(repo):
    """failed_checks khác [] (WQ tự FAIL >=1 check) -> KHÔNG được coi sẵn sàng dù self-corr ổn."""
    alpha_repo = AlphaRepository(repo.session_factory)
    _seed_passed_sim(
        repo, alpha_repo, expr="close - open", wq_alpha_id="BAD1", sharpe=1.6,
        failed_checks=["CONCENTRATED_WEIGHT"],
    )
    repo.record_brain_sim(
        canonical_hash="h2", expr_string="close - open", wq_alpha_id="BAD1", region="USA",
        universe="TOP3000", sharpe=1.6, fitness=0.8, turnover=0.5, self_corr=0.3,
        status="passed",
    )
    assert repo.submit_ready_alphas(self_corr_max=0.70) == []


def test_submit_ready_alphas_loai_alpha_failed_checks_null(repo):
    """failed_checks=NULL nghĩa là CHƯA TỪNG chạy is.checks thật (cột thêm bằng ALTER TABLE
    không DEFAULT — mọi row trước sub-project B là NULL), KHÁC HẲN '[]' = đã kiểm và không
    fail check nào -> KHÔNG được liệt là sẵn sàng (nhất quán cách loại self_corr=None)."""
    from src.storage.models import AlphaModel, SimulationModel

    alpha_repo = AlphaRepository(repo.session_factory)
    alpha_id = alpha_repo.save_alpha("legacy_expr", source="test")
    # Ghi SimulationModel trực tiếp với failed_checks=None — mô phỏng row cũ trước sub-project
    # B (save_simulation ngày nay luôn json.dumps nên không tạo được NULL qua đường đó).
    session = repo.session_factory()
    try:
        session.add(
            SimulationModel(
                id="simnull", alpha_id=alpha_id, wq_alpha_id="LEGACY1", region="USA",
                universe="TOP3000", sharpe=1.6, status="passed", failed_checks=None,
            )
        )
        session.commit()
    finally:
        session.close()
    repo.record_brain_sim(
        canonical_hash="hnull", expr_string="legacy_expr", wq_alpha_id="LEGACY1",
        region="USA", universe="TOP3000", sharpe=1.6, fitness=0.9, turnover=0.4,
        self_corr=0.3, status="passed",
    )
    assert repo.submit_ready_alphas(self_corr_max=0.70) == []


def test_submit_ready_alphas_bo_qua_row_failed_checks_json_hong(repo):
    """failed_checks chứa JSON hỏng (dữ liệu rác) -> bỏ qua row đó, KHÔNG crash, KHÔNG dám
    khẳng định 'sẵn sàng'."""
    from src.storage.models import SimulationModel

    alpha_repo = AlphaRepository(repo.session_factory)
    alpha_id = alpha_repo.save_alpha("corrupt_expr", source="test")
    session = repo.session_factory()
    try:
        session.add(
            SimulationModel(
                id="simbad", alpha_id=alpha_id, wq_alpha_id="CORRUPT1", region="USA",
                universe="TOP3000", sharpe=1.6, status="passed",
                failed_checks="{không phải JSON hợp lệ",
            )
        )
        session.commit()
    finally:
        session.close()
    repo.record_brain_sim(
        canonical_hash="hbad", expr_string="corrupt_expr", wq_alpha_id="CORRUPT1",
        region="USA", universe="TOP3000", sharpe=1.6, fitness=0.9, turnover=0.4,
        self_corr=0.3, status="passed",
    )
    assert repo.submit_ready_alphas(self_corr_max=0.70) == []


def test_submit_ready_alphas_loai_alpha_chua_verify_self_corr(repo):
    """failed_checks=[] + status=passed nhưng self-corr CHƯA từng ghi (None trong DB, vd nộp
    qua đường cũ trước khi có cầu BrainSimLinkModel) -> KHÔNG liệt là sẵn sàng (tránh báo sai)."""
    alpha_repo = AlphaRepository(repo.session_factory)
    _seed_passed_sim(
        repo, alpha_repo, expr="ts_mean(close, 5)", wq_alpha_id="UNVERIFIED", sharpe=1.6,
        failed_checks=[],
    )
    assert repo.submit_ready_alphas(self_corr_max=0.70) == []


def test_submit_ready_alphas_loai_alpha_self_corr_vuot_nguong(repo):
    """self_corr >= ngưỡng -> loại (đúng gate SELF_CORR_MAX, không hardcode ở call site)."""
    alpha_repo = AlphaRepository(repo.session_factory)
    _seed_passed_sim(
        repo, alpha_repo, expr="ts_rank(volume, 10)", wq_alpha_id="HICORR", sharpe=1.6,
        failed_checks=[],
    )
    repo.record_brain_sim(
        canonical_hash="h3", expr_string="ts_rank(volume, 10)", wq_alpha_id="HICORR",
        region="USA", universe="TOP3000", sharpe=1.6, fitness=0.9, turnover=0.4,
        self_corr=0.85, status="passed",
    )
    assert repo.submit_ready_alphas(self_corr_max=0.70) == []


def test_submit_ready_alphas_loai_alpha_da_nop_roi(repo):
    """Alpha đã có SubmissionModel.status='submitted' -> không báo lại là 'sẵn sàng nộp'."""
    alpha_repo = AlphaRepository(repo.session_factory)
    alpha_id = _seed_passed_sim(
        repo, alpha_repo, expr="rank(close)", wq_alpha_id="ALREADYSUB", sharpe=1.5,
        failed_checks=[],
    )
    repo.record_brain_sim(
        canonical_hash="h4", expr_string="rank(close)", wq_alpha_id="ALREADYSUB",
        region="USA", universe="TOP3000", sharpe=1.5, fitness=0.8, turnover=0.3,
        self_corr=0.2, status="passed",
    )
    session = repo.session_factory()
    try:
        session.add(SubmissionModel(id="sub1", alpha_id=alpha_id, status="submitted"))
        session.commit()
    finally:
        session.close()
    assert repo.submit_ready_alphas(self_corr_max=0.70) == []


def test_submit_ready_alphas_sort_sharpe_giam_dan(repo):
    """Nhiều alpha sẵn sàng -> sort Sharpe giảm dần (ứng viên mạnh nhất lên đầu)."""
    alpha_repo = AlphaRepository(repo.session_factory)
    _seed_passed_sim(
        repo, alpha_repo, expr="a_expr", wq_alpha_id="LOW", sharpe=1.2, failed_checks=[],
    )
    repo.record_brain_sim(
        canonical_hash="hlow", expr_string="a_expr", wq_alpha_id="LOW", region="USA",
        universe="TOP3000", sharpe=1.2, fitness=0.7, turnover=0.3, self_corr=0.3,
        status="passed",
    )
    _seed_passed_sim(
        repo, alpha_repo, expr="b_expr", wq_alpha_id="HIGH", sharpe=1.9, failed_checks=[],
    )
    repo.record_brain_sim(
        canonical_hash="hhigh", expr_string="b_expr", wq_alpha_id="HIGH", region="USA",
        universe="TOP3000", sharpe=1.9, fitness=1.1, turnover=0.3, self_corr=0.4,
        status="passed",
    )
    ready = repo.submit_ready_alphas(self_corr_max=0.70)
    assert [r.wq_alpha_id for r in ready] == ["HIGH", "LOW"]
