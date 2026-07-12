"""TDD cho repository.brain_proven_signals — nguồn tín hiệu con Brain-proven cho combiner
(Task 2 Fix 1, thay `good_signals_for_combine`: fitness LOCAL có ρ=0.308 với Brain — xếp
hạng theo nó chọn toàn GP junk (đo được trong `logs/diag_combiner_20260712.md`); sharpe
BRAIN thật (`BrainSimLinkModel.sharpe`) mới là thước đo đúng để chọn thành phần combo.
KHÔNG lọc theo `status`: alpha 'failed' vì LOW_SHARPE (vd 1.04 < ngưỡng nộp 1.58) vẫn là
component quý — Grinold-Kahn √N có thể đẩy nó lên ngưỡng nộp khi ghép."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


def _repo() -> MiniBrainRepository:
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    return MiniBrainRepository(sessionmaker(bind=engine, future=True, expire_on_commit=False))


def _seed(repo: MiniBrainRepository, hsh: str, expr: str, sharpe: float, status: str) -> None:
    repo.record_brain_sim(
        canonical_hash=hsh, expr_string=expr, wq_alpha_id=None, region="USA",
        universe="TOP3000", sharpe=sharpe, fitness=1.0, turnover=0.1, self_corr=0.4,
        status=status,
    )


def test_loc_theo_min_sharpe_khong_loc_status():
    repo = _repo()
    # sharpe 1.04 nhưng status='failed' (LOW_SHARPE, dưới ngưỡng nộp 1.58) -> vẫn PHẢI vào
    # danh sách vì brief cấm lọc theo status.
    _seed(repo, "h1", "rank(ts_delta(close, 5))", 1.04, "failed")
    _seed(repo, "h2", "rank(ts_delta(volume, 5))", 0.3, "failed")  # dưới min_sharpe -> loại

    out = repo.brain_proven_signals(0.8)

    assert len(out) == 1
    assert out[0][0] == "rank(ts_delta(close, 5))"
    assert out[0][1] == 1.04


def test_sort_sharpe_giam_dan():
    repo = _repo()
    _seed(repo, "h1", "a", 0.9, "passed")
    _seed(repo, "h2", "b", 1.5, "passed")
    _seed(repo, "h3", "c", 1.0, "passed")

    out = repo.brain_proven_signals(0.8)

    assert [e for e, _ in out] == ["b", "c", "a"]


def test_distinct_expr_giu_sharpe_cao_nhat():
    repo = _repo()
    # Cùng expr_string, hai canonical_hash khác nhau (vd sim lại sau khi đổi setting) ->
    # DISTINCT expr_string chỉ giữ 1 dòng, ưu tiên sharpe CAO NHẤT đo được.
    _seed(repo, "h1", "a", 0.9, "passed")
    _seed(repo, "h2", "a", 1.5, "passed")

    out = repo.brain_proven_signals(0.8)

    assert out == [("a", 1.5)]


def test_db_rong_tra_list_rong():
    assert _repo().brain_proven_signals(0.8) == []
