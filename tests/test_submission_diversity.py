"""Test greedy loại alpha trùng cấu trúc trong tập nộp (GĐ7: T7.1)."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.storage.db import init_db, make_session_factory
from src.storage.models import AlphaModel, SimulationModel
from src.submission.manager import SubmissionManager


def _engine():
    return create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})


class _FakeCorr:
    """Self-correlation luôn chấp nhận (cô lập riêng tác động của lọc đa dạng cấu trúc)."""

    def __init__(self, max_self_corr=0.7):
        self.max_self_corr = max_self_corr

    def max_self_correlation(self, _):
        return 0.1

    def is_acceptable(self, _):
        return True


def _seed(session_factory, rows):
    """rows: [(alpha_id, wq_id, expression, sharpe, fitness, score)]."""
    session = session_factory()
    try:
        for alpha_id, wq_id, expr, sharpe, fitness, score in rows:
            session.add(AlphaModel(id=alpha_id, expression=expr, source="llm"))
            session.add(
                SimulationModel(
                    id="s_" + alpha_id, alpha_id=alpha_id, wq_alpha_id=wq_id,
                    region="USA", universe="TOP3000", sharpe=sharpe, fitness=fitness,
                    score=score, status="passed",
                )
            )
        session.commit()
    finally:
        session.close()


def test_run_daily_loai_alpha_trung_cau_truc_voi_alpha_da_chon():
    """Hai alpha cùng canon (đổi field/window) -> chỉ giữ cái điểm cao hơn."""
    sf = make_session_factory(init_db(_engine()))
    _seed(sf, [
        ("a1", "WQ1", "rank(ts_mean(close, 5))", 2.0, 1.5, 0.9),
        ("a2", "WQ2", "rank(ts_mean(volume, 60))", 1.8, 1.4, 0.8),  # cùng canon với a1 -> loại
        ("a3", "WQ3", "ts_delta(returns, 10)", 1.7, 1.3, 0.7),       # khác cấu trúc -> giữ
    ])
    mgr = SubmissionManager(None, sf, _FakeCorr(), diversify=True, max_struct_similarity=0.9)
    selected = mgr.run_daily(dry_run=True)
    ids = [c.wq_alpha_id for c in selected]
    assert "WQ1" in ids       # điểm cao nhất, luôn giữ
    assert "WQ2" not in ids    # trùng cấu trúc với WQ1 -> loại
    assert "WQ3" in ids        # đa dạng -> giữ


def test_run_daily_khong_diversify_thi_giu_het():
    """Mặc định (diversify tắt) -> không loại theo cấu trúc (tương thích ngược)."""
    sf = make_session_factory(init_db(_engine()))
    _seed(sf, [
        ("a1", "WQ1", "rank(ts_mean(close, 5))", 2.0, 1.5, 0.9),
        ("a2", "WQ2", "rank(ts_mean(volume, 60))", 1.8, 1.4, 0.8),
    ])
    mgr = SubmissionManager(None, sf, _FakeCorr())
    selected = mgr.run_daily(dry_run=True)
    assert len(selected) == 2  # giữ cả hai dù trùng cấu trúc
