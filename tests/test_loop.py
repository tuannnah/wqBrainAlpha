"""Test RefinementLoop: vòng greedy, trần sim, cache, zoo, failure (GĐ2: T2.14)."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.llm.hypothesis import Hypothesis
from src.llm.loop import RefinementLoop
from src.llm.translator import AlphaCandidate
from src.simulation.pre_filter import PreFilter
from src.simulation.simulator import SimulationResult
from src.storage.db import init_db, make_session_factory
from src.storage.repository import AlphaRepository
from tests.fakes import FakeSimulator


def _repo():
    engine = init_db(create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False}))
    return AlphaRepository(make_session_factory(engine))


def _prefilter():
    return PreFilter(
        known_operators={"rank", "ts_mean", "ts_delta", "ts_decay_linear"},
        known_fields={"close", "volume"},
    )


class _FakeHyp:
    def generate(self, direction):
        return Hypothesis("o", "b", "r", "s")


class _FakeTranslator:
    def __init__(self, expr):
        self.expr = expr

    def translate(self, hyp):
        return AlphaCandidate(hyp, "mô tả gốc", self.expr)


class _FakeTranslatorNone:
    def translate(self, hyp):
        return None


class _FakeRefiner:
    """Trả lần lượt các biểu thức cải tiến (None khi hết)."""

    def __init__(self, exprs):
        self.exprs = list(exprs)
        self.i = 0

    def refine(self, candidate, metrics, weak_dimension):
        if self.i >= len(self.exprs):
            return None
        e = self.exprs[self.i]
        self.i += 1
        return AlphaCandidate(candidate.hypothesis, "mô tả cải tiến", e)


def _result(expr, sharpe, status="passed"):
    return SimulationResult(
        expression=expr, alpha_id="wq-" + expr, status=status,
        sharpe=sharpe, fitness=1.2, turnover=0.3, drawdown=0.1, raw={},
    )


def _loop(translator, refiner, sim, repo, **kw):
    return RefinementLoop(
        hypothesis_gen=_FakeHyp(),
        translator=translator,
        refiner=refiner,
        simulator=sim,
        prefilter=_prefilter(),
        repo=repo,
        region="USA",
        universe="TOP3000",
        **kw,
    )


def test_loop_ton_trong_tran_sim():
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    repo = _repo()
    refiner = _FakeRefiner([f"rank(ts_mean(close, {d}))" for d in range(5, 60, 5)])
    loop = _loop(_FakeTranslator("rank(close)"), refiner, sim, repo, max_simulations=3)
    res = loop.run("hướng X")
    assert res.sims_used == 3
    assert len(sim.calls) == 3


def test_loop_cai_thien_best_qua_cac_vong():
    # sharpe tăng dần theo biểu thức -> best total cải thiện.
    scores = {
        "rank(close)": 1.0,
        "rank(ts_mean(close, 5))": 1.5,
        "rank(ts_mean(close, 10))": 1.9,
    }
    sim = FakeSimulator(results=lambda e: _result(e, scores[e]))
    repo = _repo()
    refiner = _FakeRefiner(["rank(ts_mean(close, 5))", "rank(ts_mean(close, 10))"])
    loop = _loop(_FakeTranslator("rank(close)"), refiner, sim, repo, max_simulations=10, no_improve_patience=3)
    res = loop.run("X")
    assert res.best_candidate.expression == "rank(ts_mean(close, 10))"
    totals = [h["total"] for h in res.history]
    assert totals == sorted(totals)  # không giảm


def test_loop_zoo_va_failure():
    # seed sharpe thấp (fail hard filter) ; refine sharpe cao (vào zoo).
    scores = {"rank(close)": 1.0, "rank(ts_mean(close, 5))": 1.8}
    sim = FakeSimulator(results=lambda e: _result(e, scores[e]))
    repo = _repo()
    refiner = _FakeRefiner(["rank(ts_mean(close, 5))"])
    loop = _loop(_FakeTranslator("rank(close)"), refiner, sim, repo, max_simulations=10, no_improve_patience=2)
    res = loop.run("X")
    assert res.zoo_added >= 1
    assert repo.zoo(10)  # có alpha pass trong DB
    cats = {f.category for f in repo.recent_failures(10)}
    assert "low_score" in cats  # seed bị ghi failure


def test_loop_cache_khong_sim_trung():
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    repo = _repo()
    # refiner trả lại đúng biểu thức seed -> lần 2 phải dùng cache.
    refiner = _FakeRefiner(["rank(close)", "rank(close)"])
    loop = _loop(_FakeTranslator("rank(close)"), refiner, sim, repo, max_simulations=10, no_improve_patience=1)
    loop.run("X")
    assert len(sim.calls) == 1  # chỉ sim 1 lần cho biểu thức trùng


def test_loop_callback_tien_do():
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    repo = _repo()
    refiner = _FakeRefiner(["rank(ts_mean(close, 5))"])
    loop = _loop(_FakeTranslator("rank(close)"), refiner, sim, repo, max_simulations=2)
    events = []
    loop.run("X", on_progress=events.append)
    assert events  # có phát sự kiện tiến độ
    assert all(hasattr(e, "sims_used") for e in events)


# --------------------------------------------------- T3.5 originality pre-filter
def test_loop_loai_alpha_trung_cau_truc_zoo_truoc_sim():
    """Seed trùng cấu trúc zoo (originality dưới ngưỡng) -> loại, KHÔNG sim."""
    from src.decorrelation.zoo import ReferenceZoo

    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    repo = _repo()
    zoo = ReferenceZoo(["rank(ts_mean(close, 5))"])
    # seed đổi field/window vẫn cùng canon -> originality ~ 0 -> bị loại.
    refiner = _FakeRefiner([])
    loop = _loop(
        _FakeTranslator("rank(ts_mean(volume, 60))"), refiner, sim, repo,
        max_simulations=10, zoo=zoo, min_originality=0.2,
    )
    res = loop.run("X")
    assert len(sim.calls) == 0  # không tốn sim nào cho alpha gần-trùng
    assert res.best_candidate is None
    cats = {f.category for f in repo.recent_failures(10)}
    assert "duplicate" in cats


def test_loop_giu_alpha_doc_dao_qua_prefilter():
    """Alpha độc đáo (operator khác hẳn zoo) vẫn được sim bình thường."""
    from src.decorrelation.zoo import ReferenceZoo

    sim = FakeSimulator(results=lambda e: _result(e, 1.8))
    repo = _repo()
    zoo = ReferenceZoo(["rank(ts_mean(close, 5))"])
    refiner = _FakeRefiner([])
    # ts_delta khác hẳn rank/ts_mean của zoo -> độc đáo; vẫn nằm trong whitelist prefilter.
    loop = _loop(
        _FakeTranslator("ts_delta(volume, 5)"), refiner, sim, repo,
        max_simulations=10, zoo=zoo, min_originality=0.2,
    )
    res = loop.run("X")
    assert len(sim.calls) == 1
    assert res.best_candidate is not None


def test_loop_khong_zoo_thi_bo_qua_prefilter_originality():
    """Không truyền zoo -> không lọc originality (tương thích ngược)."""
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    repo = _repo()
    refiner = _FakeRefiner([])
    loop = _loop(_FakeTranslator("rank(close)"), refiner, sim, repo, max_simulations=10)
    res = loop.run("X")
    assert len(sim.calls) == 1
    assert res.best_candidate is not None


# ------------------------------------------------ T4.2 alignment pre-filter
class _FakeAligner:
    """Trả điểm nhất quán cố định cho mọi candidate."""

    def __init__(self, value):
        from src.llm.alignment import AlignmentScore

        self._score = AlignmentScore(value, "fake")

    def score(self, candidate):
        return self._score


def test_loop_loai_alpha_lech_gia_thuyet_truoc_sim():
    """Điểm nhất quán dưới ngưỡng -> loại, KHÔNG sim, ghi hypothesis_mismatch."""
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    repo = _repo()
    refiner = _FakeRefiner([])
    loop = _loop(
        _FakeTranslator("rank(close)"), refiner, sim, repo,
        max_simulations=10, aligner=_FakeAligner(0.2), min_alignment=0.5,
    )
    res = loop.run("X")
    assert len(sim.calls) == 0
    assert res.best_candidate is None
    cats = {f.category for f in repo.recent_failures(10)}
    assert "hypothesis_mismatch" in cats


def test_loop_giu_alpha_khop_gia_thuyet():
    """Điểm nhất quán đạt ngưỡng -> vẫn được sim bình thường."""
    sim = FakeSimulator(results=lambda e: _result(e, 1.8))
    repo = _repo()
    refiner = _FakeRefiner([])
    loop = _loop(
        _FakeTranslator("rank(close)"), refiner, sim, repo,
        max_simulations=10, aligner=_FakeAligner(0.9), min_alignment=0.5,
    )
    res = loop.run("X")
    assert len(sim.calls) == 1
    assert res.best_candidate is not None


def test_loop_khong_aligner_thi_bo_qua_loc_alignment():
    """Không truyền aligner -> không lọc nhất quán (tương thích ngược)."""
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    repo = _repo()
    refiner = _FakeRefiner([])
    loop = _loop(_FakeTranslator("rank(close)"), refiner, sim, repo, max_simulations=10)
    res = loop.run("X")
    assert len(sim.calls) == 1
    assert res.best_candidate is not None


# ----------------------------------------- T4.4 điểm điều chuẩn chọn best
# seed đơn giản, refine phức tạp hơn nhưng raw total CAO hơn một chút.
_REG_SEED = "rank(close)"
_REG_COMPLEX = "rank(ts_mean(ts_delta(close, 5), 10))"
_REG_SCORES = {_REG_SEED: 1.5, _REG_COMPLEX: 1.6}


def test_loop_tat_regularize_thi_alpha_phuc_tap_thang():
    """Mặc định (tắt điều chuẩn): raw total quyết định -> alpha phức tạp thắng."""
    sim = FakeSimulator(results=lambda e: _result(e, _REG_SCORES[e]))
    repo = _repo()
    refiner = _FakeRefiner([_REG_COMPLEX])
    loop = _loop(
        _FakeTranslator(_REG_SEED), refiner, sim, repo,
        max_simulations=10, no_improve_patience=1,
    )
    res = loop.run("X")
    assert res.best_candidate.expression == _REG_COMPLEX


def test_loop_bat_regularize_thi_phat_phuc_tap_giu_alpha_don_gian():
    """Bật điều chuẩn: phạt độ phức tạp kéo alpha rối xuống -> seed đơn giản được giữ."""
    sim = FakeSimulator(results=lambda e: _result(e, _REG_SCORES[e]))
    repo = _repo()
    refiner = _FakeRefiner([_REG_COMPLEX])
    loop = _loop(
        _FakeTranslator(_REG_SEED), refiner, sim, repo,
        max_simulations=10, no_improve_patience=1,
        regularize=True, penalty_lambda=1.0,
    )
    res = loop.run("X")
    assert res.best_candidate.expression == _REG_SEED


# --------------------------------------------------------- T6.1 MCTS
def test_run_mcts_tim_duoc_alpha_tot_hon_seed():
    """MCTS khám phá nhiều nhánh, trả về alpha điểm cao nhất."""
    scores = {
        "rank(close)": 1.0,
        "rank(ts_mean(close, 5))": 1.6,
        "rank(ts_mean(close, 10))": 2.0,
    }
    sim = FakeSimulator(results=lambda e: _result(e, scores.get(e, 1.0)))
    repo = _repo()
    refiner = _FakeRefiner(["rank(ts_mean(close, 5))", "rank(ts_mean(close, 10))"])
    loop = _loop(_FakeTranslator("rank(close)"), refiner, sim, repo, max_simulations=10)
    res = loop.run_mcts("X", iterations=2)
    assert res.best_candidate.expression == "rank(ts_mean(close, 10))"


def test_run_mcts_ton_trong_tran_sim():
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    repo = _repo()
    refiner = _FakeRefiner([f"rank(ts_mean(close, {d}))" for d in range(5, 80, 5)])
    loop = _loop(_FakeTranslator("rank(close)"), refiner, sim, repo, max_simulations=3)
    res = loop.run_mcts("X", iterations=50)
    assert res.sims_used <= 3


def test_run_mcts_seed_loi_tra_ket_qua_rong():
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    repo = _repo()
    # translator trả None -> không dịch được giả thuyết.
    loop = _loop(_FakeTranslatorNone(), _FakeRefiner([]), sim, repo, max_simulations=10)
    res = loop.run_mcts("X", iterations=5)
    assert res.best_candidate is None
