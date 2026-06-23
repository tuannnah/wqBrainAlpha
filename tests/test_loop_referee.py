"""RefinementLoop với trọng tài LLM: sau mỗi sim, referee quyết refine_formula |
tune_config | abandon. Giữ trần cứng (patience/max_sims). Tương thích ngược khi
không truyền referee."""

from __future__ import annotations

from sqlalchemy import create_engine

from src.llm.hypothesis import Hypothesis
from src.llm.loop import RefinementLoop
from src.llm.referee import ABANDON, REFINE_FORMULA, TUNE_CONFIG, Verdict
from src.llm.translator import AlphaCandidate
from src.simulation.config import SimConfig
from src.simulation.pre_filter import PreFilter
from src.simulation.simulator import SimulationResult
from src.storage.db import init_db, make_session_factory
from src.storage.repository import AlphaRepository
from tests.fakes import FakeSimulator


def _repo():
    engine = init_db(create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False}))
    return AlphaRepository(make_session_factory(engine))


def _prefilter():
    return PreFilter(known_operators={"rank", "ts_mean", "ts_delta"}, known_fields={"close", "volume"})


class _FakeHyp:
    def generate(self, direction, palette=None):
        return Hypothesis("o", "b", "r", "s")


class _FakeTranslator:
    def __init__(self, expr):
        self.expr = expr

    def field_palette(self, text):
        return []

    def translate(self, hyp):
        return AlphaCandidate(hyp, "mô tả gốc", self.expr)


class _FakeRefiner:
    def __init__(self, exprs):
        self.exprs = list(exprs)
        self.i = 0
        self.calls = 0

    def refine(self, candidate, metrics, weak_dimension):
        self.calls += 1
        if self.i >= len(self.exprs):
            return None
        e = self.exprs[self.i]
        self.i += 1
        return AlphaCandidate(candidate.hypothesis, "mô tả cải tiến", e)


class _FakeReferee:
    """Trả lần lượt các action; hết thì mặc định refine_formula."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0
        self.calls = 0

    def judge(self, direction, history, metrics):
        self.calls += 1
        a = self.actions[self.i] if self.i < len(self.actions) else REFINE_FORMULA
        self.i += 1
        return Verdict(a, "r")


class _FakeTuner:
    def __init__(self, new_config):
        self.new_config = new_config
        self.calls = 0

    def tune(self, config, metrics, reason):
        self.calls += 1
        return self.new_config


def _result(expr, sharpe, status="passed"):
    return SimulationResult(
        expression=expr, alpha_id="wq-" + expr, status=status,
        sharpe=sharpe, fitness=1.2, turnover=0.3, drawdown=0.1, raw={},
    )


def _loop(translator, refiner, sim, repo, **kw):
    return RefinementLoop(
        hypothesis_gen=_FakeHyp(), translator=translator, refiner=refiner,
        simulator=sim, prefilter=_prefilter(), repo=repo,
        region="USA", universe="TOP3000", **kw,
    )


# --------------------------------------------------------------------- abandon
def test_referee_abandon_dung_som_truoc_khi_refine():
    """Referee abandon ngay sau seed -> không refine, stop_reason='abandon'."""
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    refiner = _FakeRefiner(["rank(ts_mean(close, 5))"])
    loop = _loop(
        _FakeTranslator("rank(close)"), refiner, sim, _repo(),
        max_simulations=10, no_improve_patience=5, referee=_FakeReferee([ABANDON]),
    )
    res = loop.run("X")
    assert refiner.calls == 0          # bỏ hướng trước khi refine
    assert res.stop_reason == "abandon"
    assert res.best_candidate.expression == "rank(close)"


# ------------------------------------------------------------------ tune_config
def test_referee_tune_config_sim_lai_cung_expr_voi_config_moi():
    """tune_config -> giữ nguyên biểu thức, sim lại với config mới."""
    class _SettingsSim:
        def __init__(self):
            self.calls = []

        def simulate(self, expr, settings=None):
            self.calls.append((expr, settings))
            # config mới cho sharpe cao hơn -> được nhận
            sharpe = 2.0 if settings.get("decay") == 12 else 1.5
            return _result(expr, sharpe)

    base = SimConfig(region="USA", universe="TOP3000", delay=1,
                     decay=4, truncation=0.01, neutralization="MARKET")
    tuned = base.with_overrides(decay=12)
    sim = _SettingsSim()
    tuner = _FakeTuner(tuned)
    loop = _loop(
        _FakeTranslator("rank(close)"), _FakeRefiner([]), sim, _repo(),
        max_simulations=10, no_improve_patience=5, sim_config=base,
        referee=_FakeReferee([TUNE_CONFIG, ABANDON]), config_tuner=tuner,
    )
    res = loop.run("X")
    assert tuner.calls == 1
    # 2 sim CÙNG biểu thức, lần 2 dùng config mới (decay=12)
    assert [c[0] for c in sim.calls] == ["rank(close)", "rank(close)"]
    assert sim.calls[0][1]["decay"] == 4
    assert sim.calls[1][1]["decay"] == 12
    assert res.best_vector.total >= 0  # config mới tốt hơn -> được giữ


def test_referee_tune_config_khong_doi_thi_khong_treo():
    """config_tuner trả config y hệt -> không sim lại, nhưng vẫn tiến tới trần cứng
    (không kẹt vô hạn)."""
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    base = SimConfig(decay=4, truncation=0.01, neutralization="MARKET")
    loop = _loop(
        _FakeTranslator("rank(close)"), _FakeRefiner([]), sim, _repo(),
        max_simulations=10, no_improve_patience=2, sim_config=base,
        referee=_FakeReferee([TUNE_CONFIG, TUNE_CONFIG, TUNE_CONFIG]),
        config_tuner=_FakeTuner(base),  # không đổi
    )
    res = loop.run("X")
    assert len(sim.calls) == 1          # chỉ seed sim, không sim lại
    assert res.stop_reason == "patience"


# --------------------------------------------------------------- refine_formula
def test_referee_refine_formula_van_refine_binh_thuong():
    scores = {"rank(close)": 1.0, "rank(ts_mean(close, 5))": 1.9}
    sim = FakeSimulator(results=lambda e: _result(e, scores[e]))
    refiner = _FakeRefiner(["rank(ts_mean(close, 5))"])
    loop = _loop(
        _FakeTranslator("rank(close)"), refiner, sim, _repo(),
        max_simulations=10, no_improve_patience=2,
        referee=_FakeReferee([REFINE_FORMULA]),
    )
    res = loop.run("X")
    assert refiner.calls >= 1
    assert res.best_candidate.expression == "rank(ts_mean(close, 5))"


# ----------------------------------------------------------- tương thích ngược
def test_khong_referee_giu_hanh_vi_cu_va_co_stop_reason():
    sim = FakeSimulator(results=lambda e: _result(e, 1.5))
    loop = _loop(
        _FakeTranslator("rank(close)"), _FakeRefiner([]), sim, _repo(),
        max_simulations=1, no_improve_patience=3,
    )
    res = loop.run("X")
    assert res.stop_reason == "budget"  # hết trần sim
