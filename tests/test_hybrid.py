"""Test engine hybrid: seed LLM -> GA tiến hóa -> LLM refine bơm vào vòng."""

from __future__ import annotations

import random

from src.optimization.hybrid import HybridEngine
from src.simulation.pre_filter import PreFilter


class FakeSim:
    def __init__(self):
        self.calls = []

    def simulate(self, expr, settings=None):
        self.calls.append(expr)
        # Metrics dict để score_vector/normalize đọc được; volume -> điểm cao hơn.
        return {"sharpe": 1.0 + expr.count("volume"), "fitness": 1.0,
                "turnover": 0.3, "drawdown": 0.05}


class FakeLLMGen:
    def __init__(self, ideas=None, exprs=None, raise_on=None):
        self._ideas = ideas or ["ý tưởng A"]
        self._exprs = exprs or ["rank(close)"]
        self._raise_on = raise_on or set()

    def generate_ideas(self, n):
        if "ideas" in self._raise_on:
            raise RuntimeError("Error code: 402 - hết token")
        return self._ideas[:n]

    def generate(self, idea, n=5):
        return list(self._exprs)


class FakeRefiner:
    """Trả biến thể cố định; có thể ném 402 để test tắt LLM-in-loop."""

    def __init__(self, out="ts_mean(volume, 5)", raise_402=False):
        self.out = out
        self.raise_402 = raise_402
        self.calls = 0

    def refine(self, candidate, metrics, weak_dimension):
        self.calls += 1
        if self.raise_402:
            raise RuntimeError("Error code: 402 - hết token")
        from src.llm.translator import AlphaCandidate
        return AlphaCandidate(
            hypothesis=candidate.hypothesis, description="cải thiện",
            expression=self.out,
        )


class FakeZoo:
    def __init__(self, originality=1.0):
        self._orig = originality
        self.added = []

    def originality(self, expr):
        return self._orig

    def add(self, expr):
        self.added.append(expr)
        return True


def _engine(**kw):
    pf = PreFilter(known_operators=None, known_fields=None)
    defaults = dict(
        simulator=FakeSim(), prefilter=pf, fields=["close", "volume"],
        llm_generator=FakeLLMGen(), refiner=FakeRefiner(), zoo=FakeZoo(),
        inject_every=2, refine_top=1, population_size=4, generations=4,
        rng=random.Random(0),
    )
    defaults.update(kw)
    return HybridEngine(**defaults)


def test_seed_tu_llm_va_inject_bom_bien_the():
    """Seed lấy từ LLM; refiner được gọi và biến thể vào quần thể + zoo."""
    sim = FakeSim()
    refiner = FakeRefiner(out="ts_mean(volume, 5)")
    zoo = FakeZoo(originality=1.0)
    eng = _engine(simulator=sim, refiner=refiner, zoo=zoo)
    eng.run()
    assert refiner.calls >= 1
    assert "ts_mean(volume, 5)" in zoo.added
    assert any("ts_mean(volume" in c for c in sim.calls)


def test_bien_the_trung_zoo_bi_loai():
    """originality < ngưỡng -> không bơm, không add zoo."""
    refiner = FakeRefiner(out="ts_mean(volume, 5)")
    zoo = FakeZoo(originality=0.1)  # < originality_min=0.4
    eng = _engine(refiner=refiner, zoo=zoo, originality_min=0.4)
    eng.run()
    assert zoo.added == []


def test_llm_402_o_refine_khong_dung_ga():
    """Refiner ném 402 -> tắt LLM-in-loop nhưng GA vẫn chạy hết, trả Node."""
    refiner = FakeRefiner(raise_402=True)
    eng = _engine(refiner=refiner)
    best = eng.run()
    assert best  # GA vẫn trả quần thể, không raise


def test_seed_fallback_template_khi_llm_rong():
    """LLM ném 402 ở seed -> fallback template_generator."""
    class FakeTemplate:
        def generate(self, count, max_attempts=None):
            return ["rank(volume)"]

    eng = _engine(
        llm_generator=FakeLLMGen(raise_on={"ideas"}),
        template_generator=FakeTemplate(),
    )
    best = eng.run()
    assert best


def test_max_simulations_dung_xac_dinh():
    """generations=None + max_simulations nhỏ -> kết thúc xác định."""
    eng = _engine(generations=None, max_simulations=5)
    best = eng.run()
    assert best
