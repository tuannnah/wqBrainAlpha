"""Test generate_many: drive (fake) GPEngine.run, score lại bằng _score_one_full, rồi
build_shortlist. Dùng fake GPEngine — KHÔNG chạy GP thực."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from src.backtest.config import PortfolioConfig
from src.lang.parser import parse
from src.pipeline.runner import generate_many


@dataclass
class _FakeFitness:
    sharpe_deflated: float = 1.0


@dataclass
class _FakeIndividual:
    expr: object  # Node thực từ parse()
    fitness: object | None = None
    generation: int = 0


@dataclass
class _FakeRunResult:
    final_population: list = dc_field(default_factory=list)


class _FakeGPEngine:
    """Fake GPEngine: run() trả GPRunResult-like với final_population gồm Individual có AST
    parse từ string cố định — không chạy GP thực (không cần registry/evaluator/...)."""

    def __init__(self, exprs: list[str]) -> None:
        self._exprs = exprs

    def run(self) -> _FakeRunResult:
        return _FakeRunResult(final_population=[
            _FakeIndividual(expr=parse(e), fitness=_FakeFitness()) for e in self._exprs
        ])


def test_generate_many_returns_shortlist_from_fake_gp(small_panel) -> None:  # noqa: ANN001
    engine = _FakeGPEngine(["close", "volume"])
    out = generate_many(
        gp_engine=engine, cfg=PortfolioConfig(delay=1), data=small_panel,
        top_k=5, max_corr=0.99,
    )
    assert len(out) <= 2
    assert all(c.expr in ("close", "volume") for c in out)


def test_generate_many_skips_individuals_with_no_fitness(small_panel) -> None:  # noqa: ANN001
    class _Partial(_FakeGPEngine):
        def run(self) -> _FakeRunResult:
            return _FakeRunResult(final_population=[
                _FakeIndividual(expr=parse("close"), fitness=_FakeFitness()),
                _FakeIndividual(expr=parse("volume"), fitness=None),  # chưa eval -> bỏ qua
            ])

    out = generate_many(
        gp_engine=_Partial([]), cfg=PortfolioConfig(delay=1), data=small_panel,
        top_k=5, max_corr=0.99,
    )
    assert [c.expr for c in out] == ["close"]


def test_generate_many_respects_top_k(small_panel) -> None:  # noqa: ANN001
    engine = _FakeGPEngine(["close", "volume"])
    out = generate_many(
        gp_engine=engine, cfg=PortfolioConfig(delay=1), data=small_panel,
        top_k=1, max_corr=0.99,
    )
    assert len(out) <= 1
