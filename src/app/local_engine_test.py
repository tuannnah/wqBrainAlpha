"""Test cục bộ 1 lượt cho engine sinh alpha (mục 4 menu "Test engine").

GP sinh nhanh 1 ứng viên (đã chấm local) -> LLM THẬT refine -> chấm lại local. Không đụng
WQ Brain API/quota — dùng để tự bắt lỗi wiring (DB/GP/registry/LLM) TRƯỚC khi chạy vòng kín
thật (mục 5, tốn sim quota). Composition-root layer (như `closed_loop_adapters.py`): được
phép import src.gp/src.llm/src.pipeline.

KHÔNG dùng `ClosedLoop`/`repo.record_brain_sim`: bảng đó dành cho kết quả SIM thật từ Brain,
ghi dữ liệu giả (không wq_alpha_id thật) vào đây sẽ làm sai lệch calibration/ρ về sau.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.app.closed_loop_adapters import GPIdeaSource
from src.llm.hypothesis import Hypothesis
from src.llm.refiner import AlphaRefiner
from src.llm.translator import AlphaCandidate, AlphaTranslator
from src.pipeline.runner import score_one


@dataclass
class LocalEngineTestResult:
    """Kết quả 1 lượt test engine cục bộ — không raise, mọi lỗi gói vào `error`."""

    idea_expr: str | None = None
    refined_expr: str | None = None
    sharpe_before: float | None = None
    fitness_before: float | None = None
    sharpe_after: float | None = None
    fitness_after: float | None = None
    passed: bool | None = None
    hard_failures: tuple[str, ...] = field(default_factory=tuple)
    llm_ok: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_local_engine_test(
    *, data, repo, config, registry, deepseek, field_repo, operator_repo, prefilter,
    pop_size: int = 10, n_generations: int = 1,
) -> LocalEngineTestResult:
    """1 lượt: GP sinh nhanh 1 ý tưởng -> LLM refine thật -> re-score local. Bắt mọi lỗi
    nội bộ (wiring/parse/LLM) và trả về trong `result.error` thay vì raise, để nơi gọi
    (menu) in gọn kết quả mà không crash tiến trình."""
    result = LocalEngineTestResult()
    try:
        idea_source = GPIdeaSource(
            data, repo, config, registry, pop_size=pop_size, n_generations=n_generations, top_k=1,
        )
        batch = idea_source.next_batch()
        if not batch:
            result.error = "GP không sinh được ứng viên nào (quần thể rỗng sau lọc/decorrelate)."
            return result

        candidate = batch[0]
        result.idea_expr = candidate.expr
        result.sharpe_before = candidate.metrics.sharpe
        result.fitness_before = candidate.metrics.fitness

        translator = AlphaTranslator(deepseek, field_repo, operator_repo, prefilter)
        refiner = AlphaRefiner(deepseek, translator)
        seed_candidate = AlphaCandidate(
            hypothesis=Hypothesis(), description=candidate.expr, expression=candidate.expr,
        )
        metrics = {
            "sharpe": candidate.metrics.sharpe,
            "fitness": candidate.metrics.fitness,
            "turnover": candidate.metrics.turnover,
        }
        refined = refiner.refine(seed_candidate, metrics, weak_dimension="sharpe")
        if refined is None:
            result.error = "LLM refine không trả về biểu thức mới (rỗng hoặc trùng đã thử)."
            return result
        result.llm_ok = True
        result.refined_expr = refined.expression

        new_metrics, verdict = score_one(
            refined.expression, config, data, pool=repo.load_pool() or None,
        )
        result.sharpe_after = new_metrics.sharpe
        result.fitness_after = new_metrics.fitness
        result.passed = verdict.passed
        result.hard_failures = tuple(verdict.hard_failures)
    except Exception as exc:  # noqa: BLE001 - test-engine: gói MỌI lỗi để menu in gọn, không crash
        result.error = f"{type(exc).__name__}: {exc}"
    return result
