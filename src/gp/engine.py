"""GPEngine — vòng lặp tiến hóa MiniBrain ghép 6 building block Phase 7 với Phase 2/3/4/6
(Evaluator/Backtester/MetricsCalculator/GateEvaluator/PoolCorrelation) + persist mọi
kết quả qua MiniBrainRepository (Phase 5).

Stage separation (B5): tìm kiếm BARE SIGNAL CORE; neut/decay/trunc/scale/delay được áp
ngoài qua PortfolioConfig truyền vào constructor, KHÔNG bọc vào ``Individual.expr``.

Determinism (R8): mọi randomness đi qua ``np.random.default_rng(seed)`` inject; cùng seed +
cùng config phải cho cùng quần thể cuối. Không dùng ``np.random`` toàn cục.

Dependency rule (B1): module này được phép import lang/engine/backtest/storage/operators_local
nhưng KHÔNG import ``src.llm`` (seed LLM lấy qua ``all_seed_cores`` với dependency truyền vào).
A2 (2026-07-18): thêm ``src.reporting.diagnostics.classify_family`` (họ nhân tố, chuỗi->chuỗi
thuần) để lọc họ-đã-đóng TRƯỚC backtest — ``diagnostics`` chỉ import ngược
``src.generation.frontier_seeds`` -> ``src.lang.registry``, không chạm ``src.gp``/``src.llm``
nên không tạo vòng import.

C1 (2026-07-18): ``_evaluate_population`` song song hoá PHẦN THUẦN (eval AST → backtest →
metrics) qua ``ProcessPoolExecutor`` khi ``n_jobs > 1`` VÀ có ``executor`` truyền vào
(``GPIdeaSource`` dựng pool MỘT LẦN, sống xuyên nhiều ``GPEngine``/batch) — xem
``_prefetch_parallel`` + ``src/gp/parallel_eval.py`` (worker module-level, Windows spawn).
Phần TRẠNG THÁI (gate/pool_corr/fitness/``_persist`` SQLite) LUÔN chạy tuần tự trong process
chính THEO ĐÚNG THỨ TỰ INDEX GỐC của quần thể — pool self-corr lớn dần theo thứ tự persist
nên xử lý lệch thứ tự sẽ đổi kết quả gate (song song ≡ tuần tự là bất biến bắt buộc, C2 sẽ
test parity n_jobs=1 với n_jobs=2). ``n_jobs=1`` (mặc định): đường cũ nguyên vẹn, không đụng.

C1 fix review cuối (2026-07-18): entry ``eval_cache["ok"]`` chỉ giữ ``(daily_pnl, metrics)``
— KHÔNG giữ ``BacktestResult``/``weights`` nữa (ma trận ``weights`` (T,N) ~10-15MB/cá thể trên
panel thật; giữ ``pop_size`` × nhiều thế hệ trong dict CHIA SẺ xuyên phiên gây OOM). ``bt`` chỉ
sống trong phạm vi cục bộ lúc backtest tươi; mọi nơi cần dữ liệu backtest sau đó (gate/
pool_rho/persist/save_pool_pnl) dùng ``daily_pnl`` (ndarray 1 chiều, vài chục KB) +
``metrics`` (đã tính, KHÔNG recompute ở ``_persist``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from config.thresholds import COMBINER_MAX_COMPONENT_DEPTH, GP_BEST_COMBINABLE_TOP_K
from src.backtest.backtester import Backtester
from src.backtest.config import PortfolioConfig
from src.backtest.gates import GateEvaluator
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.backtest.pool_corr import PoolCorrelation
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.engine.subexpr_cache import SubexprCache
from src.gp.fitness_vec import FitnessVector, from_metrics
from src.gp.individual import Individual
from src.gp.init import init_population
from src.gp.parallel_eval import eval_thuan
from src.gp.seeds import all_seed_cores
from src.gp.selection import nsga2_select
from src.gp.variation import (
    crossover,
    dedup_population,
    hoist_mutation,
    point_mutation,
    subtree_mutation,
)
from src.lang.meaningfulness import check_meaningful
from src.lang.registry import OperatorRegistry
from src.lang.visitors import (
    CanonicalHasher,
    ComplexityVisitor,
    DepthVisitor,
    FieldCollector,
    Serializer,
)
# A2: classify_family suy họ nhân tố từ chuỗi expr (heuristic substring, không parse ngược) —
# src.reporting.diagnostics chỉ import src.generation.frontier_seeds -> src.lang.registry,
# KHÔNG import ngược src.gp nên import module-level ở đây không tạo vòng import (B1 vẫn giữ:
# engine.py không import src.llm).
from src.reporting.diagnostics import classify_family
from src.storage.repository import MiniBrainRepository

if TYPE_CHECKING:
    # C1: chỉ dùng cho type hint tham số ``executor`` — ``from __future__ import
    # annotations`` (trên) khiến annotation không evaluate lúc runtime, nên import này
    # không bắt buộc phải chạy thật (tránh engine.py phải kéo concurrent.futures ở mọi
    # đường n_jobs=1, đúng yêu cầu "đường cũ nguyên vẹn, không import concurrent.futures").
    from concurrent.futures import ProcessPoolExecutor


@dataclass(frozen=True, slots=True)
class GPRunResult:
    """Kết quả một lần chạy GPEngine: quần thể cuối + cá thể tốt nhất + thống kê + seed.

    ``best_by_sharpe`` là cá thể có ``sharpe_deflated`` cao nhất trong quần thể cuối (chỉ
    xét cá thể đã đánh giá thành công); ``None`` nếu không có cá thể nào hợp lệ. Đây thường
    là cây SÂU NHẤT/overfit nhất (bối cảnh task-2-brief.md) — giữ nguyên field này CHỈ cho
    báo cáo/chẩn đoán, KHÔNG dùng để feed combiner/DB-good-signals.

    ``best_combinable`` (T2.1, WS2 task-2-brief.md): trong top-``GP_BEST_COMBINABLE_TOP_K``
    cá thể theo ``sharpe_deflated``, cá thể sharpe cao NHẤT có depth <=
    ``COMBINER_MAX_COMPONENT_DEPTH`` (combinable-aware) — ``None`` nếu top-K không có cá
    thể nào đủ nông. Đây là bản nên dùng khi cần MỘT cá thể đại diện để feed combiner/kho
    alpha tốt (khác ``generate_many``/``build_shortlist`` — pipeline hiện tại vốn đã duyệt
    TOÀN BỘ ``final_population`` chứ không chỉ một "best" đơn lẻ, xem report T2.1 để rõ lý
    do field này chỉ bổ sung, không thay thế đường tiêu thụ hiện có)."""

    generations_run: int
    final_population: list[Individual]
    best_by_sharpe: Individual | None
    n_evaluated: int
    n_passed: int
    seed: int
    best_combinable: Individual | None = None


def _select_best_combinable(
    evaluated: list[Individual],
    top_k: int = GP_BEST_COMBINABLE_TOP_K,
    max_component_depth: int = COMBINER_MAX_COMPONENT_DEPTH,
) -> Individual | None:
    """(T2.1) Trong top-``top_k`` cá thể ĐÃ EVAL xếp theo ``sharpe_deflated`` giảm dần, trả
    cá thể sharpe cao NHẤT có ``depth() <= max_component_depth`` (combinable) — KHÔNG đơn
    thuần cá thể sharpe cao nhất tuyệt đối (đó là ``best_by_sharpe`` ở ``GPRunResult``,
    thường là cây sâu nhất/overfit nhất khiến combiner chết trần, xem bối cảnh
    task-2-brief.md). ``None`` nếu top-K không có cá thể nào đủ nông — KHÔNG hạ tiêu chuẩn
    quét ra ngoài top-K hay lấy đại cá thể quá sâu (phá bất biến combinable của combiner)."""
    ranked = sorted(
        evaluated, key=lambda i: i.fitness.sharpe_deflated, reverse=True,  # type: ignore[union-attr]
    )
    for ind in ranked[:top_k]:
        if ind.depth() <= max_component_depth:
            return ind
    return None


class GPEngine:
    """Vòng lặp GP: init seed → đánh giá → biến đổi → chọn lọc → đánh giá → ... → kết quả.

    Mọi tham số tinh chỉnh nhận qua keyword-only để gọi rõ ràng; ``data``/``repo``/``config``/
    ``registry`` là phụ thuộc bắt buộc (positional). ``max_depth`` phải <= MAX_DEPTH cấu hình
    trong ``config/thresholds.py`` để không sinh cây vượt trần gate.
    """

    def __init__(
        self,
        data: MarketData,
        repo: MiniBrainRepository,
        config: PortfolioConfig,
        registry: OperatorRegistry,
        *,
        pop_size: int = 50,
        n_generations: int = 5,
        max_depth: int = 7,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.3,
        seed: int = 42,
        seed_offset: int = 0,
        data_window: str = "default",
        with_llm_seeds: bool = False,
        n_jobs: int = 1,
        executor: "ProcessPoolExecutor | None" = None,
        saturated_families: "frozenset[str] | set[str]" = frozenset(),
        eval_cache: "dict[str, tuple] | None" = None,
        fields_override: "tuple[str, ...] | None" = None,
        field_groups: "tuple[tuple[str, ...], ...] | None" = None,
    ) -> None:
        self.data = data
        self.repo = repo
        self.config = config
        self.registry = registry
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.seed_offset = seed_offset
        self.data_window = data_window
        self.with_llm_seeds = with_llm_seeds
        self.n_jobs = n_jobs
        # C1: pool process CHIA SẺ do GPIdeaSource dựng MỘT LẦN (không tạo/hủy pool mỗi
        # GPEngine — tốn kém). None = không song song hoá (n_jobs=1 mặc định, hoặc caller
        # chưa nối executor) -> _evaluate_population đi đường tuần tự NGUYÊN VẸN như trước
        # C1, không import concurrent.futures ở engine.py. LƯU Ý: song song CHỈ có hiệu lực
        # khi caller CŨNG truyền ``eval_cache`` (dict thật, không None) — đó là kênh DUY NHẤT
        # đưa kết quả worker về (xem _prefetch_parallel); executor+n_jobs>1 mà eval_cache=None
        # vẫn ĐÚNG (đường tuần tự tự lo) nhưng KHÔNG tăng tốc.
        self.executor = executor
        # A2: họ nhân tố ClosedLoop đã đóng (0 pass sau max_per_family) — cá thể thuộc họ này
        # bị chặn TRƯỚC backtest trong _evaluate_population (xem gate A2 ở đó).
        self.saturated_families = frozenset(saturated_families)
        # A3: cache in-memory cấp phiên (canonical_hash -> ("ok", daily_pnl, metrics) |
        # ("error", fail_reasons)) cho phần THUẦN eval/backtest/metrics — hàm xác định của
        # (expr, config, data) bất biến trong phiên. ``daily_pnl`` (ndarray 1 chiều) THAY cho
        # ``BacktestResult``/weights đầy đủ (fix OOM review cuối C1 — xem module docstring).
        # Chủ sở hữu (GPIdeaSource) truyền dict CHIA SẺ xuyên nhiều GPEngine/batch; None = tắt
        # cache (mặc định, dùng cho test độc lập không quan tâm cache).
        self.eval_cache = eval_cache
        # B1: nhóm field CỐ ĐỊNH cho epoch reseed (GPIdeaSource xoay dataset field ưu tiên
        # mỗi epoch) — None = dùng toàn bộ field của data (hành vi cũ, mọi caller trước B1).
        # PHẢI là tập con của data.field_names() — caller (composition root) lọc trước khi
        # truyền; run() không tự kiểm tra lại.
        self.fields_override = fields_override
        # B2: nhóm field theo dataset để init_population sinh leaf ngẫu nhiên two-stage
        # (chọn nhóm dataset uniform trước, field trong nhóm uniform sau) — dataset ít field
        # không bị dataset đông field (vd price/volume) áp đảo xác suất. None = uniform phẳng
        # (hành vi cũ, mọi caller trước B2).
        self.field_groups = field_groups

    def _evaluate_individual(
        self, ind: Individual, pool_corr: PoolCorrelation,
    ) -> tuple[FitnessVector | None, str, list[str], "np.ndarray | None", AlphaMetrics | None]:
        """Đánh giá một cá thể: eval signal → build danh mục → backtest → metrics → gate.

        Trả ``(fitness, status, fail_reasons, daily_pnl, metrics)``; việc persist do caller
        (``run``) đảm nhiệm. ``status`` là một trong: ``'passed'`` | ``'failed_gate'`` |
        ``'invalid'`` | ``'error'``. ``fail_reasons`` LUÔN là ``list[str]`` (rỗng khi pass).
        Quy ước bắt lỗi:

        - Eval AST hỏng (operator thiếu impl, kiểu sai) → ``'error'`` (fitness/daily_pnl/
          metrics None).
        - Backtest/metrics ném exception → ``'error'``.
        - Gate hard-fail (depth/fields/self_corr/concentration) → ``'failed_gate'`` (vẫn có
          daily_pnl + metrics + fitness để cá thể còn tham gia chọn lọc, không bị loại khỏi
          quần thể).
        - Pass mọi hard gate → ``'passed'``.

        NGOẠI LỆ (A2): hàm này KHÔNG được gọi cho cá thể vô nghĩa/thuộc họ-đã-đóng — nhánh đó
        bị ``_evaluate_population`` chặn TRƯỚC khi tới đây, tự persist status ``'failed_gate'``
        riêng với ``daily_pnl=None``, ``metrics=None``, ``fitness=None`` (KHÔNG có backtest,
        KHÔNG tham gia chọn lọc) — khác hẳn ``'failed_gate'`` sinh RA TỪ HÀM NÀY (luôn có
        daily_pnl + metrics + fitness như mô tả trên).

        Lưu ý: ``Evaluator`` hiện gói lỗi parse-time vào exception runtime nên không có nhánh
        ``'invalid'`` riêng ở đây; ``'invalid'`` để dành cho cây sai cấu trúc registry (nếu
        tầng eval phân biệt sau này). ``SubexprCache`` tạo MỚI mỗi cá thể — tránh chia sẻ
        state cache giữa các lần eval khác nhau (B6).

        A3 (cache xuyên batch): phần THUẦN (eval AST → build danh mục → backtest → metrics)
        là hàm xác định của ``(expr, config, data)`` — bất biến trong một phiên chạy — nên
        được cache theo ``canonical_hash`` (``self.eval_cache``, dict CHIA SẺ do
        ``GPIdeaSource`` truyền vào xuyên nhiều ``GPEngine``/batch). Phần phụ thuộc pool
        (gate/pool_rho/fitness ở dưới) LUÔN tính lại tươi vì pool lớn dần trong phiên.

        Fix OOM (review cuối C1): entry cache "ok" chỉ giữ ``(daily_pnl, metrics)`` — KHÔNG
        giữ ``BacktestResult``/``weights`` (nặng ~10-15MB/cá thể trên panel thật). ``bt`` chỉ
        tồn tại cục bộ trong nhánh miss cache, không lọt ra khỏi hàm này."""
        ch = ind.expr.accept(CanonicalHasher())
        cached = self.eval_cache.get(ch) if self.eval_cache is not None else None
        if cached is not None and cached[0] == "error":
            # Copy list để caller sửa list trả về không làm hỏng entry cache nội bộ.
            return None, "error", list(cached[1]), None, None
        if cached is not None:
            _tag, daily_pnl, metrics = cached
        else:
            try:
                ctx = EvalContext(data=self.data, registry=self.registry, cache=SubexprCache())
                evaluator = Evaluator(ctx)
                signal = evaluator.evaluate(ind.expr)
            except Exception as exc:  # noqa: BLE001 — engine phải sống sót mọi lỗi cây
                reasons = [f"eval: {type(exc).__name__}: {exc}"]
                # Lưu BẢN SAO vào cache — nếu lưu thẳng ``reasons`` (cùng object trả cho
                # caller) thì caller mutate list trả về sẽ rò ngược vào entry cache nội bộ.
                if self.eval_cache is not None:
                    self.eval_cache[ch] = ("error", list(reasons))
                return None, "error", reasons, None, None

            try:
                weights = PortfolioBuilder().build(signal, self.config, self.data)
                bt = Backtester().run(weights, self.data)
            except Exception as exc:  # noqa: BLE001
                reasons = [f"backtest: {type(exc).__name__}: {exc}"]
                if self.eval_cache is not None:
                    self.eval_cache[ch] = ("error", list(reasons))
                return None, "error", reasons, None, None

            try:
                metrics = MetricsCalculator().compute(bt, self.data)
            except Exception as exc:  # noqa: BLE001
                reasons = [f"metrics: {type(exc).__name__}: {exc}"]
                if self.eval_cache is not None:
                    self.eval_cache[ch] = ("error", list(reasons))
                return None, "error", reasons, None, None

            # Chỉ giữ lại daily_pnl (ndarray 1 chiều) — bt/weights bị vứt khi hàm return,
            # không giữ tham chiếu nào khác tới BacktestResult đầy đủ.
            daily_pnl = bt.daily_pnl
            if self.eval_cache is not None:
                self.eval_cache[ch] = ("ok", daily_pnl, metrics)

        depth = ind.expr.accept(DepthVisitor())
        fields = ind.expr.accept(FieldCollector(self.registry))
        fields_ok = bool(fields) and fields.issubset(self.data.field_names())

        verdict = GateEvaluator().evaluate_with_pool(
            metrics, candidate_pnl=daily_pnl, candidate_dates=self.data.dates,
            pool_corr=pool_corr, depth=depth, fields_ok=fields_ok,
        )
        # self_corr tính một lần (gate cũng tính nội bộ; ta cần lại cho fitness + persist).
        pool_rho, _worst_id = pool_corr.max_corr(daily_pnl, self.data.dates)
        complexity = ind.expr.accept(ComplexityVisitor())
        # n_trials=1: chưa theo dõi số lần thử per-cá-thể nên không haircut deflation ở đây;
        # đa dạng quần thể đã do NSGA-II + pool_corr_penalty đảm nhiệm (xem fitness_vec).
        fv = from_metrics(
            metrics, complexity=complexity, pool_corr=pool_rho, pop_corr=0.0, n_trials=1,
        )

        if not verdict.passed:
            return fv, "failed_gate", list(verdict.hard_failures), daily_pnl, metrics
        return fv, "passed", [], daily_pnl, metrics

    def _config_json(self) -> str:
        """Khóa cấu hình stage cho cache/DB — ``sort_keys=True`` để canonical (Minor P5):
        cùng cấu hình PortfolioConfig phải sinh CÙNG chuỗi bất kể thứ tự key."""
        return json.dumps(
            {
                "neutralization": self.config.neutralization.name,
                "decay": self.config.decay,
                "truncation": self.config.truncation,
                "scale_book": self.config.scale_book,
                "delay": self.config.delay,
            },
            sort_keys=True,
        )

    def _persist(
        self,
        ind: Individual,
        status: str,
        fail_reasons: list[str],
        daily_pnl: "np.ndarray | None",
        metrics: AlphaMetrics | None,
        self_corr: float | None,
    ) -> None:
        """Upsert expression + ``record_evaluation`` (mọi outcome: pass/fail/seed — B11
        avoid-list) + ``save_pool_pnl`` khi pass. ``metrics`` do CALLER truyền vào (đã tính
        ở ``_evaluate_individual``, KHÔNG recompute lại ở đây — fix OOM/lãng phí review cuối
        C1: trước đây hàm này gọi lại ``MetricsCalculator().compute(bt, ...)`` dù caller đã
        có sẵn kết quả, vừa tốn CPU vừa buộc giữ ``bt`` đầy đủ chỉ để tính lại). ``metrics``
        chỉ được dùng cho trạng thái ``passed``/``failed_gate``; ``invalid``/``error`` ->
        ``metrics_for_db=None`` (cột metric DB để trống) dù caller có lỡ truyền metrics khác
        None (không xảy ra trong thực tế — mọi nhánh error/invalid trả metrics=None).

        NGOẠI LỆ (A2): cá thể vô nghĩa/thuộc họ-đã-đóng cũng persist status ``'failed_gate'``
        nhưng gọi ``_persist`` với ``daily_pnl=None``, ``metrics=None`` (chưa từng backtest,
        bị chặn TRƯỚC bước đó ở ``_evaluate_population``) — nhánh ``if status in {...}`` bên
        dưới tự nhiên rơi vào ``metrics_for_db=None`` cho trường hợp này, KHÔNG phải lỗi."""
        expr_string = ind.expr.accept(Serializer())
        canonical_hash = ind.expr.accept(CanonicalHasher())
        depth = ind.expr.accept(DepthVisitor())
        complexity = ind.expr.accept(ComplexityVisitor())
        fields = ind.expr.accept(FieldCollector(self.registry))

        expr_id = self.repo.upsert_expression(
            expr_string, canonical_hash, depth, complexity, fields,
        )

        metrics_for_db: AlphaMetrics | None = (
            metrics if status in {"passed", "failed_gate"} else None
        )

        eval_id = self.repo.record_evaluation(
            expression_id=expr_id,
            config_json=self._config_json(),
            data_window=self.data_window,
            metrics=metrics_for_db,
            self_corr_max=self_corr,
            status=status,
            fail_reasons=fail_reasons,
            seed=self.seed,
        )

        if status == "passed" and daily_pnl is not None:
            self.repo.save_pool_pnl(eval_id, self.data.dates, daily_pnl)

    def _prefetch_parallel(self, population: list[Individual]) -> None:
        """C1: chạy PHẦN THUẦN (eval AST → danh mục → backtest → metrics) SONG SONG qua
        ``self.executor`` cho các cá thể sẽ được backtest, rồi ghi thẳng kết quả vào
        ``self.eval_cache`` — CÙNG ĐỊNH DẠNG ``eval_thuan`` trả về, đúng những gì
        ``_evaluate_individual`` đọc ở đầu hàm (``("ok", daily_pnl, metrics) | ("error",
        reasons)`` — ``daily_pnl`` là ``bt.daily_pnl``, KHÔNG phải ``BacktestResult`` đầy đủ,
        xem fix OOM ở module docstring). Nhờ vậy vòng lặp TUẦN TỰ THEO INDEX GỐC bên dưới
        (không đổi so với trước C1) tự động HIT cache thay vì backtest lại trong process
        chính — không cần đường code riêng cho gate/pool_corr/fitness/persist song song.

        Lọc TRƯỚC khi submit lặp lại đúng điều kiện A2 (``check_meaningful`` + họ-đã-đóng)
        của vòng lặp tuần tự bên dưới — cá thể bị A2 chặn KHÔNG được submit (không tốn worker
        cho cá thể sẽ bị vứt trước backtest); vòng tuần tự tính lại A2 (rẻ, thuần, không
        side-effect) để tự persist reasons — không trùng lặp logic, chỉ trùng lặp một phép
        tính rẻ.

        CHỈ chạy khi có ``self.eval_cache``: đây là kênh DUY NHẤT chuyển kết quả từ process
        con về vòng lặp tuần tự (không có SQLite/IPC nào khác ở tầng này) — ``eval_cache=None``
        thì bỏ qua hoàn toàn (không submit gì), vòng tuần tự phía sau tự lo — vẫn ĐÚNG (chỉ
        không tăng tốc, không phá kết quả)."""
        if self.eval_cache is None:
            return
        to_submit: dict[str, str] = {}  # canonical_hash -> expr_string, dedup trong 1 lô
        for ind in population:
            if ind.fitness is not None:
                continue
            ok, _ly_do = check_meaningful(ind.expr, self.registry)
            ho: str | None = None
            if ok:
                expr_str = ind.expr.accept(Serializer())
                ho = classify_family(expr_str)
                if ho not in self.saturated_families:
                    ho = None
            if not ok or ho is not None:
                continue  # A2 chặn — không backtest, vòng tuần tự phía sau tự persist
            ch = ind.expr.accept(CanonicalHasher())
            if ch in self.eval_cache or ch in to_submit:
                continue  # đã có sẵn trong cache (A3) hoặc đã đưa vào lô nộp lần này
            to_submit[ch] = ind.expr.accept(Serializer())
        if not to_submit:
            return
        futures = {
            ch: self.executor.submit(eval_thuan, expr) for ch, expr in to_submit.items()  # type: ignore[union-attr]
        }
        for ch, fut in futures.items():
            self.eval_cache[ch] = fut.result()

    def _evaluate_population(
        self, population: list[Individual], pool_corr: PoolCorrelation,
    ) -> tuple[int, int]:
        """Đánh giá + persist mọi cá thể CHƯA có fitness. Trả ``(n_evaluated, n_passed)``.
        Cá thể đã eval ở thế hệ trước (fitness != None, được NSGA-II giữ lại) bỏ qua.

        C1: khi ``self.n_jobs > 1`` và có ``self.executor``, phần THUẦN (eval/backtest/
        metrics) của các cá thể MISS cache được tính SONG SONG trước (``_prefetch_parallel``,
        ghi kết quả vào ``eval_cache``) — vòng lặp bên dưới sau đó CHẠY Y HỆT đường tuần tự
        (index gốc), chỉ khác là phần lớn cá thể đã HIT cache nên bỏ qua backtest lặp lại.
        ``n_jobs=1`` (mặc định) hoặc không có executor: bỏ qua bước này, đường cũ nguyên vẹn."""
        if self.n_jobs > 1 and self.executor is not None:
            self._prefetch_parallel(population)
        n_evaluated = 0
        n_passed = 0
        for ind in population:
            if ind.fitness is not None:
                continue
            # A2: chặn TRƯỚC backtest — cá thể vô nghĩa/họ-đóng không tốn eval, không chiếm
            # suất NSGA-II (fitness=None -> bị loại khỏi chọn lọc), vẫn persist để avoid-list học.
            ok, ly_do = check_meaningful(ind.expr, self.registry)
            ho: str | None = None
            if ok:
                expr_str = ind.expr.accept(Serializer())
                ho = classify_family(expr_str)
                if ho not in self.saturated_families:
                    ho = None  # không vi phạm
            if not ok or ho is not None:
                reasons = [f"degenerate: {ly_do}"] if not ok else [f"họ đã đóng: {ho}"]
                self._persist(ind, "failed_gate", reasons, None, None, None)
                ind.fitness = None
                n_evaluated += 1
                continue
            fv, status, reasons, daily_pnl, metrics = self._evaluate_individual(ind, pool_corr)
            self_corr: float | None = None
            if daily_pnl is not None:
                rho, _worst = pool_corr.max_corr(daily_pnl, self.data.dates)
                self_corr = float(rho)
            self._persist(ind, status, reasons, daily_pnl, metrics, self_corr)
            ind.fitness = fv  # slots non-frozen — gán sau init (xem Individual Task 7.1)
            n_evaluated += 1
            if status == "passed":
                n_passed += 1
        return n_evaluated, n_passed

    def _make_offspring(
        self,
        evaluated: list[Individual],
        fields: tuple[str, ...],
        rng: np.random.Generator,
        generation: int,
    ) -> list[Individual]:
        """Sinh ``pop_size`` con từ các cá thể đã eval: crossover (lấy 2 cha) /
        mutation (point 0.4 / subtree 0.4 / hoist 0.2) / sao chép, theo
        ``crossover_rate``/``mutation_rate``. Con KHÔNG kế thừa fitness (=None)."""
        offspring: list[Individual] = []
        while len(offspring) < self.pop_size:
            u = rng.random()
            if u < self.crossover_rate and len(evaluated) >= 2:
                i, j = rng.choice(len(evaluated), size=2, replace=False)
                c1, c2 = crossover(
                    evaluated[int(i)].expr, evaluated[int(j)].expr, rng, self.max_depth,
                )
                offspring.append(Individual(expr=c1, generation=generation))
                if len(offspring) < self.pop_size:
                    offspring.append(Individual(expr=c2, generation=generation))
            elif u < self.crossover_rate + self.mutation_rate:
                parent = evaluated[int(rng.integers(0, len(evaluated)))]
                v = rng.random()
                if v < 0.4:
                    mutated = point_mutation(parent.expr, self.registry, rng, fields)
                elif v < 0.8:
                    mutated = subtree_mutation(
                        parent.expr, self.registry, rng, fields, self.max_depth,
                    )
                else:
                    mutated = hoist_mutation(parent.expr, rng, self.registry)
                offspring.append(Individual(expr=mutated, generation=generation))
            else:
                parent = evaluated[int(rng.integers(0, len(evaluated)))]
                offspring.append(Individual(expr=parent.expr, generation=generation))
        return offspring

    def run(self) -> GPRunResult:
        """Vòng lặp tiến hóa end-to-end (thuần Python, xác định theo ``seed``):

        1. ``rng = np.random.default_rng(seed)``; ``seeds = all_seed_cores(...)``.
        2. ``population = init_population(...)`` (ramped half-and-half + seeding).
        3. Lặp ``n_generations`` lần: đánh giá + persist toàn quần thể (pass/fail/seed),
           sinh offspring (crossover/mutation/sao chép), ``dedup_population`` theo
           canonical_hash, ``nsga2_select`` giữ ``pop_size`` cá thể (Pareto + crowding).
        4. Đánh giá thế hệ cuối (offspring chưa eval) → persist.
        5. Trả ``GPRunResult`` (quần thể cuối + best theo ``sharpe_deflated`` + best
           combinable-aware (T2.1, ``_select_best_combinable``) + thống kê).

        ``pool_corr`` được nạp lại từ DB ở đầu mỗi vòng (pool lớn dần khi có alpha pass)."""
        rng = np.random.default_rng(self.seed)
        fields = (
            tuple(sorted(self.fields_override))
            if self.fields_override
            else tuple(sorted(self.data.field_names()))
        )
        seed_cores = all_seed_cores(
            with_llm=self.with_llm_seeds,
            field_names=set(self.data.field_names()),
        )
        population = init_population(
            registry=self.registry,
            rng=rng,
            population_size=self.pop_size,
            seed_cores=seed_cores,
            fields=fields,
            max_depth=self.max_depth,
            seed_offset=self.seed_offset,
            field_groups=self.field_groups,
        )

        total_eval = 0
        total_passed = 0
        for gen in range(self.n_generations):
            pool_corr = PoolCorrelation(pool=self.repo.load_pool())
            n_ev, n_pa = self._evaluate_population(population, pool_corr)
            total_eval += n_ev
            total_passed += n_pa

            evaluated = [i for i in population if i.fitness is not None]
            if not evaluated:
                break  # toàn quần thể error/invalid — không biến đổi được, dừng sớm
            offspring = self._make_offspring(evaluated, fields, rng, gen + 1)
            # Đánh giá offspring TRƯỚC chọn lọc (μ+λ): NSGA-II yêu cầu mọi cá thể có
            # fitness. pool_corr nạp lại vì pool có thể vừa lớn lên ở bước eval trên.
            pool_corr = PoolCorrelation(pool=self.repo.load_pool())
            n_ev, n_pa = self._evaluate_population(offspring, pool_corr)
            total_eval += n_ev
            total_passed += n_pa
            # Chỉ giữ cá thể có fitness để chọn lọc; loại error/invalid (đã persist fail).
            pool_for_select = [
                i for i in (population + offspring) if i.fitness is not None
            ]
            combined = dedup_population(pool_for_select, self.registry)
            population = nsga2_select(combined, self.pop_size, rng)

        # Đánh giá thế hệ cuối: offspring vừa được chọn nhưng chưa eval.
        pool_corr = PoolCorrelation(pool=self.repo.load_pool())
        n_ev, n_pa = self._evaluate_population(population, pool_corr)
        total_eval += n_ev
        total_passed += n_pa

        evaluated_final = [i for i in population if i.fitness is not None]
        best = max(
            evaluated_final,
            key=lambda i: i.fitness.sharpe_deflated,  # type: ignore[union-attr]
            default=None,
        )
        # T2.1: bản combinable-aware, xem docstring GPRunResult/_select_best_combinable.
        best_combinable = _select_best_combinable(evaluated_final)

        return GPRunResult(
            generations_run=self.n_generations,
            final_population=population,
            best_by_sharpe=best,
            n_evaluated=total_eval,
            n_passed=total_passed,
            seed=self.seed,
            best_combinable=best_combinable,
        )
