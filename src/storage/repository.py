"""Lưu alpha và kết quả simulation vào DB."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.backtest.metrics_local import AlphaMetrics
from src.local_types import Dates
from src.simulation.simulator import SimulationResult
from src.storage.models import (
    AlphaModel,
    BrainSimLinkModel,
    DeadFieldModel,
    EvaluationModel,
    ExpressionModel,
    FailureModel,
    InvalidFieldModel,
    PoolPnlModel,
    SimulationModel,
    SubmissionModel,
    TriedHashModel,
)


def expr_hash(expression: str, config: str | None = None) -> str:
    """Hash biểu thức (+config) để cache simulation. GĐ2 dùng config mặc định cố định."""
    payload = expression if config is None else f"{expression}|{config}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class InvalidFieldRepository:
    """Blacklist field 'chết' (WQ từ chối khi simulate) — tự học để khỏi sinh lại."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def record(
        self, field_id: str, region: str | None = None,
        universe: str | None = None, reason: str = "",
    ) -> None:
        """Ghi/cập nhật field chết (idempotent theo field_id)."""
        session = self.session_factory()
        try:
            session.merge(
                InvalidFieldModel(
                    field_id=field_id, region=region, universe=universe, reason=reason
                )
            )
            session.commit()
        finally:
            session.close()

    def blacklist(self) -> set[str]:
        """Trả tập field id bị đánh dấu chết."""
        session = self.session_factory()
        try:
            return {r.field_id for r in session.query(InvalidFieldModel).all()}
        finally:
            session.close()


class AlphaRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def save_alpha(
        self,
        expression: str,
        source: str = "manual",
        hypothesis=None,
        description: str | None = None,
        parent_id: str | None = None,
    ) -> str:
        alpha_id = uuid.uuid4().hex
        hyp = json.dumps(hypothesis, ensure_ascii=False) if isinstance(hypothesis, (dict, list)) else hypothesis
        session = self.session_factory()
        try:
            session.add(
                AlphaModel(
                    id=alpha_id,
                    expression=expression,
                    source=source,
                    hypothesis=hyp,
                    description=description,
                    parent_id=parent_id,
                )
            )
            session.commit()
            return alpha_id
        finally:
            session.close()

    def save_simulation(
        self,
        result: SimulationResult,
        region: str,
        universe: str,
        source: str = "manual",
        score: float | None = None,
        alpha_id: str | None = None,
        config_key: str | None = None,
    ) -> str:
        """Lưu alpha (nếu chưa có) + simulation. Trả simulation id."""
        session = self.session_factory()
        try:
            if alpha_id is None:
                alpha_id = uuid.uuid4().hex
                session.add(AlphaModel(id=alpha_id, expression=result.expression, source=source))

            sim_id = uuid.uuid4().hex
            session.add(
                SimulationModel(
                    id=sim_id,
                    alpha_id=alpha_id,
                    expr_hash=expr_hash(result.expression, config_key),
                    wq_alpha_id=result.alpha_id,
                    region=region,
                    universe=universe,
                    sharpe=result.sharpe,
                    fitness=result.fitness,
                    turnover=result.turnover,
                    drawdown=result.drawdown,
                    margin=result.margin,
                    returns=result.returns,
                    score=score,
                    status=result.status,
                    raw_result=json.dumps(result.raw, ensure_ascii=False),
                    failed_checks=json.dumps(result.failed_checks, ensure_ascii=False),
                )
            )
            session.commit()
            return sim_id
        finally:
            session.close()

    def get_cached_simulation(self, expression: str, config_key: str | None = None) -> SimulationModel | None:
        """Trả simulation đã lưu (mới nhất) cho biểu thức — bỏ qua kết quả 'error'."""
        h = expr_hash(expression, config_key)
        session = self.session_factory()
        try:
            return (
                session.query(SimulationModel)
                .filter(SimulationModel.expr_hash == h, SimulationModel.status != "error")
                .order_by(SimulationModel.sim_at.desc())
                .first()
            )
        finally:
            session.close()

    def record_failure(
        self, expression: str, category: str, reason: str = "", source: str = "llm"
    ) -> str:
        fail_id = uuid.uuid4().hex
        session = self.session_factory()
        try:
            session.add(
                FailureModel(
                    id=fail_id,
                    expression=expression,
                    category=category,
                    reason=reason,
                    source=source,
                )
            )
            session.commit()
            return fail_id
        finally:
            session.close()

    def recent_failures(self, limit: int = 20) -> list[FailureModel]:
        session = self.session_factory()
        try:
            return (
                session.query(FailureModel)
                .order_by(FailureModel.created_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def top_simulated(self, limit: int = 5) -> list[tuple[str, float, float]]:
        """Top alpha ĐÃ mô phỏng (có metric thật, bỏ status='error') theo sharpe
        giảm dần. Trả [(expression, sharpe, fitness)] để khai thác (exploit) — đề
        xuất biến thể của tín hiệu đã cho kết quả tốt, kể cả khi chưa pass."""
        session = self.session_factory()
        try:
            rows = (
                session.query(
                    AlphaModel.expression, SimulationModel.sharpe, SimulationModel.fitness
                )
                .join(SimulationModel, SimulationModel.alpha_id == AlphaModel.id)
                .filter(SimulationModel.status != "error", SimulationModel.sharpe.isnot(None))
                .order_by(SimulationModel.sharpe.desc())
                .limit(limit)
                .all()
            )
            return [(r[0], r[1], r[2] if r[2] is not None else 0.0) for r in rows]
        finally:
            session.close()

    def zoo(self, limit: int = 20) -> list[AlphaModel]:
        """Alpha zoo (T2.10): các alpha đã pass, sort giảm theo score của simulation."""
        session = self.session_factory()
        try:
            rows = (
                session.query(AlphaModel)
                .join(SimulationModel, SimulationModel.alpha_id == AlphaModel.id)
                .filter(SimulationModel.status == "passed")
                .order_by(SimulationModel.score.desc().nullslast())
                .limit(limit)
                .all()
            )
            return rows
        finally:
            session.close()


@dataclass(frozen=True, slots=True)
class SubmitReadyAlpha:
    """Một alpha đạt CẢ BA điều kiện nộp Regular thật (Task 8): `SimulationModel.status ==
    'passed'` (WQ đã chấm Sharpe/Fitness/Turnover/IS-Ladder theo tier tài khoản — không tự
    đoán lại ngưỡng, xem `SubmissionManager.select_candidates`), `failed_checks == []` (không
    WQ check nào tự FAIL), và `self_corr` ĐÃ VERIFY (đo Brain thật, khác cờ Power Pool eligible
    chỉ là cấu trúc — commit `e27821d`) nhỏ hơn `SELF_CORR_MAX`. Dùng cho khối "SẴN SÀNG NỘP"
    in cuối phiên `ClosedLoop._report`."""

    wq_alpha_id: str
    expression: str
    sharpe: float | None
    self_corr: float


class MiniBrainRepository:
    """Repository cho luồng MiniBrain local (Expression/Evaluation/PoolPnl/DeadField).
    Tách khỏi AlphaRepository (luồng Brain-sim cũ) — hai luồng dữ liệu độc lập, schema
    khác nhau, không chia sẻ session pattern ngoài cấu trúc try/finally. NGOẠI LỆ có chủ đích:
    `submit_ready_alphas` (Task 8) đọc CHÉO sang AlphaModel/SimulationModel/SubmissionModel —
    báo cáo cuối phiên cần bức tranh đầy đủ CẢ HAI luồng (một alpha sẵn sàng nộp có thể sinh
    ra từ luồng cũ `research`/`submit` HOẶC từ closed-loop), không phải ghi dữ liệu mới nên
    không phá vỡ ranh giới ghi (write) giữa hai luồng."""

    # session_factory không annotate, giữ nhất quán AlphaRepository/InvalidFieldRepository
    # (constructor pattern hiện có, mypy --strict đã báo no-untyped-def tiền-tồn ở đó).
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    def upsert_expression(
        self, expr_string: str, canonical_hash: str, depth: int, complexity: int,
        fields: set[str],
    ) -> int:
        """Dedup theo canonical_hash: đã có -> trả id cũ (không insert/update trùng);
        chưa có -> insert mới và trả id mới."""
        session = self.session_factory()
        try:
            existing = (
                session.query(ExpressionModel)
                .filter_by(canonical_hash=canonical_hash)
                .first()
            )
            if existing is not None:
                # mypy --strict: existing.id suy luận ra Any (SQLAlchemy ORM động) ->
                # no-any-return; cùng hạn chế tiền-tồn như session.query(...) ở
                # AlphaRepository (mypy không gõ kiểu instrumented attribute).
                return existing.id  # type: ignore[no-any-return]
            row = ExpressionModel(
                canonical_hash=canonical_hash, expr_string=expr_string, depth=depth,
                complexity=complexity, fields_json=json.dumps(sorted(fields)),
            )
            session.add(row)
            session.commit()
            # mypy --strict: ở đây row là ExpressionModel(...) mới (không qua query) nên
            # mypy gõ được class cụ thể -> row.id là Column[int] (kiểu khai báo cấp
            # class), khác nhánh existing.id (Any, suy luận từ session.query động).
            return row.id  # type: ignore[return-value]
        finally:
            session.close()

    def record_evaluation(
        self, expression_id: int, config_json: str, data_window: str,
        metrics: AlphaMetrics | None, self_corr_max: float | None, status: str,
        fail_reasons: list[str], seed: int | None,
    ) -> int:
        """Lưu CẢ pass lẫn fail (B11: avoid-list cần biết alpha nào đã thử và vì sao
        fail). Merge thủ công theo khóa duy nhất (expression_id, config_json,
        data_window): đã tồn tại -> cập nhật outcome mới nhất (không nhân đôi); chưa có
        -> insert mới. metrics=None (fail trước khi backtest sinh được metrics) -> mọi
        cột metric numeric = None."""
        session = self.session_factory()
        try:
            existing = (
                session.query(EvaluationModel)
                .filter_by(
                    expression_id=expression_id, config_json=config_json,
                    data_window=data_window,
                )
                .first()
            )
            row = existing or EvaluationModel(
                expression_id=expression_id, config_json=config_json,
                data_window=data_window,
            )
            row.status = status
            row.fail_reasons = json.dumps(fail_reasons, ensure_ascii=False)
            row.self_corr_max = self_corr_max
            row.seed = seed
            if metrics is not None:
                row.sharpe = metrics.sharpe
                row.annual_return = metrics.annual_return
                row.turnover = metrics.turnover
                row.max_drawdown = metrics.max_drawdown
                row.fitness = metrics.fitness
                row.weight_concentration = metrics.weight_concentration
                row.per_year_json = json.dumps(metrics.per_year_sharpe)
            else:
                row.sharpe = None
                row.annual_return = None
                row.turnover = None
                row.max_drawdown = None
                row.fitness = None
                row.weight_concentration = None
                row.per_year_json = None
            if existing is None:
                session.add(row)
            session.commit()
            # mypy --strict: row = existing or EvaluationModel(...) -> kiểu hợp nhất
            # "Any | Column[int]" (existing.id Any từ query động, nhánh mới Column[int]
            # cấp class) -> không khớp int khai báo; .id là int thật ở runtime.
            return row.id  # type: ignore[return-value]
        finally:
            session.close()

    def load_pool(self) -> dict[int, tuple[Dates, npt.NDArray[np.float64]]]:
        """Đọc TẤT CẢ PoolPnlModel, trả {evaluation_id: (dates, pnl)} — PoolCorrelation
        (Task 6.1) cần CẢ dates để align candidate với từng alpha trong pool có lịch sử
        dài-ngắn khác nhau (PoolCorrelation.__init__ nhận đúng dict[int, tuple[Dates,
        NDArray[float64]]] này)."""
        session = self.session_factory()
        try:
            rows = session.query(PoolPnlModel).all()
            result: dict[int, tuple[Dates, npt.NDArray[np.float64]]] = {}
            for row in rows:
                dates = np.frombuffer(row.dates_blob, dtype="datetime64[D]")
                # .copy() bắt buộc cho pnl: np.frombuffer() trần trả mảng read-only (view
                # trên buffer bytes) — Phase 6 (max_corr) thao tác in-place (demean) trên
                # mảng này sẽ raise ValueError nếu không copy thành bản ghi-được. dates chỉ
                # đọc (sort/searchsorted không in-place) nên không cần copy.
                pnl = np.frombuffer(row.pnl_blob, dtype=np.float64).copy()
                result[row.evaluation_id] = (dates, pnl)
            return result
        finally:
            session.close()

    def save_pool_pnl(
        self, evaluation_id: int, dates: npt.NDArray[np.datetime64],
        pnl: npt.NDArray[np.float64],
    ) -> None:
        """Pack dates/pnl thành blob nhị phân; merge (ghi đè) theo evaluation_id — gọi
        lại cùng evaluation_id phải cập nhật, không lỗi PK trùng."""
        session = self.session_factory()
        try:
            session.merge(
                PoolPnlModel(
                    evaluation_id=evaluation_id,
                    dates_blob=dates.astype("datetime64[D]").tobytes(),
                    pnl_blob=pnl.astype(np.float64).tobytes(),
                )
            )
            session.commit()
        finally:
            session.close()

    def add_dead_field(self, name: str, reason: str = "") -> None:
        """Ghi/cập nhật field 'chết' theo nghĩa MiniBrain (idempotent theo name)."""
        session = self.session_factory()
        try:
            session.merge(DeadFieldModel(name=name, reason=reason))
            session.commit()
        finally:
            session.close()

    def is_dead_field(self, name: str) -> bool:
        session = self.session_factory()
        try:
            return (
                session.query(DeadFieldModel).filter_by(name=name).first() is not None
            )
        finally:
            session.close()

    def result_cache_get(
        self, canonical_hash: str, config_json: str, data_window: str,
    ) -> AlphaMetrics | None:
        """Chỉ cache hit cho kết quả PASS (status == 'passed' và sharpe có giá trị) —
        fail không có metrics đầy đủ để tái dùng an toàn. Dựng lại AlphaMetrics từ cột,
        ép key per_year_json về int (JSON chỉ có key chuỗi)."""
        session = self.session_factory()
        try:
            row = (
                session.query(EvaluationModel)
                .join(ExpressionModel, EvaluationModel.expression_id == ExpressionModel.id)
                .filter(
                    ExpressionModel.canonical_hash == canonical_hash,
                    EvaluationModel.config_json == config_json,
                    EvaluationModel.data_window == data_window,
                    EvaluationModel.status == "passed",
                )
                .first()
            )
            if row is None or row.sharpe is None:
                return None
            per_year = {
                int(k): v for k, v in json.loads(row.per_year_json or "{}").items()
            }
            return AlphaMetrics(
                sharpe=row.sharpe, annual_return=row.annual_return, turnover=row.turnover,
                max_drawdown=row.max_drawdown, fitness=row.fitness,
                per_year_sharpe=per_year, weight_concentration=row.weight_concentration,
            )
        finally:
            session.close()

    def result_cache_put(
        self, canonical_hash: str, expr_string: str, depth: int, complexity: int,
        fields: set[str], config_json: str, data_window: str, metrics: AlphaMetrics,
        seed: int | None,
    ) -> int:
        """Hàm tiện ích gộp upsert_expression + record_evaluation(status="passed") cho
        ResultCache.put (Task 5.4). self_corr_max=None: self-corr phụ thuộc pool tại
        thời điểm eval, không phải thuộc tính bất biến của expression — cache nó sẽ
        stale khi pool đổi. Caller cần lưu self_corr cùng lúc thì gọi record_evaluation
        trực tiếp."""
        expr_id = self.upsert_expression(expr_string, canonical_hash, depth, complexity, fields)
        return self.record_evaluation(
            expr_id, config_json, data_window, metrics, self_corr_max=None,
            status="passed", fail_reasons=[], seed=seed,
        )

    def top_n(self, n: int) -> list[tuple[str, float, float]]:
        """Top n biểu thức đã PASS theo sharpe giảm dần (NULL cuối), giống style
        top_simulated ở AlphaRepository, áp cho luồng MiniBrain."""
        session = self.session_factory()
        try:
            rows = (
                session.query(ExpressionModel.expr_string, EvaluationModel.sharpe,
                              EvaluationModel.fitness)
                .join(EvaluationModel, EvaluationModel.expression_id == ExpressionModel.id)
                .filter(EvaluationModel.status == "passed")
                .order_by(EvaluationModel.sharpe.desc().nullslast())
                .limit(n)
                .all()
            )
            return [(r[0], r[1], r[2] if r[2] is not None else 0.0) for r in rows]
        finally:
            session.close()

    def record_brain_sim(
        self, canonical_hash: str, expr_string: str, *, wq_alpha_id: str | None,
        region: str, universe: str, sharpe: float | None, fitness: float | None,
        turnover: float | None, self_corr: float | None, status: str,
        raw_json: str | None = None,
    ) -> int:
        """Ghi kết quả SIM Brain cho 1 expression MiniBrain. Merge theo khóa duy nhất
        (canonical_hash, region, universe): đã có -> cập nhật outcome mới nhất (không nhân
        đôi); chưa có -> insert. Trả id row."""
        session = self.session_factory()
        try:
            existing = (
                session.query(BrainSimLinkModel)
                .filter_by(canonical_hash=canonical_hash, region=region, universe=universe)
                .first()
            )
            row = existing or BrainSimLinkModel(
                canonical_hash=canonical_hash, region=region, universe=universe,
            )
            row.expr_string = expr_string
            row.wq_alpha_id = wq_alpha_id
            row.sharpe = sharpe
            row.fitness = fitness
            row.turnover = turnover
            row.self_corr = self_corr
            row.status = status
            row.raw_json = raw_json
            if existing is None:
                session.add(row)
            session.commit()
            return row.id  # type: ignore[return-value]
        finally:
            session.close()

    def load_brain_sims(self) -> list[BrainSimLinkModel]:
        """Trả mọi link Brain SIM đã ghi (cho calibrate feedback + avoid-list)."""
        session = self.session_factory()
        try:
            return session.query(BrainSimLinkModel).all()  # type: ignore[no-any-return]
        finally:
            session.close()

    def brain_pnl_pool(self) -> dict[str, float]:
        """Trả {canonical_hash: self_corr} cho các link status='passed' có self_corr != None
        — tra cứu nhanh self-corr Brain THẬT của alpha đã nộp (decorrelate tầng 2)."""
        session = self.session_factory()
        try:
            rows = (
                session.query(BrainSimLinkModel)
                .filter(BrainSimLinkModel.status == "passed")
                .filter(BrainSimLinkModel.self_corr.isnot(None))
                .all()
            )
            return {r.canonical_hash: float(r.self_corr) for r in rows}
        finally:
            session.close()

    def avoided_exprs(self) -> set[str]:
        """Trả {expr_string} của các link Brain SIM status='failed' — vòng kín bỏ qua, tránh
        refine lại ý tưởng đã hỏng trên Brain (avoid-list bền B11)."""
        session = self.session_factory()
        try:
            rows = (
                session.query(BrainSimLinkModel.expr_string)
                .filter(BrainSimLinkModel.status == "failed")
                .all()
            )
            return {r[0] for r in rows}
        finally:
            session.close()

    def avoided_hashes(self) -> set[str]:
        """Trả {canonical_hash} của các link Brain SIM status='failed' — avoid-list CROSS-
        SESSION theo hash (Pha 1.2). Bắt được cả biến thể fold scale dương (multiply(4,X) vs
        multiply(2,X)) mà avoided_exprs (so chuỗi thô) bỏ lọt."""
        session = self.session_factory()
        try:
            rows = (
                session.query(BrainSimLinkModel.canonical_hash)
                .filter(BrainSimLinkModel.status == "failed")
                .all()
            )
            return {r[0] for r in rows if r[0]}
        finally:
            session.close()

    def brain_proven_signals(self, min_sharpe: float, limit: int = 50) -> list[tuple[str, float]]:
        """Nguồn tín hiệu con 'đã chứng minh tốt trên BRAIN THẬT' cho combiner (Task 2 Fix 1,
        thay `good_signals_for_combine`): calibration đo được ρ=0.308 giữa fitness LOCAL và
        sharpe Brain (`logs/diag_combiner_20260712.md`) — xếp hạng theo fitness local chọn
        toàn GP junk (`group_neutralize(volume, sector)`…), các core Brain-proven KHÔNG có
        mặt. Trả (expr_string, sharpe) DISTINCT theo expr_string từ `BrainSimLinkModel` có
        sharpe >= min_sharpe, giữ sharpe CAO NHẤT nếu một expr có nhiều lần sim. KHÔNG lọc
        theo `status`: alpha 'failed' vì LOW_SHARPE (vd 1.04 < ngưỡng nộp IS_LADDER_FAIL 1.58)
        vẫn là component quý — Grinold-Kahn √N có thể đẩy nó lên ngưỡng nộp khi ghép. Sort
        sharpe giảm dần, cắt TOP `limit` (review fix: DB tích luỹ vô hạn theo thời gian —
        combiner chỉ cần các component mạnh nhất, không cần backtest cả kho)."""
        session = self.session_factory()
        try:
            rows = (
                session.query(BrainSimLinkModel.expr_string, BrainSimLinkModel.sharpe)
                .filter(BrainSimLinkModel.sharpe.isnot(None))
                .filter(BrainSimLinkModel.sharpe >= min_sharpe)
                .all()
            )
            best: dict[str, float] = {}
            for expr_string, sharpe in rows:
                sharpe_f = float(sharpe)
                if expr_string not in best or sharpe_f > best[expr_string]:
                    best[expr_string] = sharpe_f
            ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
            return ranked[:limit]
        finally:
            session.close()

    def record_avoided_hash(self, hash_value: str) -> None:
        """Ghi hash GỐC (pre-tune) của một ứng viên đã refine+sim — phục vụ pre-check
        cross-session đúng không gian hash (Task 6 fix). Idempotent theo PK `hash`; gọi lại
        với cùng giá trị không lỗi, không nhân đôi row."""
        session = self.session_factory()
        try:
            session.merge(TriedHashModel(hash=hash_value))
            session.commit()
        finally:
            session.close()

    def avoided_hashes_original(self) -> set[str]:
        """Trả mọi hash GỐC (pre-tune) đã ghi qua `record_avoided_hash` — nạp vào `seen` ở
        đầu phiên `ClosedLoop.run` để pre-check cross-session khớp đúng không gian hash gốc
        (khác `avoided_hashes()` vốn trả hash SAU tune từ `BrainSimLinkModel`, chỉ khớp status
        'failed')."""
        session = self.session_factory()
        try:
            rows = session.query(TriedHashModel.hash).all()
            return {r[0] for r in rows if r[0]}
        finally:
            session.close()

    def brain_local_sharpe_pairs(self) -> list[tuple[float, float]]:
        """Trả [(local_sharpe, brain_sharpe)] cho expression có CẢ local evaluation lẫn Brain
        sim (match theo canonical_hash), cả hai sharpe != None. Phục vụ calibrate ρ Spearman
        (local vs Brain). Mỗi canonical_hash lấy 1 local sharpe (eval đầu tiên có sharpe)."""
        session = self.session_factory()
        try:
            pairs: list[tuple[float, float]] = []
            links = (
                session.query(BrainSimLinkModel)
                .filter(BrainSimLinkModel.sharpe.isnot(None))
                .all()
            )
            for link in links:
                expr = (
                    session.query(ExpressionModel)
                    .filter_by(canonical_hash=link.canonical_hash)
                    .first()
                )
                if expr is None:
                    continue
                ev = (
                    session.query(EvaluationModel)
                    .filter(EvaluationModel.expression_id == expr.id)
                    .filter(EvaluationModel.sharpe.isnot(None))
                    .order_by(EvaluationModel.id)
                    .first()
                )
                if ev is None:
                    continue
                pairs.append((float(ev.sharpe), float(link.sharpe)))
            return pairs
        finally:
            session.close()

    def submit_ready_alphas(self, self_corr_max: float) -> list[SubmitReadyAlpha]:
        """Task 8: alpha THẬT SỰ sẵn sàng nộp Regular — `SimulationModel.status == 'passed'`
        + `failed_checks == []` + `self_corr` đã verify (< `self_corr_max`). `self_corr` KHÔNG
        có cột trên `SimulationModel` (chỉ WQ tính bất đồng bộ lúc submit thật) — tra theo
        `wq_alpha_id` trong `BrainSimLinkModel.self_corr`, nguồn duy nhất trong DB đã ghi
        self-corr Brain thật (`ClosedLoop.run` ghi qua `record_brain_sim` mỗi lần refine+sim).
        Alpha đã pass nhưng CHƯA từng có self-corr ghi lại (self_corr=None, vd nộp qua đường
        `research`/`submit` cũ trước khi cầu này tồn tại) KHÔNG được liệt ở đây — cần tự chạy
        `submit --dry-run` để verify trước khi nộp thật, tránh báo sai "sẵn sàng" khi chưa biết
        self-corr. `abs(self_corr)` (nhất quán `src/backtest/gates.py`): anti-correlation cũng
        đáng ngại như correlation dương. Loại alpha đã `SubmissionModel.status == 'submitted'`
        (đỡ báo lại cái đã nộp rồi). Sort sharpe giảm dần (ứng viên mạnh nhất lên đầu)."""
        session = self.session_factory()
        try:
            submitted_ids = {
                row[0]
                for row in session.query(SubmissionModel.alpha_id)
                .filter(SubmissionModel.status == "submitted")
                .all()
            }
            rows = (
                session.query(SimulationModel, AlphaModel)
                .join(AlphaModel, SimulationModel.alpha_id == AlphaModel.id)
                .filter(SimulationModel.status == "passed")
                .filter(SimulationModel.wq_alpha_id.isnot(None))
                # Tất định: dedup seen_wq_ids giữ row ĐẦU TIÊN gặp — sort sim mới nhất lên
                # trước để cùng một wq_alpha_id nhiều lần sim luôn lấy kết quả mới nhất.
                .order_by(SimulationModel.sim_at.desc())
                .all()
            )
            ready: list[SubmitReadyAlpha] = []
            seen_wq_ids: set[str] = set()
            for sim, alpha in rows:
                wq_id = sim.wq_alpha_id
                if wq_id in seen_wq_ids or sim.alpha_id in submitted_ids:
                    continue
                if sim.failed_checks is None:
                    # NULL = CHƯA TỪNG chạy is.checks thật (cột thêm bằng ALTER TABLE không
                    # DEFAULT — mọi row trước sub-project B là NULL), KHÁC HẲN '[]' = đã kiểm
                    # và 0 check fail. Không dám khẳng định "sẵn sàng" khi chưa verify —
                    # nhất quán với cách loại self_corr chưa verify (None) ở dưới.
                    continue
                try:
                    checks = json.loads(sim.failed_checks)
                except (json.JSONDecodeError, TypeError):
                    continue  # dữ liệu hỏng -> không dám khẳng định "sẵn sàng"
                if checks != []:
                    continue
                corr_row = (
                    session.query(BrainSimLinkModel.self_corr)
                    .filter(BrainSimLinkModel.wq_alpha_id == wq_id)
                    .filter(BrainSimLinkModel.self_corr.isnot(None))
                    .order_by(BrainSimLinkModel.created_at.desc())
                    .first()
                )
                if corr_row is None or corr_row[0] is None:
                    continue  # self-corr chưa verify -> không liệt là sẵn sàng
                self_corr = float(corr_row[0])
                if abs(self_corr) >= self_corr_max:
                    continue
                seen_wq_ids.add(wq_id)
                ready.append(
                    SubmitReadyAlpha(
                        wq_alpha_id=wq_id, expression=alpha.expression,
                        sharpe=sim.sharpe, self_corr=self_corr,
                    )
                )
            ready.sort(key=lambda r: r.sharpe if r.sharpe is not None else float("-inf"), reverse=True)
            return ready
        finally:
            session.close()
