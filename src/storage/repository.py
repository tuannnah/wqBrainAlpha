"""Lưu alpha và kết quả simulation vào DB."""

from __future__ import annotations

import hashlib
import json
import uuid

from src.simulation.simulator import SimulationResult
from src.storage.models import (
    AlphaModel,
    FailureModel,
    InvalidFieldModel,
    SimulationModel,
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
