"""Lưu alpha và kết quả simulation vào DB."""

from __future__ import annotations

import json
import uuid

from src.simulation.simulator import SimulationResult
from src.storage.models import AlphaModel, SimulationModel


class AlphaRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def save_alpha(self, expression: str, source: str = "manual") -> str:
        alpha_id = uuid.uuid4().hex
        session = self.session_factory()
        try:
            session.add(AlphaModel(id=alpha_id, expression=expression, source=source))
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
    ) -> str:
        """Lưu alpha (nếu chưa có) + simulation. Trả simulation id."""
        session = self.session_factory()
        try:
            alpha_id = uuid.uuid4().hex
            session.add(AlphaModel(id=alpha_id, expression=result.expression, source=source))

            sim_id = uuid.uuid4().hex
            session.add(
                SimulationModel(
                    id=sim_id,
                    alpha_id=alpha_id,
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
