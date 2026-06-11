"""Lấy & cache operators từ WQ Brain (phục vụ validation và sinh alpha)."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session

from src.data.client import WQBrainClient
from src.storage.models import OperatorModel


@dataclass
class Operator:
    name: str
    category: str
    definition: str
    description: str
    arity: int


def _count_arity(definition: str) -> int:
    """Ước lượng số tham số từ chữ ký dạng `name(x, y, z)`."""
    if "(" not in definition or ")" not in definition:
        return 0
    inside = definition[definition.find("(") + 1 : definition.rfind(")")].strip()
    if not inside:
        return 0
    return len([p for p in inside.split(",") if p.strip()])


def _parse_operator(raw: dict) -> Operator:
    definition = raw.get("definition", "") or raw.get("name", "")
    return Operator(
        name=raw.get("name", ""),
        category=raw.get("category", ""),
        definition=definition,
        description=raw.get("description", ""),
        arity=_count_arity(definition),
    )


class OperatorRepository:
    def __init__(self, client: WQBrainClient, session_factory):
        self.client = client
        self.session_factory = session_factory

    def fetch_all(self) -> list[Operator]:
        resp = self.client.get("/operators")
        resp.raise_for_status()
        payload = resp.json()
        # /operators có thể trả list trực tiếp hoặc {"results": [...]}.
        results = payload.get("results", payload) if isinstance(payload, dict) else payload
        operators = [_parse_operator(r) for r in results]
        self._cache(operators)
        logger.info("Đã lấy {} operators", len(operators))
        return operators

    def _cache(self, operators: list[Operator]) -> None:
        session: Session = self.session_factory()
        try:
            for op in operators:
                session.merge(
                    OperatorModel(
                        name=op.name,
                        category=op.category,
                        definition=op.definition,
                        description=op.description,
                        arity=op.arity,
                    )
                )
            session.commit()
        finally:
            session.close()

    def load_cached(self) -> list[Operator]:
        session: Session = self.session_factory()
        try:
            rows = session.query(OperatorModel).all()
            return [
                Operator(
                    name=r.name,
                    category=r.category or "",
                    definition=r.definition or "",
                    description=r.description or "",
                    arity=r.arity or 0,
                )
                for r in rows
            ]
        finally:
            session.close()
