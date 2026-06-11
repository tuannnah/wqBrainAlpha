"""Lấy & cache data-fields và datasets từ WQ Brain."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session

from src.data.client import WQBrainClient
from src.storage.models import DataFieldModel


@dataclass
class DataField:
    id: str
    description: str
    type: str  # MATRIX / VECTOR / GROUP
    dataset_id: str
    region: str
    delay: int
    universe: str


def _parse_field(raw: dict, region: str, universe: str, delay: int) -> DataField:
    dataset = raw.get("dataset") or {}
    dataset_id = dataset.get("id", "") if isinstance(dataset, dict) else str(dataset)
    return DataField(
        id=raw.get("id", ""),
        description=raw.get("description", ""),
        type=raw.get("type", ""),
        dataset_id=dataset_id,
        region=raw.get("region", region),
        delay=raw.get("delay", delay),
        universe=raw.get("universe", universe),
    )


class FieldRepository:
    PAGE_SIZE = 50

    def __init__(self, client: WQBrainClient, session_factory):
        self.client = client
        self.session_factory = session_factory

    def fetch_all(
        self, region: str, universe: str, delay: int, page_size: int | None = None
    ) -> list[DataField]:
        """Phân trang qua offset/limit, cache toàn bộ vào bảng data_fields."""
        limit = page_size or self.PAGE_SIZE
        offset = 0
        fields: list[DataField] = []

        while True:
            resp = self.client.get(
                "/data-fields",
                params={
                    "region": region,
                    "delay": delay,
                    "universe": universe,
                    "limit": limit,
                    "offset": offset,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            if not results:
                break

            for raw in results:
                fields.append(_parse_field(raw, region, universe, delay))

            offset += limit
            total = payload.get("count")
            if total is not None and offset >= total:
                break

        self._cache(fields)
        logger.info("Đã lấy {} data-fields ({}/{}/delay={})", len(fields), region, universe, delay)
        return fields

    def fetch_datasets(self) -> list[dict]:
        """GET /data-sets — danh sách dataset categories."""
        resp = self.client.get("/data-sets")
        resp.raise_for_status()
        return resp.json().get("results", [])

    def _cache(self, fields: list[DataField]) -> None:
        session: Session = self.session_factory()
        try:
            for f in fields:
                session.merge(
                    DataFieldModel(
                        id=f.id,
                        description=f.description,
                        type=f.type,
                        dataset_id=f.dataset_id,
                        region=f.region,
                        universe=f.universe,
                        delay=f.delay,
                    )
                )
            session.commit()
        finally:
            session.close()

    def load_cached(self) -> list[DataField]:
        session: Session = self.session_factory()
        try:
            rows = session.query(DataFieldModel).all()
            return [
                DataField(
                    id=r.id,
                    description=r.description or "",
                    type=r.type or "",
                    dataset_id=r.dataset_id or "",
                    region=r.region or "",
                    delay=r.delay or 0,
                    universe=r.universe or "",
                )
                for r in rows
            ]
        finally:
            session.close()
