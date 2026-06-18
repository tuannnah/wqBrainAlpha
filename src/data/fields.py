"""Lấy & cache data-fields và datasets từ WQ Brain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy.orm import Session

from config.settings import settings
from src.data.client import WQBrainClient
from src.storage.models import DataFieldModel, FetchStateModel


def _now() -> datetime:
    """UTC dạng naive (nhất quán với DateTime trong SQLite)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class FieldFetchError(RuntimeError):
    """Lỗi khi tải data-fields từ WorldQuant.

    status_code: mã HTTP nếu lỗi đến từ phản hồi server (401/403/429/4xx/5xx);
    None nếu lỗi không gắn với HTTP.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


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

    # ----------------------------------------------------------- public API
    def get_fields(
        self,
        region: str,
        universe: str,
        delay: int,
        force_reload: bool = False,
        instrument_type: str = "EQUITY",
    ) -> list[DataField]:
        """Load từ DB nếu đã cache (và còn hạn); ngược lại fetch từ API một lần."""
        if not force_reload and self._is_cached(region, universe, delay):
            logger.info("Load data fields từ DB (không gọi API) — {}", self._key(region, universe, delay))
            return self._load_from_db(region, universe, delay)
        logger.info("Fetch data fields từ WQ API — {}", self._key(region, universe, delay))
        return self._fetch_and_store(region, universe, delay, instrument_type)

    def ensure(
        self,
        region: str,
        universe: str,
        delay: int,
        instrument_type: str = "EQUITY",
        force: bool = False,
    ) -> tuple[list[DataField], bool]:
        """Trả (fields, đã_tải_mới). Wrapper quanh get_fields cho wizard/CLI."""
        was_cached = self._is_cached(region, universe, delay)
        fields = self.get_fields(region, universe, delay, force_reload=force, instrument_type=instrument_type)
        return fields, (force or not was_cached)

    def fetch_all(
        self,
        region: str,
        universe: str,
        delay: int,
        instrument_type: str = "EQUITY",
        page_size: int | None = None,
    ) -> list[DataField]:
        """Ép fetch từ API và ghi đè cache (giữ tương thích tên cũ)."""
        return self._fetch_and_store(region, universe, delay, instrument_type, page_size)

    def fetch_datasets(self) -> list[dict]:
        """GET /data-sets — danh sách dataset categories."""
        resp = self.client.get("/data-sets")
        resp.raise_for_status()
        return resp.json().get("results", [])

    # ------------------------------------------------------------- cache state
    def _key(self, region: str, universe: str, delay: int) -> str:
        return f"data_fields:{region}:{universe}:{delay}"

    def _is_cached(self, region: str, universe: str, delay: int) -> bool:
        session: Session = self.session_factory()
        try:
            state = session.get(FetchStateModel, self._key(region, universe, delay))
            if state is None or state.status != "complete" or state.fetched_at is None:
                return False
            age = _now() - state.fetched_at
            if age > timedelta(days=settings.cache_ttl_days):
                logger.info("Cache quá hạn ({} ngày) — sẽ tải lại", age.days)
                return False
            return True
        finally:
            session.close()

    def cached_count(self, region: str | None = None, universe: str | None = None,
                     delay: int | None = None) -> int:
        """Số field đã cache (lọc theo scope nếu truyền)."""
        session: Session = self.session_factory()
        try:
            query = session.query(DataFieldModel)
            if region is not None:
                query = query.filter(DataFieldModel.region == region)
            if universe is not None:
                query = query.filter(DataFieldModel.universe == universe)
            if delay is not None:
                query = query.filter(DataFieldModel.delay == delay)
            return query.count()
        finally:
            session.close()

    def get_state(self, region: str, universe: str, delay: int) -> FetchStateModel | None:
        session: Session = self.session_factory()
        try:
            return session.get(FetchStateModel, self._key(region, universe, delay))
        finally:
            session.close()

    def all_states(self) -> list[FetchStateModel]:
        session: Session = self.session_factory()
        try:
            return session.query(FetchStateModel).all()
        finally:
            session.close()

    # ------------------------------------------------------------- fetch/store
    def _fetch_and_store(
        self, region: str, universe: str, delay: int,
        instrument_type: str = "EQUITY", page_size: int | None = None,
    ) -> list[DataField]:
        fields = self._fetch_all_pages(region, universe, delay, instrument_type, page_size)
        self._replace_in_db(fields, region, universe, delay)
        self._update_state(region, universe, delay, len(fields))
        logger.success("Đã lưu {} fields vào DB ({}/{}/delay={})", len(fields), region, universe, delay)
        return fields

    def _fetch_all_pages(
        self, region: str, universe: str, delay: int,
        instrument_type: str = "EQUITY", page_size: int | None = None,
    ) -> list[DataField]:
        limit = page_size or self.PAGE_SIZE
        offset = 0
        fields: list[DataField] = []
        while True:
            resp = self.client.get(
                "/data-fields",
                params={
                    "instrumentType": instrument_type,
                    "region": region,
                    "delay": delay,
                    "universe": universe,
                    "limit": limit,
                    "offset": offset,
                },
            )
            if resp.status_code >= 400:
                logger.error("GET /data-fields lỗi {}: {}", resp.status_code, resp.text[:500])
                if resp.status_code == 429:
                    raise FieldFetchError(
                        "Bị giới hạn tần suất (429) sau nhiều lần thử. Hãy chờ vài phút rồi tải lại.",
                        status_code=429,
                    )
                raise FieldFetchError(
                    f"Không tải được data-fields (HTTP {resp.status_code}). "
                    "Kiểm tra region/universe/delay hợp lệ và tài khoản có quyền.",
                    status_code=resp.status_code,
                )
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
        return fields

    def _replace_in_db(self, fields: list[DataField], region: str, universe: str, delay: int) -> None:
        """Xóa cache cũ của đúng tổ hợp rồi ghi mới (replace, không append)."""
        session: Session = self.session_factory()
        try:
            session.query(DataFieldModel).filter_by(
                region=region, universe=universe, delay=delay
            ).delete()
            for f in fields:
                session.add(
                    DataFieldModel(
                        id=f.id,
                        region=region,
                        universe=universe,
                        delay=delay,
                        description=f.description,
                        type=f.type,
                        dataset_id=f.dataset_id,
                        cached_at=_now(),
                    )
                )
            session.commit()
        finally:
            session.close()

    def _update_state(self, region: str, universe: str, delay: int, count: int) -> None:
        session: Session = self.session_factory()
        try:
            key = self._key(region, universe, delay)
            state = session.get(FetchStateModel, key) or FetchStateModel(key=key)
            state.entity = "data_fields"
            state.region, state.universe, state.delay = region, universe, delay
            state.total_count = count
            state.fetched_at = _now()
            state.status = "complete"
            session.merge(state)
            session.commit()
        finally:
            session.close()

    def mark_no_access(self, region: str, universe: str, delay: int) -> None:
        """Đánh dấu scope tài khoản không truy cập được (resume sẽ bỏ qua nhanh)."""
        session: Session = self.session_factory()
        try:
            key = self._key(region, universe, delay)
            state = session.get(FetchStateModel, key) or FetchStateModel(key=key)
            state.entity = "data_fields"
            state.region, state.universe, state.delay = region, universe, delay
            state.fetched_at = _now()
            state.status = "no_access"
            session.merge(state)
            session.commit()
        finally:
            session.close()

    # ---------------------------------------------------------------- loaders
    def _load_from_db(self, region: str, universe: str, delay: int) -> list[DataField]:
        return self._rows_to_fields(
            lambda q: q.filter_by(region=region, universe=universe, delay=delay)
        )

    def load_cached(
        self, region: str | None = None, universe: str | None = None, delay: int | None = None
    ) -> list[DataField]:
        """Load fields đã cache; lọc theo scope nếu truyền (đa region — T6.4).
        Không truyền scope -> trả tất cả (tương thích ngược)."""
        def refine(q):
            if region is not None:
                q = q.filter(DataFieldModel.region == region)
            if universe is not None:
                q = q.filter(DataFieldModel.universe == universe)
            if delay is not None:
                q = q.filter(DataFieldModel.delay == delay)
            return q
        return self._rows_to_fields(refine)

    def _rows_to_fields(self, refine) -> list[DataField]:
        session: Session = self.session_factory()
        try:
            rows = refine(session.query(DataFieldModel)).all()
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
