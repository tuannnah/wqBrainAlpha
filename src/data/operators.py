"""Lấy & cache operators từ WQ Brain (phục vụ validation và sinh alpha)."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session

from src.data.client import WQBrainClient
from src.storage.models import OperatorModel


class OperatorFetchError(RuntimeError):
    """Lỗi khi tải operators từ WorldQuant."""


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


def _split_top_level(s: str) -> list[str]:
    """Tách chuỗi theo dấu phẩy ở mức ngoài cùng — bỏ qua phẩy trong ngoặc tròn
    hoặc trong ngoặc kép (kể cả ngoặc kép cong “ ” mà WQ dùng)."""
    parts: list[str] = []
    depth = 0
    in_quote = False
    buf: list[str] = []
    for ch in s:
        if ch in "“”\"'":
            if ch == "“":
                in_quote = True
            elif ch == "”":
                in_quote = False
            else:  # ngoặc kép/đơn ASCII -> bật/tắt
                in_quote = not in_quote
            buf.append(ch)
            continue
        if not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append("".join(buf))
                buf = []
                continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def count_max_arity(definition: str) -> int:
    """Số tham số TỐI ĐA (gồm CẢ param có default `=`) trong chữ ký đầu tiên.

    WQ CHO truyền positional cả param có default: `ts_backfill(close, 120)`,
    `rank(x, 2)`, `winsorize(x, 4)` đều hợp lệ (default chỉ là giá trị mặc định khi
    bỏ trống, KHÔNG phải named-only). Bản cũ (`count_positional_arity`) bỏ param có
    `=` khỏi cap -> ts_backfill (`ts_backfill(x, lookback=d, k=1)`) bị cap=1 CHẶN OAN
    `ts_backfill(x, 22)` (biểu thức hợp lệ trên Brain). Cap đúng = TỔNG param của chữ
    ký đầu; PreFilter chỉ chặn khi THỪA hơn cap này (Brain là trọng tài cuối)."""
    if "(" not in definition or ")" not in definition:
        return 0
    # Một số operator có nhiều dạng chữ ký phân tách bởi 'or'/xuống dòng -> lấy dòng đầu.
    head = definition.splitlines()[0]
    if "(" not in head or ")" not in head:
        return 0
    inside = head[head.find("(") + 1 : head.rfind(")")]
    if not inside.strip():
        return 0
    return sum(1 for p in _split_top_level(inside) if p.strip())


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
        if resp.status_code >= 400:
            logger.error("GET /operators lỗi {}: {}", resp.status_code, resp.text[:500])
            raise OperatorFetchError(f"Không tải được operators (HTTP {resp.status_code}).")
        payload = resp.json()
        # /operators có thể trả list trực tiếp hoặc {"results": [...]}.
        results = payload.get("results", payload) if isinstance(payload, dict) else payload
        operators = [_parse_operator(r) for r in results]
        self._cache(operators)
        logger.info("Đã lấy {} operators", len(operators))
        return operators

    def cached_count(self) -> int:
        session: Session = self.session_factory()
        try:
            return session.query(OperatorModel).count()
        finally:
            session.close()

    def ensure(self, force: bool = False) -> tuple[list[Operator], bool]:
        """Trả (operators, đã_tải_mới). Dùng cache nếu có và không force."""
        if not force and self.cached_count() > 0:
            return self.load_cached(), False
        return self.fetch_all(), True

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
