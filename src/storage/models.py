"""SQLAlchemy models cho cache fields/operators và lưu alpha/simulation."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DataFieldModel(Base):
    __tablename__ = "data_fields"

    # Khóa kép: cùng field id nhưng khác tổ hợp scope = dòng khác nhau.
    id = Column(String, primary_key=True)
    region = Column(String, primary_key=True)
    universe = Column(String, primary_key=True)
    delay = Column(Integer, primary_key=True)
    description = Column(Text)
    type = Column(String)
    dataset_id = Column(String)
    cached_at = Column(DateTime, default=_utcnow)


class FetchStateModel(Base):
    """Trạng thái fetch theo từng tổ hợp scope (phục vụ cache-once + TTL)."""

    __tablename__ = "fetch_state"

    key = Column(String, primary_key=True)  # "data_fields:USA:TOP3000:1"
    entity = Column(String)  # "data_fields" | "operators"
    region = Column(String)
    universe = Column(String)
    delay = Column(Integer)
    total_count = Column(Integer)
    fetched_at = Column(DateTime)
    status = Column(String)  # "complete" | "partial"


class OperatorModel(Base):
    __tablename__ = "operators"

    name = Column(String, primary_key=True)
    category = Column(String)
    definition = Column(Text)
    description = Column(Text)
    arity = Column(Integer)
    cached_at = Column(DateTime, default=_utcnow)


class AlphaModel(Base):
    __tablename__ = "alphas"

    id = Column(String, primary_key=True)
    expression = Column(Text, nullable=False)
    source = Column(String)  # template/ga/llm/random
    hypothesis = Column(Text)  # JSON giả thuyết 4 phần (GĐ2)
    description = Column(Text)  # mô tả bằng lời (GĐ2)
    parent_id = Column(String, ForeignKey("alphas.id"), nullable=True)  # lineage tinh chỉnh
    created_at = Column(DateTime, default=_utcnow)


class SimulationModel(Base):
    __tablename__ = "simulations"

    id = Column(String, primary_key=True)
    alpha_id = Column(String, ForeignKey("alphas.id"))
    expr_hash = Column(String, index=True)  # hash biểu thức để cache sim (T1.15)
    wq_alpha_id = Column(String)  # id alpha trên nền tảng WQ (phục vụ submit/correlation)
    region = Column(String)
    universe = Column(String)
    sharpe = Column(Float)
    fitness = Column(Float)
    turnover = Column(Float)
    drawdown = Column(Float)
    margin = Column(Float)
    returns = Column(Float)
    score = Column(Float)
    status = Column(String)  # passed/failed/error
    raw_result = Column(Text)  # full JSON
    sim_at = Column(DateTime, default=_utcnow)


class FailureModel(Base):
    """Bộ nhớ thất bại: alpha bị loại + lý do, để lần sau tránh lặp lại (T2.13)."""

    __tablename__ = "failures"

    id = Column(String, primary_key=True)
    expression = Column(Text)
    category = Column(String)  # syntax | low_score | hypothesis_mismatch
    reason = Column(Text)
    source = Column(String)
    created_at = Column(DateTime, default=_utcnow)


class SubmissionModel(Base):
    __tablename__ = "submissions"

    id = Column(String, primary_key=True)
    alpha_id = Column(String, ForeignKey("alphas.id"))
    status = Column(String)  # submitted/rejected/error
    self_correlation = Column(Float)
    detail = Column(Text)
    submitted_at = Column(DateTime, default=_utcnow)
