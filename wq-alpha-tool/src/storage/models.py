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

    id = Column(String, primary_key=True)
    description = Column(Text)
    type = Column(String)
    dataset_id = Column(String)
    region = Column(String)
    universe = Column(String)
    delay = Column(Integer)
    cached_at = Column(DateTime, default=_utcnow)


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
    created_at = Column(DateTime, default=_utcnow)


class SimulationModel(Base):
    __tablename__ = "simulations"

    id = Column(String, primary_key=True)
    alpha_id = Column(String, ForeignKey("alphas.id"))
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


class SubmissionModel(Base):
    __tablename__ = "submissions"

    id = Column(String, primary_key=True)
    alpha_id = Column(String, ForeignKey("alphas.id"))
    status = Column(String)  # submitted/rejected/error
    self_correlation = Column(Float)
    detail = Column(Text)
    submitted_at = Column(DateTime, default=_utcnow)
