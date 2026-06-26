"""SQLAlchemy models cho cache fields/operators và lưu alpha/simulation."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
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


class InvalidFieldModel(Base):
    """Field WQ liệt kê trong /data-fields nhưng simulator từ chối ('Invalid data
    field'). Blacklist tự học để loại khỏi nguồn sinh, khỏi phí quota (vùng chết)."""

    __tablename__ = "invalid_fields"

    field_id = Column(String, primary_key=True)
    region = Column(String)
    universe = Column(String)
    reason = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


class SubmissionModel(Base):
    __tablename__ = "submissions"

    id = Column(String, primary_key=True)
    alpha_id = Column(String, ForeignKey("alphas.id"))
    status = Column(String)  # submitted/rejected/error
    self_correlation = Column(Float)
    detail = Column(Text)
    submitted_at = Column(DateTime, default=_utcnow)


class ExpressionModel(Base):
    """Biểu thức canonical đã từng được đánh giá (de-dup theo canonical_hash, Phase 1
    CanonicalHasher). Một expression có thể có nhiều EvaluationModel (config/window khác
    nhau)."""

    __tablename__ = "expressions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_hash = Column(String, nullable=False, unique=True, index=True)
    expr_string = Column(Text, nullable=False)
    depth = Column(Integer, nullable=False)
    complexity = Column(Integer, nullable=False)
    fields_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=_utcnow)


class EvaluationModel(Base):
    """Một lần backtest cụ thể (expression + PortfolioConfig + cửa sổ data). Lưu CẢ pass
    và fail (B11: avoid-list cần biết alpha nào đã thử và vì sao fail) + seed (R8)."""

    __tablename__ = "evaluations"
    __table_args__ = (
        UniqueConstraint(
            "expression_id", "config_json", "data_window",
            name="uq_evaluation_expr_config_window",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    expression_id = Column(Integer, ForeignKey("expressions.id"), nullable=False, index=True)
    config_json = Column(Text, nullable=False)
    data_window = Column(String, nullable=False)
    sharpe = Column(Float)
    annual_return = Column(Float)
    turnover = Column(Float)
    max_drawdown = Column(Float)
    fitness = Column(Float)
    weight_concentration = Column(Float)
    per_year_json = Column(Text)
    self_corr_max = Column(Float)
    status = Column(String, nullable=False, index=True)
    fail_reasons = Column(Text)
    seed = Column(Integer)
    created_at = Column(DateTime, default=_utcnow)


Index("idx_eval_sharpe", EvaluationModel.sharpe)


class PoolPnlModel(Base):
    """PnL vector hằng ngày của 1 evaluation PASSED — pool self-correlation cục bộ (Phase
    6). Khóa chính = evaluation_id (1 alpha pass góp 1 vector PnL vào pool)."""

    __tablename__ = "pool_pnl"

    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), primary_key=True)
    dates_blob = Column(LargeBinary, nullable=False)
    pnl_blob = Column(LargeBinary, nullable=False)


class DeadFieldModel(Base):
    """Field GP/LLM đề xuất bị coi là 'chết' theo nghĩa MiniBrain (khác InvalidFieldModel
    của luồng Brain-sim cũ — đây dùng để chặn GP đề xuất lại field đã biết vô dụng/sai khi
    chạy local, không phải field bị Brain API từ chối)."""

    __tablename__ = "dead_fields_minibrain"

    name = Column(String, primary_key=True)
    reason = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


class BrainSimLinkModel(Base):
    """Cầu liên kết một expression MiniBrain (theo ``canonical_hash``) với kết quả SIM THẬT
    trên WorldQuant Brain. Tách khỏi ``AlphaModel``/``SimulationModel`` (luồng LLM cũ) — cầu
    này keyed theo ``canonical_hash`` là danh tính chung của expression MiniBrain, phục vụ
    feedback vòng kín (so local↔Brain, decorrelate tầng 2 bằng ``self_corr`` Brain thật)."""

    __tablename__ = "brain_sim_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_hash = Column(String, nullable=False, index=True)
    expr_string = Column(Text, nullable=False)
    wq_alpha_id = Column(String)
    region = Column(String)
    universe = Column(String)
    sharpe = Column(Float)
    fitness = Column(Float)
    turnover = Column(Float)
    self_corr = Column(Float)
    status = Column(String, nullable=False)
    raw_json = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


class BrainRecordModel(Base):
    """Ground truth Brain-sim cho CalibrationHarness (Phase 4.5): expression + metrics thật
    từ Brain, để so Spearman ρ với metrics local."""

    __tablename__ = "brain_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    expr_string = Column(Text, nullable=False)
    brain_sharpe = Column(Float)
    brain_fitness = Column(Float)
    brain_turnover = Column(Float)
    brain_self_corr = Column(Float)
    submitted = Column(Integer)  # 0/1
    created_at = Column(DateTime, default=_utcnow)
