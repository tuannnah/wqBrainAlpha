"""Di trú dữ liệu giữa hai engine (vd SQLite -> PostgreSQL), idempotent."""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from src.storage.db import init_db
from src.storage.models import (
    AlphaModel,
    DataFieldModel,
    FailureModel,
    FetchStateModel,
    InvalidFieldModel,
    OperatorModel,
    SimulationModel,
    SubmissionModel,
)

# Thứ tự tôn trọng khóa ngoại: bảng được tham chiếu đứng trước.
MIGRATION_ORDER = [
    DataFieldModel,
    FetchStateModel,
    OperatorModel,
    InvalidFieldModel,
    AlphaModel,        # simulations/submissions tham chiếu alphas
    SimulationModel,
    FailureModel,
    SubmissionModel,
]


def migrate_all(source_engine: Engine, dest_engine: Engine) -> dict[str, int]:
    """Copy mọi bảng models từ source sang dest. Trả {tên_bảng: số_rows}.

    Dùng merge theo khóa chính -> chạy lại an toàn (không nhân đôi). Bảng không
    tồn tại ở nguồn (DB cũ thiếu) được bỏ qua với count 0.
    """
    init_db(dest_engine)
    src_tables = set(inspect(source_engine).get_table_names())
    SrcSession = sessionmaker(bind=source_engine, future=True)
    DstSession = sessionmaker(bind=dest_engine, future=True)

    counts: dict[str, int] = {}
    src = SrcSession()
    dst = DstSession()
    try:
        for model in MIGRATION_ORDER:
            table = model.__tablename__
            if table not in src_tables:
                counts[table] = 0
                continue
            rows = src.query(model).all()
            for row in rows:
                data = {c.name: getattr(row, c.name) for c in model.__table__.columns}
                dst.merge(model(**data))
            dst.commit()
            counts[table] = len(rows)
    finally:
        src.close()
        dst.close()
    return counts
