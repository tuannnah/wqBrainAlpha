"""Di trú dữ liệu giữa hai engine (vd SQLite -> PostgreSQL), idempotent."""

from __future__ import annotations

import os

from sqlalchemy import inspect
from sqlalchemy.engine import Engine, make_url
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


def _same_database(url_a: str, url_b: str) -> bool:
    """So sánh ngữ nghĩa hai URL DB, tránh nhận nhầm khi chỉ khác chữ viết.

    Với SQLite có file (khác `:memory:`): chuẩn hóa đường dẫn tuyệt đối rồi so
    sánh (vd `sqlite:///wq_alpha.db` và `sqlite:///./wq_alpha.db` là cùng 1
    file). Mỗi DB `:memory:` được coi là độc lập (không bao giờ trùng), để
    không chặn nhầm các trường hợp test/hợp lệ dùng SQLite in-memory.

    Với các backend khác (vd PostgreSQL): so sánh theo (backend, user, host,
    port, database) đã parse từ URL.
    """
    a = make_url(url_a)
    b = make_url(url_b)

    a_is_sqlite = a.get_backend_name() == "sqlite"
    b_is_sqlite = b.get_backend_name() == "sqlite"
    a_is_memory = a_is_sqlite and (not a.database or a.database == ":memory:")
    b_is_memory = b_is_sqlite and (not b.database or b.database == ":memory:")
    if a_is_memory or b_is_memory:
        # Mỗi DB in-memory là độc lập, không bao giờ coi là trùng nhau.
        return False
    if a_is_sqlite and b_is_sqlite:
        return os.path.abspath(a.database) == os.path.abspath(b.database)

    return (
        a.get_backend_name() == b.get_backend_name()
        and a.username == b.username
        and a.host == b.host
        and a.port == b.port
        and a.database == b.database
    )


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
