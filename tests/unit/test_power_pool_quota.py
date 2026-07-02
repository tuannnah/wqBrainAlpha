"""Test đếm quota Power Pool thuần (sub-project A, Task 4) — đọc cột tags đã lưu qua
SubmissionManager.set_properties() (sub-project C)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import create_engine

from src.storage.db import init_db, make_session_factory
from src.storage.models import SubmissionModel
from src.submission.power_pool_quota import count_pure_power_pool_submissions


def _engine():
    return create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})


def test_dem_dung_so_lan_nop_power_pool_thuan_trong_khoang():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    now = datetime.utcnow()
    session.add(SubmissionModel(
        id="s1", alpha_id="WQ1", status="submitted",
        tags=json.dumps(["PowerPoolSelected"]), submitted_at=now,
    ))
    session.add(SubmissionModel(
        id="s2", alpha_id="WQ2", status="submitted",
        tags=json.dumps(["khac"]), submitted_at=now,
    ))  # không có tag Power Pool -> không đếm
    session.add(SubmissionModel(
        id="s3", alpha_id="WQ3", status="rejected",
        tags=json.dumps(["PowerPoolSelected"]), submitted_at=now,
    ))  # bị reject, không phải đã nộp thành công -> không đếm
    session.add(SubmissionModel(
        id="s4", alpha_id="WQ4", status="submitted",
        tags=json.dumps(["PowerPoolSelected"]), submitted_at=now - timedelta(days=40),
    ))  # ngoài khoảng thời gian -> không đếm
    session.commit()
    session.close()

    count = count_pure_power_pool_submissions(sf, since=now - timedelta(days=1))
    assert count == 1


def test_dem_0_khi_khong_co_submission_nao():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    count = count_pure_power_pool_submissions(sf, since=datetime.utcnow() - timedelta(days=1))
    assert count == 0
