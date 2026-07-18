"""Đếm số lần đã nộp 'Power Pool thuần' (tag PowerPoolSelected) trong khoảng thời gian — phục
vụ tự kiểm tra quota 10/tháng + 1/ngày trước khi gọi submit() (sub-project A, Task 4).
Nguồn tiêu chí: docs/wq_scraped_docs/docs/consultant-information/power-pool-alphas.md
(mục "Submission Quotas After 3 Months"). CHỈ đếm dựa trên cột `SubmissionModel.tags` đã lưu
qua SubmissionManager.set_properties() (sub-project C) — KHÔNG tự gắn tag ở đây."""

from __future__ import annotations

import json
from datetime import datetime

from src.storage.models import SubmissionModel

POWER_POOL_TAG = "PowerPoolSelected"


def count_pure_power_pool_submissions(session_factory, since: datetime) -> int:
    """Đếm submission `status == "submitted"` có tag `PowerPoolSelected`, kể từ `since`."""
    session = session_factory()
    try:
        rows = (
            session.query(SubmissionModel.tags)
            .filter(SubmissionModel.status == "submitted")
            .filter(SubmissionModel.submitted_at >= since)
            .all()
        )
    finally:
        session.close()

    count = 0
    for (tags_json,) in rows:
        if not tags_json:
            continue
        try:
            tags = json.loads(tags_json)
        except (ValueError, TypeError):
            continue
        if isinstance(tags, list) and POWER_POOL_TAG in tags:
            count += 1
    return count
