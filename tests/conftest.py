"""Cấu hình chung cho test suite.

Mục tiêu chính: KHÔNG để test ghi đè vào log production
(`logs/wq_alpha_<date>.log`). Loguru dùng handler toàn cục, nên nếu một test
gọi `main._setup_logging()` (qua lệnh Typer chẳng hạn) thì file sink dính cho
cả phiên → mọi logger.error của các test fixture (foo_bar, rank(a,b,c)...) đổ
vào log thật, gây nhiễu khó soi lỗi. Ta đặt biến môi trường WQ_NO_FILE_LOG
ngay từ đầu phiên để `_setup_logging` bỏ qua file sink, và chủ động gỡ mọi
file sink có thể đã được thêm trước đó.
"""

from __future__ import annotations

import os

import pytest
from loguru import logger


@pytest.fixture(scope="session", autouse=True)
def _no_production_log_during_tests():
    os.environ["WQ_NO_FILE_LOG"] = "1"
    # Gỡ sạch handler hiện có (kể cả file sink lỡ dính) rồi chỉ giữ stderr.
    import sys

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    yield
