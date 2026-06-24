"""Loader: AlphaModel + SimulationModel (+ SubmissionModel) DB hiện có -> BrainRecord.

KHÔNG có bảng brain_record riêng (B11 master spec) vì Phase 5 (mở rộng models.py) chưa
chạy. Phase 4.5 đọc dữ liệu simulation lịch sử đã có sẵn trong wq_alpha_*.db hiện tại.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.storage.models import AlphaModel, SimulationModel, SubmissionModel

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass(frozen=True, slots=True)
class BrainRecord:
    """Một alpha đã mô phỏng THẬT trên Brain, dùng làm ground-truth cho calibration."""

    expr_string: str
    brain_sharpe: float | None
    brain_fitness: float | None
    brain_turnover: float | None
    brain_self_corr: float | None  # None nếu alpha chưa từng submit


def load_brain_records(
    session_factory: Callable[[], Session], limit: int | None = None
) -> list[BrainRecord]:
    """Đọc alpha đã sim (status != 'error', sharpe không NULL), lấy bản sim MỚI NHẤT
    mỗi alpha_id, LEFT JOIN self_correlation từ submission (None nếu chưa submit).

    ⚠️ PRECONDITION CALIBRATION HỢP LỆ: hàm này KHÔNG lọc theo config sim (neutralization/
    decay/truncation) vì SimulationModel hiện không lưu các knob đó thành cột riêng. ρ chỉ có
    nghĩa khi MỌI record được sim với CÙNG config mà `make_local_scorer` re-score (ground-truth
    Phase 4.5: NONE/decay0/trunc0/delay1). Trỏ vào `wq_alpha_*.db` thường (sim chạy SECTOR/
    decay/truncation lẫn lộn) sẽ cho ρ VÔ NGHĨA dù vẫn tính ra số. Dùng DB ground-truth chuyên
    dụng. (Follow-up: thêm cột config_key vào SimulationModel ở Phase 5 rồi lọc tại đây.)"""
    session = session_factory()
    try:
        rows = (
            session.query(AlphaModel, SimulationModel)
            .join(SimulationModel, SimulationModel.alpha_id == AlphaModel.id)
            .filter(SimulationModel.status != "error", SimulationModel.sharpe.isnot(None))
            .order_by(AlphaModel.id, SimulationModel.sim_at.desc())
            .all()
        )

        latest_by_alpha: dict[str, tuple[AlphaModel, SimulationModel]] = {}
        for alpha, sim in rows:
            if alpha.id not in latest_by_alpha:
                latest_by_alpha[alpha.id] = (alpha, sim)  # dòng đầu = mới nhất (order_by desc)

        submissions = {
            row.alpha_id: row.self_correlation
            for row in session.query(SubmissionModel).all()
        }

        records = [
            # Thuộc tính instance ORM lúc chạy là scalar; mypy thấy Column[T] (SQLAlchemy
            # classic chưa typed) nên báo arg-type sai — bỏ qua đúng dòng đọc giá trị.
            BrainRecord(
                expr_string=alpha.expression,  # type: ignore[arg-type]
                brain_sharpe=sim.sharpe,  # type: ignore[arg-type]
                brain_fitness=sim.fitness,  # type: ignore[arg-type]
                brain_turnover=sim.turnover,  # type: ignore[arg-type]
                brain_self_corr=submissions.get(alpha.id),  # type: ignore[arg-type]
            )
            for alpha, sim in latest_by_alpha.values()
        ]
        return records[:limit] if limit is not None else records
    finally:
        session.close()
