"""Các đối tượng dữ liệu bất biến dùng chung giữa các module nghiên cứu."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Scope:
    """Phạm vi (scope) mà một dataset/field có thể dùng."""

    instrument_type: str
    region: str
    delay: int
    universe: str


@dataclass(frozen=True)
class AlphaDraft:
    """Bản nháp Alpha do DeepSeek sinh, trước khi validation và simulation."""

    hypothesis: str
    rationale: str
    expression: str
    dataset_ids: List[str]
    field_ids: List[str]
    operator_names: List[str]
    settings: Dict[str, Any]
    parent_id: Optional[int] = None
    generation: int = 0
    improvement_direction: Optional[str] = None


@dataclass(frozen=True)
class SimulationResult:
    """Kết quả simulation đã chuẩn hóa từ WorldQuant BRAIN."""

    worldquant_alpha_id: Optional[str]
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
