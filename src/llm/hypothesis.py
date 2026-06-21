"""Sinh giả thuyết thị trường có cấu trúc 4 phần từ một hướng nghiên cứu (T2.3).

Bốn phần: quan sát, kiến thức nền, lý giải kinh tế, đặc tả triển khai. Đây là
bước "quyết định tạo ra cái gì" trước khi dịch sang biểu thức.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.llm.jsonutil import extract_json

_PARTS = ("observation", "background", "economic_rationale", "implementation_spec")


@dataclass
class Hypothesis:
    observation: str = ""
    background: str = ""
    economic_rationale: str = ""
    implementation_spec: str = ""
    fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, str]:
        return {p: getattr(self, p) for p in _PARTS}

    @classmethod
    def from_dict(cls, data: dict) -> "Hypothesis":
        if not isinstance(data, dict):
            return cls()
        return cls(**{p: str(data.get(p, "") or "") for p in _PARTS})


def ground_fields(llm_fields, palette_ids, min_k: int = 2) -> tuple[str, ...]:
    """Lọc field LLM nêu về tập có thật (palette_ids); thiếu < min_k thì bổ sung
    từ palette. Khử trùng lặp, giữ thứ tự ưu tiên (LLM-hợp-lệ trước, palette sau)."""
    palette = list(dict.fromkeys(p for p in (palette_ids or []) if isinstance(p, str)))
    allowed = set(palette)
    if isinstance(llm_fields, str):
        llm_fields = [llm_fields]
    elif not isinstance(llm_fields, (list, tuple)):
        llm_fields = []
    grounded: list[str] = []
    for f in llm_fields:
        if isinstance(f, str) and f in allowed and f not in grounded:
            grounded.append(f)
    if len(grounded) < min_k:
        for f in palette:
            if f not in grounded:
                grounded.append(f)
            if len(grounded) >= min_k:
                break
    return tuple(grounded)


SYSTEM_PROMPT = (
    "Bạn là nhà nghiên cứu alpha định lượng trên WorldQuant BRAIN. "
    "Sinh MỘT giả thuyết thị trường có cấu trúc, trả JSON đúng 4 khoá: "
    '{"observation": "...", "background": "...", "economic_rationale": "...", '
    '"implementation_spec": "..."}. '
    "observation = hiện tượng quan sát được; background = lý thuyết/trực giác tài chính nền; "
    "economic_rationale = vì sao tín hiệu tồn tại về mặt kinh tế; "
    "implementation_spec = gợi ý dữ liệu/operator/tham số (cửa sổ) để triển khai."
)


class HypothesisGenerator:
    def __init__(self, deepseek):
        self.deepseek = deepseek

    def generate(self, research_direction: str) -> Hypothesis:
        user = (
            f'Hướng nghiên cứu: "{research_direction}". '
            "Đề xuất một giả thuyết alpha mới, cụ thể, có thể kiểm chứng. Trả JSON 4 phần."
        )
        content = self.deepseek.complete(SYSTEM_PROMPT, user, json_mode=True, task="hypothesis")
        data = extract_json(content)
        if not isinstance(data, dict):
            logger.warning("Hypothesis: không parse được JSON, trả rỗng.")
            return Hypothesis()
        return Hypothesis.from_dict(data)
