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

    def generate(self, research_direction: str, palette=None) -> Hypothesis:
        user = (
            f'Hướng nghiên cứu: "{research_direction}". '
            "Đề xuất một giả thuyết alpha mới, cụ thể, có thể kiểm chứng. Trả JSON 4 phần."
        )
        system = SYSTEM_PROMPT
        palette_ids: list[str] = []
        if palette:
            palette_ids = [getattr(f, "id", None) for f in palette if getattr(f, "id", None)]
            listing = "\n".join(
                f"- {getattr(f, 'id', '')}: {(getattr(f, 'description', '') or '')[:60]}"
                for f in palette if getattr(f, "id", None)
            )
            system = (
                SYSTEM_PROMPT
                + "\nFIELD CÓ THẬT (chỉ nêu ID lấy ĐÚNG từ danh sách này):\n" + listing
                + '\nTrả thêm khoá "fields" = danh sách ID field bạn dùng; '
                "implementation_spec phải nêu chính các field ID đó."
            )
        content = self.deepseek.complete(system, user, json_mode=True, task="hypothesis")
        data = extract_json(content)
        if not isinstance(data, dict):
            logger.warning("Hypothesis: không parse được JSON, trả rỗng.")
            return Hypothesis(fields=ground_fields(None, palette_ids)) if palette_ids else Hypothesis()
        h = Hypothesis.from_dict(data)
        if palette_ids:
            h.fields = ground_fields(data.get("fields"), palette_ids)
        return h
