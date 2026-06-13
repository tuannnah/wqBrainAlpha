"""Chấm nhất quán giả thuyết–mô tả–công thức bằng một lần gọi LLM phụ (T4.1).

Hỏi LLM hai điều: (a) mô tả có triển khai đúng giả thuyết không, (b) biểu thức có
phản ánh đúng mô tả không. Trả điểm ∈ [0,1] (cao = khớp). Dùng làm bộ lọc thứ hai
TRƯỚC sim (T4.2) và đóng góp vào số hạng điều chuẩn (T4.4). JSON lỗi -> điểm trung
lập 0.5 (không chặn pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.jsonutil import extract_json
from src.llm.translator import AlphaCandidate

SYSTEM_PROMPT = (
    "Bạn là giám khảo nhất quán alpha. Cho một giả thuyết, mô tả bằng lời và biểu "
    "thức FASTEXPR, chấm mức ĐỘ KHỚP: (a) mô tả có triển khai đúng giả thuyết, "
    "(b) biểu thức có phản ánh đúng mô tả. Ví dụ: tuyên bố về thanh khoản nhưng "
    "biểu thức không có thành phần volume/spread -> điểm thấp. "
    'Trả JSON {"score": <0..1>, "reason": "..."}.'
)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass
class AlignmentScore:
    value: float       # ∈ [0,1], cao = khớp tốt
    reason: str = ""


class AlignmentScorer:
    def __init__(self, deepseek):
        self.deepseek = deepseek

    def score(self, candidate: AlphaCandidate) -> AlignmentScore:
        h = candidate.hypothesis
        user = (
            f"Giả thuyết:\n"
            f"- Quan sát: {h.observation}\n"
            f"- Nền tảng: {h.background}\n"
            f"- Lý giải kinh tế: {h.economic_rationale}\n"
            f"- Triển khai: {h.implementation_spec}\n"
            f"Mô tả: {candidate.description}\n"
            f"Biểu thức: {candidate.expression}\n"
            "Chấm độ khớp giả thuyết–mô tả–biểu thức."
        )
        data = extract_json(self.deepseek.complete(SYSTEM_PROMPT, user, json_mode=True))
        if not isinstance(data, dict) or "score" not in data:
            return AlignmentScore(0.5, "không parse được điểm — coi trung lập")
        try:
            value = _clamp01(float(data["score"]))
        except (TypeError, ValueError):
            return AlignmentScore(0.5, "điểm không phải số — coi trung lập")
        return AlignmentScore(value, str(data.get("reason", "")))
