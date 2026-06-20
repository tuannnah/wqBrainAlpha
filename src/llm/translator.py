"""Dịch giả thuyết -> mô tả bằng lời -> biểu thức FASTEXPR (T2.4) + repair (T2.5).

Bắt buộc đi qua bước mô tả: trước hết sinh mô tả ngôn ngữ tự nhiên từ giả thuyết,
rồi mới dịch mô tả đó sang công thức. Sai cú pháp thì gửi lỗi để model tự sửa.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.llm import expr_synth
from src.llm.hypothesis import Hypothesis
from src.llm.jsonutil import extract_json


@dataclass
class AlphaCandidate:
    hypothesis: Hypothesis
    description: str
    expression: str
    parent_id: str | None = None


class AlphaTranslator:
    def __init__(self, deepseek, field_repo, operator_repo, prefilter):
        self.deepseek = deepseek
        self.field_repo = field_repo
        self.operator_repo = operator_repo
        self.prefilter = prefilter
        self.avoid_subtrees: list[str] = []
        self._scope: dict | None = None  # (region,universe,delay) để lọc fields đúng region (T6.4)

    def set_avoid_subtrees(self, canons) -> None:
        """Đặt danh sách canon subtree LLM nên tránh dùng lại (T3.6)."""
        self.avoid_subtrees = list(canons or [])

    def set_scope(self, region: str, universe: str, delay: int) -> None:
        """Giới hạn fields đưa vào prompt theo đúng tổ hợp scope (T6.4 đa region)."""
        self._scope = {"region": region, "universe": universe, "delay": delay}

    def _avoid_context(self) -> str:
        if not self.avoid_subtrees:
            return ""
        joined = ", ".join(self.avoid_subtrees)
        return (
            "TRÁNH lặp lại các bộ khung sau (đã phổ biến trong alpha tốt, "
            f"F=field, N=số): {joined}. Ưu tiên cấu trúc mới để giữ đa dạng.\n"
        )

    # ----------------------------------------------------------- step 1
    def _describe(self, hypothesis: Hypothesis) -> str:
        system = (
            "Bạn là chuyên gia thiết kế alpha. Từ giả thuyết, mô tả BẰNG LỜI cách hiện "
            "thực hoá tín hiệu (dữ liệu nào, biến đổi gì, vì sao) — CHƯA viết công thức. "
            'Trả JSON {"description": "..."}.'
        )
        h = hypothesis
        user = (
            f"Quan sát: {h.observation}\nNền tảng: {h.background}\n"
            f"Lý giải kinh tế: {h.economic_rationale}\nGợi ý triển khai: {h.implementation_spec}\n"
            "Mô tả cách hiện thực tín hiệu."
        )
        data = extract_json(self.deepseek.complete(system, user, json_mode=True, task="describe"))
        if isinstance(data, dict) and data.get("description"):
            return str(data["description"])
        return hypothesis.implementation_spec or hypothesis.observation

    # ----------------------------------------------------------- step 2
    def _to_expression(self, description: str, relevance_text: str = "") -> str | None:
        system = (
            "Bạn là chuyên gia viết biểu thức FASTEXPR trên WorldQuant BRAIN.\n"
            f"{expr_synth.build_symbol_context(self.field_repo, self.operator_repo, self.prefilter, self._scope, relevance_text or description)}\n"
            f"{self._avoid_context()}"
            f"{expr_synth.build_syntax_constraints(self.prefilter)}"
            "Dịch MÔ TẢ thành MỘT biểu thức FASTEXPR dùng đúng operators/fields được liệt kê. "
            'Trả JSON {"expression": "..."}.'
        )
        user = f"MÔ TẢ: {description}\nViết biểu thức FASTEXPR."
        return expr_synth.repair_to_expression(
            self.deepseek, self.prefilter, self.field_repo, self._scope, system, user, task="translate"
        )

    # ----------------------------------------------------------- public
    def translate(self, hypothesis: Hypothesis) -> AlphaCandidate | None:
        description = self._describe(hypothesis)
        # Gộp toàn bộ ngữ cảnh hướng/giả thuyết để chọn field liên quan cho prompt.
        relevance_text = " ".join(
            [
                hypothesis.observation,
                hypothesis.background,
                hypothesis.economic_rationale,
                hypothesis.implementation_spec,
                description,
            ]
        )
        expression = self._to_expression(description, relevance_text)
        if not expression:
            return None
        return AlphaCandidate(hypothesis=hypothesis, description=description, expression=expression)
