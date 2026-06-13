"""Dịch giả thuyết -> mô tả bằng lời -> biểu thức FASTEXPR (T2.4) + repair (T2.5).

Bắt buộc đi qua bước mô tả: trước hết sinh mô tả ngôn ngữ tự nhiên từ giả thuyết,
rồi mới dịch mô tả đó sang công thức. Sai cú pháp thì gửi lỗi để model tự sửa.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.llm.hypothesis import Hypothesis
from src.llm.jsonutil import extract_json

MAX_FIELDS_IN_PROMPT = 40
MAX_REPAIR_ATTEMPTS = 3

FEWSHOT_EXAMPLES = [
    "rank(ts_delta(close, 5))",
    "-rank(ts_zscore(volume, 20))",
    "group_neutralize(rank(returns), sector)",
    "rank(ts_corr(close, volume, 20))",
]


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

    # ----------------------------------------------------------- context
    def _symbol_context(self) -> str:
        operators = [o.name for o in self.operator_repo.load_cached() if getattr(o, "name", None)]
        fields = [f.id for f in self.field_repo.load_cached() if getattr(f, "id", None)]
        op_line = ", ".join(operators[:80]) or "rank, ts_delta, ts_mean, group_neutralize, ts_corr"
        field_line = ", ".join(fields[:MAX_FIELDS_IN_PROMPT]) or "close, open, high, low, volume, vwap, returns"
        examples = "\n".join(f"- {e}" for e in FEWSHOT_EXAMPLES)
        return (
            f"OPERATORS hợp lệ: {op_line}\n"
            f"FIELDS khả dụng: {field_line}\n"
            "GROUPS cho neutralize: market, sector, industry, subindustry\n"
            f"Ví dụ alpha hợp lệ:\n{examples}"
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
        data = extract_json(self.deepseek.complete(system, user, json_mode=True))
        if isinstance(data, dict) and data.get("description"):
            return str(data["description"])
        return hypothesis.implementation_spec or hypothesis.observation

    # ----------------------------------------------------------- step 2
    def _to_expression(self, description: str) -> str | None:
        system = (
            "Bạn là chuyên gia viết biểu thức FASTEXPR trên WorldQuant BRAIN.\n"
            f"{self._symbol_context()}\n"
            "Dịch MÔ TẢ thành MỘT biểu thức FASTEXPR dùng đúng operators/fields được liệt kê. "
            'Trả JSON {"expression": "..."}.'
        )
        user = f"MÔ TẢ: {description}\nViết biểu thức FASTEXPR."
        for attempt in range(MAX_REPAIR_ATTEMPTS):
            data = extract_json(self.deepseek.complete(system, user, json_mode=True))
            expr = data.get("expression") if isinstance(data, dict) else None
            if not expr:
                user = 'Trả ĐÚNG JSON {"expression": "..."}.'
                continue
            ok, reason = self.prefilter.check(expr)
            if ok:
                return expr
            logger.info("Translator expr lỗi (lần {}): {} — {}", attempt + 1, expr, reason)
            user = f'Biểu thức "{expr}" bị lỗi: {reason}. Sửa lại, trả JSON.'
        return None

    # ----------------------------------------------------------- public
    def translate(self, hypothesis: Hypothesis) -> AlphaCandidate | None:
        description = self._describe(hypothesis)
        expression = self._to_expression(description)
        if not expression:
            return None
        return AlphaCandidate(hypothesis=hypothesis, description=description, expression=expression)
