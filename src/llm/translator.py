"""Dịch giả thuyết -> mô tả bằng lời -> biểu thức FASTEXPR (T2.4) + repair (T2.5).

Bắt buộc đi qua bước mô tả: trước hết sinh mô tả ngôn ngữ tự nhiên từ giả thuyết,
rồi mới dịch mô tả đó sang công thức. Sai cú pháp thì gửi lỗi để model tự sửa.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

from src.llm.hypothesis import Hypothesis
from src.llm.jsonutil import extract_json

MAX_FIELDS_IN_PROMPT = 40
MAX_REPAIR_ATTEMPTS = 3

# Ví dụ minh hoạ CÚ PHÁP, cố ý đa dạng cấu trúc và tránh các bộ khung kinh điển
# trùng Alpha101 (vd rank(ts_delta(close,N))) để LLM không neo vào mẫu dễ trùng.
FEWSHOT_EXAMPLES = [
    "ts_decay_linear(rank(ts_std_dev(returns, 20)), 5)",
    "group_neutralize(ts_zscore(vwap, 60), industry)",
    "rank(divide(ts_mean(volume, 10), ts_mean(volume, 60)))",
    "ts_rank(ts_corr(close, volume, 20), 120)",
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

    # ----------------------------------------------------------- context
    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def _relevant_fields(self, cached_fields, text: str) -> list[str]:
        """Xếp hạng fields theo độ liên quan với hypothesis/mô tả (text), rồi cắt
        MAX_FIELDS_IN_PROMPT. Đảm bảo field hướng nêu đích danh luôn vào prompt thay
        vì lấy 40 field đầu theo alphabet. Text rỗng -> giữ thứ tự gốc (tương thích)."""
        text_low = (text or "").lower()
        text_tokens = self._tokens(text_low)
        scored = []
        for idx, f in enumerate(cached_fields):
            fid = getattr(f, "id", None)
            if not fid:
                continue
            dataset = (getattr(f, "dataset_id", "") or "").lower()
            score = 0
            if fid.lower() in text_low:        # nêu đích danh field -> ưu tiên mạnh
                score += 100
            if dataset and dataset in text_low:  # nêu dataset (vd option9, earnings4)
                score += 20
            score += len(self._tokens(fid + " " + (getattr(f, "description", "") or "")) & text_tokens)
            scored.append((score, idx, fid))
        # score cao trước; hoà thì giữ thứ tự gốc (idx) cho ổn định/tương thích.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [fid for _, _, fid in scored[:MAX_FIELDS_IN_PROMPT]]

    def _symbol_context(self, text: str = "") -> str:
        operators = [o.name for o in self.operator_repo.load_cached() if getattr(o, "name", None)]
        cached_fields = self.field_repo.load_cached(**self._scope) if self._scope else self.field_repo.load_cached()
        fields = self._relevant_fields(cached_fields, text)
        op_line = ", ".join(operators[:80]) or "rank, ts_delta, ts_mean, group_neutralize, ts_corr"
        field_line = ", ".join(fields) or "close, open, high, low, volume, vwap, returns"
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
        data = extract_json(self.deepseek.complete(system, user, json_mode=True, task="describe"))
        if isinstance(data, dict) and data.get("description"):
            return str(data["description"])
        return hypothesis.implementation_spec or hypothesis.observation

    # ----------------------------------------------------------- step 2
    def _syntax_constraints(self) -> str:
        """Ràng buộc cú pháp suy ra từ pre-filter để biểu thức qua lọc ngay."""
        max_depth = getattr(self.prefilter, "max_depth", 6)
        max_nodes = getattr(self.prefilter, "max_nodes", 30)
        return (
            "RÀNG BUỘC bắt buộc để qua bộ lọc cú pháp:\n"
            f"- Độ sâu lồng nhau TỐI ĐA {max_depth}; tổng số node TỐI ĐA {max_nodes}. "
            "Ưu tiên biểu thức GỌN và NÔNG, tránh lồng quá nhiều tầng.\n"
            "- CHỈ dùng đối số theo VỊ TRÍ. TUYỆT ĐỐI không dùng đối số có tên kiểu "
            "key=value (vd viết winsorize(x, 3) chứ KHÔNG viết winsorize(x, std=3)).\n"
            "- Đối số chỉ là field/group đã liệt kê, biểu thức con, hoặc SỐ NGUYÊN.\n"
        )

    def _to_expression(self, description: str, relevance_text: str = "") -> str | None:
        system = (
            "Bạn là chuyên gia viết biểu thức FASTEXPR trên WorldQuant BRAIN.\n"
            f"{self._symbol_context(relevance_text or description)}\n"
            f"{self._avoid_context()}"
            f"{self._syntax_constraints()}"
            "Dịch MÔ TẢ thành MỘT biểu thức FASTEXPR dùng đúng operators/fields được liệt kê. "
            'Trả JSON {"expression": "..."}.'
        )
        user = f"MÔ TẢ: {description}\nViết biểu thức FASTEXPR."
        for attempt in range(MAX_REPAIR_ATTEMPTS):
            data = extract_json(self.deepseek.complete(system, user, json_mode=True, task="translate"))
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
