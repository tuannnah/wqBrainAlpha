"""Sinh alpha có LLM (DeepSeek) hỗ trợ, kèm vòng lặp tự sửa khi syntax sai."""

from __future__ import annotations

from loguru import logger

from src.llm.jsonutil import extract_json as _extract_json

FEWSHOT_EXAMPLES = [
    "rank(ts_delta(close, 5))",
    "-rank(ts_zscore(volume, 20))",
    "group_neutralize(rank(returns), sector)",
    "rank(ts_mean(close, 5) - ts_mean(close, 20))",
    "rank(ts_corr(close, volume, 20))",
    "ts_rank(ts_delta(vwap, 10), 20)",
]

MAX_FIELDS_IN_PROMPT = 40
MAX_REPAIR_ATTEMPTS = 3

# Các hướng dataset ít người khai thác (correlation thấp trên nền tảng). Mỗi mục
# là một "mạch" nghiên cứu để LLM bám vào thay vì PV/fundamental kinh điển.
# Đồng bộ tinh thần với src/generation/novel_ideas.py (option/news/social/analyst/graph).
ALT_DATA_THEMES = [
    "biến động ngụ ý từ quyền chọn (implied volatility, IV-RV spread, skew put-call, "
    "term structure, put-call open interest)",
    "tin tức sự kiện (event sentiment, novelty/độ mới lạ, overreaction theo tin)",
    "tín hiệu mạng xã hội (buzz/social volume, attention-driven mispricing)",
    "điều chỉnh dự báo của analyst (net earnings revision, target vs recommendation divergence)",
    "đồ thị chuỗi cung ứng (customer return signal, competitor pagerank, lan truyền lead-lag)",
]

# Những mẫu công thức đã quá đông người dùng -> dễ trùng -> correlation cao -> bị loại.
CLICHE_PATTERNS = [
    "rank(ts_delta(close, N))",
    "rank(ebit / cap)",
    "rank(returns)",
    "momentum/reversal giá thuần dùng close/returns đơn lẻ",
]


class LLMAlphaGenerator:
    def __init__(self, deepseek, field_repo, operator_repo, prefilter):
        self.deepseek = deepseek
        self.field_repo = field_repo
        self.operator_repo = operator_repo
        self.prefilter = prefilter

    def build_system_prompt(self) -> str:
        operators = [o.name for o in self.operator_repo.load_cached() if o.name]
        fields = [f.id for f in self.field_repo.load_cached() if f.id][:MAX_FIELDS_IN_PROMPT]
        op_line = ", ".join(operators[:80]) if operators else "rank, ts_delta, ts_mean, group_neutralize, ts_corr, ts_zscore"
        field_line = ", ".join(fields) if fields else "close, open, high, low, volume, vwap, returns"
        examples = "\n".join(f"- {e}" for e in FEWSHOT_EXAMPLES)
        return (
            "Bạn là chuyên gia thiết kế Alpha trên WorldQuant BRAIN, viết biểu thức FASTEXPR.\n"
            "Cú pháp: hàm(đối_số, ...), toán tử + - * /, rank chuẩn hóa cross-sectional, "
            "tiền tố ts_ là chuỗi thời gian với tham số cửa sổ là số nguyên.\n"
            f"OPERATORS hợp lệ: {op_line}\n"
            f"FIELDS khả dụng: {field_line}\n"
            f"GROUPS cho neutralize: market, sector, industry, subindustry\n"
            "Ví dụ alpha hợp lệ:\n"
            f"{examples}\n"
            'Luôn trả về JSON đúng định dạng: {"expression": "...", "rationale": "..."}. '
            "Chỉ dùng operators và fields được liệt kê."
        )

    def _generate_one(self, idea: str) -> str | None:
        system = self.build_system_prompt()
        user = f'Ý tưởng alpha: "{idea}". Sinh MỘT biểu thức FASTEXPR. Trả JSON.'
        for attempt in range(MAX_REPAIR_ATTEMPTS):
            content = self.deepseek.complete(system, user, json_mode=True)
            data = _extract_json(content)
            expr = data.get("expression") if isinstance(data, dict) else None
            if not expr:
                user = 'Trả ĐÚNG JSON {"expression": "...", "rationale": "..."}.'
                continue
            ok, reason = self.prefilter.check(expr)
            if ok:
                return expr
            logger.info("LLM expr lỗi (lần {}): {} — {}", attempt + 1, expr, reason)
            user = f'Biểu thức "{expr}" bị lỗi: {reason}. Sửa lại và trả JSON.'
        return None

    def generate(self, idea: str, n: int = 5) -> list[str]:
        results: list[str] = []
        for _ in range(n):
            expr = self._generate_one(idea)
            if expr and expr not in results:
                results.append(expr)
        return results

    def build_ideas_system_prompt(self) -> str:
        """Prompt sinh HƯỚNG nghiên cứu, dẫn LLM sang dataset ít khai thác + cấu
        trúc lạ thay vì công thức PV/fundamental kinh điển (dễ trùng -> correlation
        cao -> bị loại). Xem feedback độ độc đáo alpha."""
        themes = "\n".join(f"- {t}" for t in ALT_DATA_THEMES)
        cliches = "; ".join(CLICHE_PATTERNS)
        return (
            "Bạn là nhà nghiên cứu alpha định lượng trên WorldQuant BRAIN, săn tín "
            "hiệu ĐỘC ĐÁO có correlation thấp với các factor đại trà.\n"
            "TRÁNH các công thức kinh điển vì cả nghìn người đã dùng nên trùng nặng "
            f"và bị loại do correlation cao: {cliches}.\n"
            "Thay vào đó, ưu tiên các MẠCH dữ liệu thay thế ít người khai thác và "
            "cấu trúc tổ hợp lạ (chuẩn hóa theo nhóm, phân kỳ hai kênh, lan truyền "
            "lead-lag, z-score đột biến...):\n"
            f"{themes}\n"
            "Mỗi ý tưởng là một câu ngắn nêu rõ NGUỒN DỮ LIỆU + hiện tượng kinh tế "
            "khai thác (không phải tên operator). Đa dạng nguồn, không lặp một mạch.\n"
            'Trả JSON đúng định dạng: {"ideas": ["...", "..."]}.'
        )

    def generate_ideas(self, n: int = 10) -> list[str]:
        system = self.build_ideas_system_prompt()
        user = (
            f"Đề xuất {n} hướng/ý tưởng alpha độc đáo, mỗi hướng dựa trên một mạch "
            "dữ liệu thay thế khác nhau. Tránh momentum/reversal giá thuần."
        )
        content = self.deepseek.complete(system, user, json_mode=True)
        data = _extract_json(content)
        if isinstance(data, dict):
            ideas = data.get("ideas", [])
        elif isinstance(data, list):
            ideas = data
        else:
            ideas = []
        return [str(i) for i in ideas][:n]
