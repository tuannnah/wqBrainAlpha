"""Sinh alpha có LLM (DeepSeek) hỗ trợ, kèm vòng lặp tự sửa khi syntax sai."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
import unicodedata

from loguru import logger

from src.generation.ast_utils import iter_leaves, parse_expression
from src.llm import expr_synth
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
MAX_IDEA_ATTEMPTS = 3
MAX_IDEA_DATASET_LINES = 8
MAX_IDEA_FIELDS_PER_DATASET = 4

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


ALT_DATA_FALLBACK_IDEAS = [
    "Option implied volatility skew divergence between put-call demand and realized volatility.",
    "News event novelty and sentiment reversal after unusually intense coverage.",
    "Social attention shock with delayed sector-neutral mean reversion.",
    "Analyst net earnings revision surprise with target-price recommendation divergence.",
    "Supply-chain customer and competitor graph lead-lag signal across related companies.",
]

ALT_DATA_KEYWORDS = (
    "analyst",
    "buzz",
    "call",
    "competitor",
    "customer",
    "event",
    "estimate",
    "graph",
    "implied",
    "iv",
    "news",
    "novelty",
    "open interest",
    "option",
    "put",
    "recommendation",
    "revision",
    "sentiment",
    "skew",
    "social",
    "supply",
    "target",
)

PRICE_VOLUME_FIELDS = {
    "adv20",
    "cap",
    "close",
    "high",
    "low",
    "open",
    "returns",
    "volume",
    "vwap",
}

CLICHE_IDEA_TERMS = (
    "bollinger",
    "close",
    "correlation",
    "dao chieu",
    "dong luong",
    "gia",
    "giam qua",
    "high",
    "khoi luong",
    "loi suat",
    "long",
    "low",
    "moving average",
    "momentum",
    "mua",
    "open",
    "price",
    "rank",
    "return",
    "reversal",
    "rsi",
    "short",
    "technical",
    "tuong quan",
    "volume",
    "vwap",
)


# Metric LLM tự BỊA nhét vào text hướng (vd "(sharpe=2.1, fitness=0.92)"). Đây là
# số bịa, KHÔNG phải đo thật — chỉ WQ-sim mới định nghĩa chất lượng. Tước sạch để
# không nhiễm xuống downstream (tránh tin tưởng nhầm số ảo).
_FAKE_METRIC_PAREN_RE = re.compile(
    r"\s*\(\s*(?:sharpe|fitness|turnover|returns|ir)\b[^)]*\)", re.IGNORECASE
)
_FAKE_METRIC_BARE_RE = re.compile(
    r"\s*\b(?:sharpe|fitness|turnover|returns|ir)\s*[=:]\s*[-+]?\d+(?:\.\d+)?", re.IGNORECASE
)


def _strip_fabricated_metrics(text: str) -> str:
    """Bỏ mọi chú thích metric LLM tự bịa khỏi text ý tưởng, giữ phần nội dung."""
    text = _FAKE_METRIC_PAREN_RE.sub("", text)
    text = _FAKE_METRIC_BARE_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _ascii_lower(text: str) -> str:
    text = text.replace("\u0111", "d").replace("\u0110", "d")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def _field_attr(field, name: str) -> str:
    return str(getattr(field, name, "") or "")


def _has_alt_data_keyword(text: str) -> bool:
    low = _ascii_lower(text)
    return any(keyword in low for keyword in ALT_DATA_KEYWORDS)


def _is_pure_price_volume_field(field) -> bool:
    fid = _ascii_lower(_field_attr(field, "id"))
    combined = f"{fid} {_field_attr(field, 'description')} {_field_attr(field, 'dataset_id')}"
    if _has_alt_data_keyword(combined):
        return False
    return fid in PRICE_VOLUME_FIELDS or fid.startswith("adv")


def _is_cliche_idea(idea: str) -> bool:
    low = _ascii_lower(idea)
    if not low.strip():
        return True
    if _has_alt_data_keyword(low):
        return False
    score = sum(1 for term in CLICHE_IDEA_TERMS if term in low)
    if re.search(r"\b\d+\s*(d|day|days|ngay|m|month|months|thang)\b", low):
        score += 1
    return score >= 2


def _common_fields(exprs, top: int = 6) -> list[str]:
    """Field xuất hiện nhiều nhất trong tập biểu thức (đếm theo số biểu thức chứa)."""
    counter: Counter[str] = Counter()
    for expr in exprs:
        try:
            tree = parse_expression(expr)
        except (ValueError, TypeError):
            continue
        counter.update({lf.value for lf in iter_leaves(tree) if isinstance(lf.value, str)})
    return [f for f, _ in counter.most_common(top)]


def build_feedback_prompt(top_alphas, weak_fields) -> str:
    """Đoạn ngữ cảnh phản hồi từ DB cho prompt sinh hướng (exploit + tránh vùng chết).

    top_alphas: [(expr, sharpe, fitness)] tốt nhất đã sim -> đề xuất biến thể.
    weak_fields: field liên tục cho kết quả yếu -> tránh."""
    parts: list[str] = []
    if top_alphas:
        lines = "\n".join(
            f"- {e}  (sharpe={s:.2f}, fitness={f:.2f})" for e, s, f in top_alphas
        )
        parts.append(
            "TÍN HIỆU ĐÃ MÔ PHỎNG TỐT NHẤT — hãy đề xuất hướng BIẾN THỂ/MỞ RỘNG các "
            "họ này (đặc biệt để nâng fitness), thay vì làm lại từ đầu:\n" + lines
        )
    if weak_fields:
        parts.append(
            "Các field liên tục cho kết quả YẾU, TRÁNH dựa vào: " + ", ".join(weak_fields)
        )
    return "\n".join(parts)


class LLMAlphaGenerator:
    def __init__(self, deepseek, field_repo, operator_repo, prefilter, repo=None, blacklist=None):
        self.deepseek = deepseek
        self.field_repo = field_repo
        self.operator_repo = operator_repo
        self.prefilter = prefilter
        self.repo = repo  # AlphaRepository để lấy phản hồi (top alpha / failures); None -> không feedback
        # Field WQ đã từ chối (chết/event) — cấm LLM nêu lại để khỏi tốn lượt sim.
        self.blacklist = set(blacklist or ())

    def _feedback_context(self) -> str:
        """Ngữ cảnh phản hồi từ DB: top alpha để khai thác + field yếu để tránh."""
        if self.repo is None:
            return ""
        top = self.repo.top_simulated(5)
        fails = [
            f.expression for f in self.repo.recent_failures(200)
            if f.category in ("low_score", "sim_error") and f.expression
        ]
        weak = _common_fields(fails)
        # Không liệt field đang dùng trong các top performer vào danh sách "tránh".
        top_fields = set(_common_fields([e for e, _, _ in top], top=20))
        weak = [w for w in weak if w not in top_fields]
        return build_feedback_prompt(top, weak)

    def build_system_prompt(self, relevance_text: str = "") -> str:
        context = expr_synth.build_symbol_context(
            self.field_repo, self.operator_repo, self.prefilter, None, relevance_text
        )
        constraints = expr_synth.build_syntax_constraints(self.prefilter)
        return (
            "Bạn là chuyên gia thiết kế Alpha trên WorldQuant BRAIN, viết biểu thức FASTEXPR.\n"
            "Cú pháp: hàm(đối_số, ...), toán tử + - * /, rank chuẩn hóa cross-sectional, "
            "tiền tố ts_ là chuỗi thời gian với tham số cửa sổ là số nguyên.\n"
            f"{context}\n{constraints}{self._feedback_context()}"
            'Luôn trả về JSON đúng định dạng: {"expression": "...", "rationale": "..."}. '
            "Chỉ dùng operators và fields được liệt kê."
        )

    def _generate_one(self, idea: str) -> str | None:
        system = self.build_system_prompt(idea)
        user = f'Ý tưởng alpha: "{idea}". Sinh MỘT biểu thức FASTEXPR. Trả JSON.'
        return expr_synth.repair_to_expression(
            self.deepseek, self.prefilter, self.field_repo, None, system, user, task=None
        )

    def generate(self, idea: str, n: int = 5) -> list[str]:
        results: list[str] = []
        for _ in range(n):
            expr = self._generate_one(idea)
            if expr and expr not in results:
                results.append(expr)
        return results

    def _cached_fields(self) -> list:
        try:
            return list(self.field_repo.load_cached())
        except TypeError:
            return list(self.field_repo.load_cached(None, None, None))

    def _idea_field_context(self) -> str:
        fields = [f for f in self._cached_fields() if _field_attr(f, "id") and not _is_pure_price_volume_field(f)]
        if not fields:
            return "- No cached alternative fields found yet; use the theme list below."

        grouped = defaultdict(list)
        for field in fields:
            dataset_id = _field_attr(field, "dataset_id") or "unknown"
            grouped[dataset_id].append(field)

        def dataset_score(item) -> tuple[int, int, str]:
            dataset_id, items = item
            combined = " ".join(
                f"{_field_attr(f, 'id')} {_field_attr(f, 'description')} {_field_attr(f, 'dataset_id')}"
                for f in items
            )
            alt_hits = sum(1 for keyword in ALT_DATA_KEYWORDS if keyword in _ascii_lower(combined))
            return (alt_hits, min(len(items), 20), dataset_id)

        lines = []
        for dataset_id, items in sorted(grouped.items(), key=dataset_score, reverse=True)[:MAX_IDEA_DATASET_LINES]:
            samples = []
            for field in items[:MAX_IDEA_FIELDS_PER_DATASET]:
                fid = _field_attr(field, "id")
                desc = _field_attr(field, "description").strip()
                samples.append(f"{fid}: {desc[:80]}" if desc else fid)
            lines.append(f"- {dataset_id}: " + "; ".join(samples))
        return "\n".join(lines)

    def _parse_ideas(self, content: str) -> list[str]:
        data = _extract_json(content)
        if isinstance(data, dict):
            ideas = data.get("ideas", [])
        elif isinstance(data, list):
            ideas = data
        else:
            ideas = []
        cleaned = (_strip_fabricated_metrics(str(i)) for i in ideas)
        return [i for i in cleaned if i]

    def _ideas_retry_prompt(self, n: int, rejected: list[str]) -> str:
        rejected_line = "; ".join(rejected[-8:]) or "generic price/volume ideas"
        return (
            f"Need {n} more alpha research directions. The previous ideas were rejected as cliche: "
            f"{rejected_line}. Return ONLY fresh low-correlation directions using cached non-price "
            'datasets/fields when available. JSON format: {"ideas": ["...", "..."]}.'
        )

    def _fallback_ideas(self) -> list[str]:
        return list(ALT_DATA_FALLBACK_IDEAS)

    def build_ideas_system_prompt(self) -> str:
        """Prompt sinh HƯỚNG nghiên cứu, dẫn LLM sang dataset ít khai thác + cấu
        trúc lạ thay vì công thức PV/fundamental kinh điển (dễ trùng -> correlation
        cao -> bị loại). Xem feedback độ độc đáo alpha."""
        themes = "\n".join(f"- {t}" for t in ALT_DATA_THEMES)
        cliches = "; ".join(CLICHE_PATTERNS)
        field_context = self._idea_field_context()
        blacklist_line = ""
        if self.blacklist:
            cam = ", ".join(sorted(self.blacklist)[:50])
            blacklist_line = (
                "TUYỆT ĐỐI KHÔNG dùng field sau (WQ đã từ chối/chết): "
                f"{cam}.\n"
            )
        return (
            "Bạn là nhà nghiên cứu alpha định lượng trên WorldQuant BRAIN, săn tín "
            "hiệu ĐỘC ĐÁO có correlation thấp với các factor đại trà.\n"
            "TRÁNH các công thức kinh điển vì cả nghìn người đã dùng nên trùng nặng "
            f"và bị loại do correlation cao: {cliches}.\n"
            "Thay vào đó, ưu tiên các MẠCH dữ liệu thay thế ít người khai thác và "
            "cấu trúc tổ hợp lạ (chuẩn hóa theo nhóm, phân kỳ hai kênh, lan truyền "
            "lead-lag, z-score đột biến...):\n"
            f"{themes}\n"
            "AVAILABLE NON-PRICE DATASETS/FIELDS from cache. Prefer these concrete IDs when they match the thesis:\n"
            f"{field_context}\n"
            f"{blacklist_line}"
            "Mỗi ý tưởng là một câu ngắn nêu rõ NGUỒN DỮ LIỆU + hiện tượng kinh tế "
            "khai thác (không phải tên operator). Đa dạng nguồn, không lặp một mạch.\n"
            'Trả JSON đúng định dạng: {"ideas": ["...", "..."]}.'
        )

    def generate_ideas(self, n: int = 10) -> list[str]:
        system = self.build_ideas_system_prompt()
        user = (
            f"Đề xuất {n} hướng/ý tưởng alpha độc đáo, mỗi hướng dựa trên một mạch "
            "dữ liệu thay thế khác nhau. Cho phép reversal/độ lệch CÓ chuẩn hóa theo "
            "nhóm nếu nó cho tín hiệu rõ, nhưng tránh momentum giá THUẦN dễ trùng."
        )
        feedback = self._feedback_context()
        if feedback:
            user += "\n\n" + feedback
        results: list[str] = []
        rejected: list[str] = []
        for _ in range(MAX_IDEA_ATTEMPTS):
            content = self.deepseek.complete(system, user, json_mode=True)
            for idea in self._parse_ideas(content):
                if _is_cliche_idea(idea):
                    rejected.append(idea)
                    continue
                if idea not in results:
                    results.append(idea)
                if len(results) >= n:
                    return results[:n]
            user = self._ideas_retry_prompt(max(1, n - len(results)), rejected)

        for idea in self._fallback_ideas():
            if idea not in results and not _is_cliche_idea(idea):
                results.append(idea)
            if len(results) >= n:
                break
        return results[:n]
