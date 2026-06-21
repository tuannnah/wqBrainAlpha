"""Tinh chỉnh tham lam một alpha, nhắm vào chiều yếu nhất (T2.12).

Đưa alpha + metrics + chiều cần cải thiện cho LLM, yêu cầu đề xuất cải tiến bằng
lời trước, rồi dịch sang công thức (tái dùng AlphaTranslator cho bước này).
"""

from __future__ import annotations

from src.llm.jsonutil import extract_json
from src.llm.translator import AlphaCandidate, AlphaTranslator

# Ngưỡng turnover (khớp FilterThresholds) để chọn hint theo tốc độ signal.
TURNOVER_HIGH = 0.70
TURNOVER_LOW = 0.01

# Gợi ý cải thiện theo từng chiều của ScoreVector.
DIMENSION_HINTS = {
    "sharpe": "tăng Sharpe (tín hiệu ổn định, ít nhiễu hơn)",
    "fitness": "tăng Fitness (chất lượng tín hiệu sau phí)",
    "turnover_fit": "đưa turnover về mức hợp lý theo tốc độ tín hiệu",
    "drawdown_fit": "giảm drawdown (kiểm soát rủi ro, bớt tập trung)",
    "pool_fit": (
        "GIẢM tương quan với pool đã có (alpha hiện quá giống rổ — không nộp được): "
        "bọc tín hiệu bằng regression_neut/vector_neut để bóc phần chồng lấn với "
        "factor phổ biến, hoặc đổi sang dataset/conditioning khác. KHÔNG chỉ winsorize/scale "
        "(không đổi tương quan). Mục tiêu là phần dư trực giao, không phải tinh chỉnh số."
    ),
    "regime_fit": (
        "tăng độ ỔN ĐỊNH theo năm (alpha hiện có năm sập): bớt phụ thuộc một chế độ "
        "thị trường — chuẩn hoá/neutralize theo thời gian, thêm điều kiện bền (vd lọc "
        "thanh khoản/biến động), hoặc dùng tín hiệu ít theo chu kỳ. Mục tiêu là Sharpe "
        "dương ở MỌI năm, không chỉ trung bình cao."
    ),
}


class AlphaRefiner:
    def __init__(self, deepseek, translator: AlphaTranslator):
        self.deepseek = deepseek
        self.translator = translator
        # Trí nhớ biểu thức đã đề xuất -> không tái đề xuất y hệt (chống vòng lặp
        # thoái hóa kiểu ts_mean(volume,5) bị bơm vô hạn trong log thật).
        self._seen: set[str] = set()

    def _dimension_hint(self, weak_dimension: str, metrics: dict) -> str:
        """Hint cải thiện cho một chiều. Riêng turnover_fit phụ thuộc TỐC ĐỘ tín hiệu
        (review 6): với tín hiệu nhanh (turnover cao), làm mượt/decay thô phá returns
        nhanh hơn phá turnover (fitness ∝ √(|ret|/turnover)) — gậy ông đập lưng ông."""
        if weak_dimension == "turnover_fit":
            turnover = metrics.get("turnover", 0.5)
            if turnover > TURNOVER_HIGH:
                return (
                    "giảm turnover NHƯNG giữ sức mạnh tín hiệu: đây là tín hiệu NHANH, "
                    "làm mượt/decay thô sẽ phá returns nhanh hơn phá turnover. Ưu tiên lọc "
                    "nhiễu/chọn lõi tín hiệu ổn định hoặc giảm tần suất giao dịch có chọn lọc, "
                    "KHÔNG decay mù toàn cục."
                )
            if turnover < TURNOVER_LOW:
                return (
                    "tăng độ nhạy/độ phân tán tín hiệu (turnover quá thấp, alpha gần như "
                    "không giao dịch — có thể đã bị làm mượt quá tay)."
                )
            return "đưa turnover về vùng hợp lý mà không hi sinh chất lượng tín hiệu"
        return DIMENSION_HINTS.get(weak_dimension, weak_dimension)

    def refine(self, candidate: AlphaCandidate, metrics: dict, weak_dimension: str) -> AlphaCandidate | None:
        hint = self._dimension_hint(weak_dimension, metrics)
        description = self._propose(candidate, metrics, hint)
        expression = self.translator._to_expression(description)
        if not expression:
            return None
        if expression in self._seen:
            return None  # đã thử rồi -> bỏ qua, để caller không tốn lượt sim/inject lặp
        self._seen.add(expression)
        return AlphaCandidate(
            hypothesis=candidate.hypothesis,
            description=description,
            expression=expression,
        )

    def _propose(self, candidate: AlphaCandidate, metrics: dict, hint: str) -> str:
        system = (
            "Bạn là chuyên gia tinh chỉnh alpha. Cải thiện alpha hiện tại theo MỘT chiều "
            "được chỉ định, mô tả cải tiến BẰNG LỜI (chưa viết công thức). "
            'Giữ ý tưởng cốt lõi. Trả JSON {"description": "..."}.'
        )
        metric_line = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in metrics.items()
        )
        user = (
            f"Alpha hiện tại: {candidate.expression}\n"
            f"Metrics: {metric_line}\n"
            f"Cần cải thiện: {hint}.\n"
            "Mô tả cách điều chỉnh tín hiệu để cải thiện đúng chiều này."
        )
        if self._seen:
            # Nhồi các biểu thức đã thử để LLM đa dạng hóa thay vì lặp lại.
            user += "\nĐã thử (TRÁNH đề xuất lại y hệt): " + "; ".join(sorted(self._seen))
        data = extract_json(self.deepseek.complete(system, user, json_mode=True, task="refine"))
        if isinstance(data, dict) and data.get("description"):
            return str(data["description"])
        # Không có mô tả mới -> giữ mô tả cũ để vẫn thử sinh biểu thức.
        return candidate.description
