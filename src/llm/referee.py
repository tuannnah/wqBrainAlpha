"""Trọng tài LLM cho marathon: sau mỗi sim quyết định hành động kế tiếp cho hướng
nghiên cứu, và bộ tinh chỉnh cấu hình (decay/truncation/neutralization).

Mỗi bước chỉ đổi MỘT biến (công thức HOẶC cấu hình) -> cải thiện quy được về đúng
nguyên nhân. Giá trị cấu hình do LLM đề xuất luôn được chốt về tập hợp lệ của WQ
trước khi dùng (tránh config rác); biến nào không hợp lệ thì giữ giá trị cũ.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.llm.jsonutil import extract_json
from src.simulation.config import VALID_NEUTRALIZATIONS, SimConfig

# Ba hành động trọng tài có thể chọn sau mỗi sim.
REFINE_FORMULA = "refine_formula"
TUNE_CONFIG = "tune_config"
ABANDON = "abandon"
_VALID_ACTIONS = {REFINE_FORMULA, TUNE_CONFIG, ABANDON}


@dataclass
class Verdict:
    action: str
    reason: str = ""


def _metric_line(metrics: dict) -> str:
    return ", ".join(
        f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
        for k, v in metrics.items()
    )


class Referee:
    """Sau mỗi sim, đọc metrics + lịch sử refine -> chọn refine_formula | tune_config
    | abandon. Action không hợp lệ / không parse được -> mặc định refine_formula
    (an toàn: tiếp tục thử, trần cứng của loop vẫn chặn vòng vô hạn)."""

    def __init__(self, deepseek):
        self.deepseek = deepseek

    def judge(self, direction: str, history: list, metrics: dict) -> Verdict:
        system = (
            "Bạn là trọng tài nghiên cứu alpha. Sau một mô phỏng, hãy quyết định "
            "bước tiếp theo cho hướng nghiên cứu này, chọn ĐÚNG MỘT trong ba:\n"
            f"- {REFINE_FORMULA}: còn dư địa cải thiện bằng cách sửa BIỂU THỨC.\n"
            f"- {TUNE_CONFIG}: biểu thức ổn nhưng nên đổi THAM SỐ mô phỏng "
            "(decay/truncation/neutralization).\n"
            f"- {ABANDON}: hướng này hết tiềm năng, nên bỏ để sang hướng khác.\n"
            'Trả JSON {"action": "...", "reason": "..."}.'
        )
        recent = history[-5:] if history else []
        hist_lines = "\n".join(
            f"  - total={h.get('total')}: {h.get('expression')}" for h in recent
        )
        user = (
            f"Hướng nghiên cứu: {direction}\n"
            f"Metrics mới nhất: {_metric_line(metrics)}\n"
            f"Lịch sử gần đây:\n{hist_lines or '  (chưa có)'}\n"
            "Chọn hành động kế tiếp."
        )
        data = extract_json(self.deepseek.complete(system, user, json_mode=True, task="referee"))
        if isinstance(data, dict):
            action = str(data.get("action", "")).strip().lower()
            if action in _VALID_ACTIONS:
                return Verdict(action, str(data.get("reason", "") or ""))
        logger.debug("Referee trả action không hợp lệ -> mặc định refine_formula: {}", data)
        return Verdict(REFINE_FORMULA)


def _coerce_decay(value, current: int) -> int:
    try:
        d = int(value)
    except (TypeError, ValueError):
        return current
    if isinstance(value, bool) or not 0 <= d <= 512:
        return current
    return d


def _coerce_truncation(value, current: float) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError):
        return current
    if isinstance(value, bool) or not 0.0 < t <= 0.5:
        return current
    return t


def _coerce_neutralization(value, current: str) -> str:
    if not isinstance(value, str):
        return current
    n = value.strip().upper()
    return n if n in VALID_NEUTRALIZATIONS else current


class ConfigTuner:
    """Đề xuất decay/truncation/neutralization mới cho CÙNG biểu thức. Mọi giá trị
    được chốt về tập hợp lệ; biến không hợp lệ giữ giá trị cũ. Không parse được ->
    trả config nguyên trạng (không đổi gì)."""

    def __init__(self, deepseek):
        self.deepseek = deepseek

    def tune(self, config: SimConfig, metrics: dict, reason: str) -> SimConfig:
        system = (
            "Bạn là chuyên gia tinh chỉnh tham số mô phỏng alpha trên WorldQuant BRAIN. "
            "GIỮ NGUYÊN biểu thức, chỉ đề xuất tham số mô phỏng tốt hơn.\n"
            "Quy ước: decay = số nguyên trong [0, 512]; truncation = số thực trong (0, 0.5]; "
            f"neutralization ∈ {sorted(VALID_NEUTRALIZATIONS)}.\n"
            'Trả JSON {"decay": <int>, "truncation": <float>, "neutralization": "..."}.'
        )
        user = (
            f"Cấu hình hiện tại: decay={config.decay}, truncation={config.truncation}, "
            f"neutralization={config.neutralization}\n"
            f"Metrics: {_metric_line(metrics)}\n"
            f"Lý do cần đổi: {reason}\n"
            "Đề xuất tham số mô phỏng mới."
        )
        data = extract_json(self.deepseek.complete(system, user, json_mode=True, task="tune_config"))
        if not isinstance(data, dict):
            return config
        return config.with_overrides(
            decay=_coerce_decay(data.get("decay"), config.decay),
            truncation=_coerce_truncation(data.get("truncation"), config.truncation),
            neutralization=_coerce_neutralization(data.get("neutralization"), config.neutralization),
        )
