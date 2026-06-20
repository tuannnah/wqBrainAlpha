"""SynergyScorer — hàm mục tiêu pool-aware phỏng theo AlphaGen (Yu et al., KDD 2023).

Khác `scorer.score()` (chấm từng alpha độc lập, có bug cho rác 0.12 điểm), đây là
callable drop-in cho `scorer` callback của GA, kết hợp:

  reward = base(score_vector) * originality^beta

với `base` = chất lượng standalone đo THẬT từ WQ-sim, `originality` = đóng góp
biên (proxy IC hiệp đồng của paper) đo bằng AST-similarity cục bộ so với pool
alpha tốt đã tìm. Sim lỗi -> -inf (loại hẳn, vá bug gradient).

Pool online: chỉ alpha `passed` mới được thêm vào zoo -> ứng viên sau bị phạt độ
độc đáo nếu giống alpha tốt đã có, đẩy GA ra vùng decorrelated (cơ chế hiệp đồng).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.decorrelation.zoo import ReferenceZoo
from src.scoring.vector import score_vector

NEG_INF = float("-inf")


@dataclass
class SynergyScorer:
    zoo: ReferenceZoo = field(default_factory=ReferenceZoo)
    beta: float = 1.0  # >1: ép độc đáo mạnh hơn; <1: nới lỏng.

    def __call__(self, result) -> float:
        if getattr(result, "status", None) == "error":
            return NEG_INF
        base = score_vector(result).total
        expr = getattr(result, "expression", None)
        originality = self.zoo.originality(expr) if expr else 1.0
        reward = base * (originality ** self.beta)
        if getattr(result, "status", None) == "passed" and expr:
            self.zoo.add(expr)
        return reward
