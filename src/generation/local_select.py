"""Chọn-lọc alpha bằng bộ lọc local khi KHÔNG có metric backtest.

User tự mô phỏng trên WQ Brain nên ở bước sinh ta không có sharpe/fitness/turnover.
Vì vậy việc "chọn lọc theo thuật toán" dựa hoàn toàn vào cấu trúc:

1. Cửa cứng `PreFilter`: cú pháp hợp lệ, operator/field tồn tại thật, trong trần.
2. `originality_score`: 1 - max similarity_ratio so với zoo Alpha101 (muốn cao).
3. `complexity_penalty`: phạt mềm độ phức tạp (muốn thấp).
4. Khử trùng nội bộ giữa các ứng viên (similarity_ratio đôi một).
5. Quota mỗi họ để output đa dạng, rồi xếp hạng giảm theo điểm local.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.decorrelation.similarity import similarity_ratio, subtree_canon
from src.generation.ast_utils import parse_expression
from src.scoring.complexity import complexity_penalty
from src.simulation.pre_filter import PreFilter

# Trọng số điểm local: ưu tiên độc đáo, phạt phức tạp.
ORIGINALITY_WEIGHT = 0.6
SIMPLICITY_WEIGHT = 0.4


@dataclass
class Candidate:
    family: str
    expression: str
    hypothesis: str = ""
    rationale: str = ""
    # điền sau khi chấm:
    score: float = 0.0
    originality: float | None = None
    complexity: float | None = None
    reasons: list[str] = field(default_factory=list)
    # override setting riêng từng alpha (decay/truncation theo bản chất tín hiệu).
    overrides: dict = field(default_factory=dict)


def originality_score(expr: str, zoo) -> float:
    """1 - similarity_ratio lớn nhất so với mọi alpha trong zoo. ∈ [0,1].

    Zoo rỗng -> coi như hoàn toàn độc đáo (1.0). Biểu thức nào trong zoo parse
    lỗi thì bỏ qua (không kéo điểm)."""
    max_sim = 0.0
    for ref in zoo:
        try:
            sim = similarity_ratio(expr, ref)
        except ValueError:
            continue
        if sim > max_sim:
            max_sim = sim
    return 1.0 - max_sim


def local_score(expr: str, zoo) -> float:
    """Điểm local ∈ [0,1] = originality*0.6 + (1-complexity)*0.4."""
    orig = originality_score(expr, zoo)
    comp = complexity_penalty(expr)
    return ORIGINALITY_WEIGHT * orig + SIMPLICITY_WEIGHT * (1.0 - comp)


def select_alphas(
    candidates,
    zoo,
    known_operators: set[str] | None = None,
    known_fields: set[str] | None = None,
    known_groups: set[str] | None = None,
    max_depth: int = 6,
    max_nodes: int = 30,
    dedup_threshold: float = 0.85,
    per_family_quota: int | None = None,
    per_canon_quota: int | None = None,
    max_total: int | None = None,
) -> list[Candidate]:
    """Lọc + chấm + khử trùng + quota đa dạng. Trả list Candidate đã sắp giảm điểm.

    - dedup_threshold: similarity_ratio >= ngưỡng giữa hai ứng viên -> coi trùng,
      giữ cái điểm cao hơn. CHỈ dùng khi per_canon_quota là None.
    - per_canon_quota: nếu set, BỎ khử trùng tuyệt đối; thay vào giữ tối đa N biến
      thể mỗi khung canon (field->F, số->N). Hợp với user tự mô phỏng: cùng khung
      nhưng khác field/cửa sổ vẫn là alpha đáng test riêng.
    - per_family_quota: số alpha tối đa giữ cho mỗi họ.
    - max_total: trần tổng số alpha output.
    """
    prefilter = PreFilter(
        known_operators=known_operators,
        known_fields=known_fields,
        known_groups=known_groups,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )

    # 1) cửa cứng + chấm điểm
    scored: list[Candidate] = []
    for c in candidates:
        ok, reason = prefilter.check(c.expression)
        if not ok:
            continue
        c.originality = originality_score(c.expression, zoo)
        c.complexity = complexity_penalty(c.expression)
        c.score = (
            ORIGINALITY_WEIGHT * c.originality
            + SIMPLICITY_WEIGHT * (1.0 - c.complexity)
        )
        c.reasons = [
            f"originality={c.originality:.2f}",
            f"complexity={c.complexity:.2f}",
            "cú pháp/field/operator hợp lệ",
        ]
        scored.append(c)

    # xét theo điểm giảm dần (ưu tiên giữ cái tốt khi khử trùng / quota)
    scored.sort(key=lambda x: x.score, reverse=True)

    # 2) khử trùng nội bộ: theo quota canon (giữ nhiều biến thể) hoặc tuyệt đối
    kept: list[Candidate] = []
    if per_canon_quota is not None:
        per_canon: dict[str, int] = {}
        for c in scored:
            canon = _canon_of(c.expression)
            if per_canon.get(canon, 0) >= per_canon_quota:
                continue
            per_canon[canon] = per_canon.get(canon, 0) + 1
            kept.append(c)
    else:
        for c in scored:
            if _trung_voi_da_chon(c.expression, kept, dedup_threshold):
                continue
            kept.append(c)

    # 3) quota đa dạng theo họ + trần tổng
    result: list[Candidate] = []
    per_family: dict[str, int] = {}
    for c in kept:
        if per_family_quota is not None and per_family.get(c.family, 0) >= per_family_quota:
            continue
        result.append(c)
        per_family[c.family] = per_family.get(c.family, 0) + 1
        if max_total is not None and len(result) >= max_total:
            break

    return result


def _canon_of(expr: str) -> str:
    """Canon cấu trúc của biểu thức (field->F, số->N). Parse lỗi -> dùng nguyên văn."""
    try:
        return subtree_canon(parse_expression(expr))
    except ValueError:
        return expr


def _trung_voi_da_chon(expr: str, kept: list[Candidate], threshold: float) -> bool:
    for k in kept:
        try:
            if similarity_ratio(expr, k.expression) >= threshold:
                return True
        except ValueError:
            continue
    return False
