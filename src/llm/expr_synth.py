"""Lõi dùng chung của hai bộ sinh biểu thức LLM (generator + translator).

Gom phần trùng lặp: dựng ngữ cảnh prompt (symbol + field type), vòng
prefilter-repair, và auto-wrap field VECTOR. Hai lớp công khai
(LLMAlphaGenerator, AlphaTranslator) chỉ uỷ thác phần lõi cho module này.
"""

from __future__ import annotations

import re

from loguru import logger

from src.generation.ast_utils import Leaf, Node, parse_expression, to_expression
from src.llm.jsonutil import extract_json
from src.simulation.simulator import extract_rejected_field

MAX_REPAIR_ATTEMPTS = 3
MAX_FIELDS_IN_PROMPT = 40

# Hint khi PreFilter chặn vì độ sâu/số node (không có field để bóc): hướng LLM làm
# gọn thay vì sinh lại biểu thức sâu y hệt.
DEPTH_REPAIR_HINT = (
    " Biểu thức vượt giới hạn độ sâu/node. Làm GỌN: (1) bỏ bớt lớp BỌC ngoài — chỉ giữ "
    "tối đa MỘT trong {scale, ts_decay_linear, group_neutralize}; (2) làm phẳng các tổ hợp "
    "field lồng nhau (multiply/divide/add/zscore chồng nhiều tầng) thành một phép kết hợp; "
    "(3) ưu tiên tín hiệu NÔNG, ít field."
)
# Từ khoá nhận diện lỗi cấu trúc (độ sâu/số node) trong reason của PreFilter.
_STRUCTURE_ERROR_KEYWORDS = ("độ sâu", "depth", "node", "tầng")


def is_structure_error(reason: str) -> bool:
    """True nếu reason là lỗi độ sâu/số node (không phải lỗi field/operator)."""
    low = (reason or "").lower()
    return any(k in low for k in _STRUCTURE_ERROR_KEYWORDS)


def build_repair_hint(reason: str, suggestions, pinned) -> str:
    """Dựng hint cho lượt repair kế tiếp theo loại lỗi.

    - Lỗi cấu trúc (độ sâu/node) -> DEPTH_REPAIR_HINT (hướng dẫn bỏ lớp wrapper).
    - Lỗi field -> gợi ý field thật gần nhất + (nếu pinned) ràng buộc CHỈ dùng field ghim.
    """
    if is_structure_error(reason):
        return DEPTH_REPAIR_HINT
    hint = ""
    if suggestions:
        hint = f" Field có thật gần nhất: {', '.join(suggestions)}."
    if pinned:
        allowed = ", ".join([p for p in pinned if isinstance(p, str)][:MAX_FIELDS_IN_PROMPT])
        hint += f" CHỈ được dùng các field: {allowed}."
    return hint

# Ví dụ minh hoạ CÚ PHÁP, đa dạng cấu trúc, tránh khung kinh điển trùng Alpha101.
# CHỈ tín hiệu LÕI — không bọc scale/decay/neutralize (config-layer xử lý ở stage sim).
FEWSHOT_EXAMPLES = [
    "rank(ts_std_dev(returns, 20))",
    "ts_zscore(vwap, 60)",
    "rank(divide(ts_mean(volume, 10), ts_mean(volume, 60)))",
    "ts_rank(ts_corr(close, volume, 20), 120)",
    "multiply(-1, ts_delta(close, 5))",
]


def autowrap_vector_fields(expr: str, field_types, matrix_only_ops) -> str:
    """Bọc vec_avg() quanh leaf field VECTOR bị đưa thẳng vào matrix-only op.

    Khớp ĐÚNG luật pre_filter._check_symbols: với Node có op ∈ matrix_only_ops,
    con TRỰC TIẾP là Leaf field có field_types[name]=='VECTOR' -> thay bằng
    vec_avg(leaf). Thiếu dữ liệu kiểu -> trả nguyên. Không parse được -> trả
    nguyên để prefilter báo lỗi (không nuốt lỗi).
    """
    if not field_types or not matrix_only_ops:
        return expr
    try:
        tree = parse_expression(expr)
    except ValueError:
        return expr

    def _walk(node):
        if isinstance(node, Leaf):
            return node
        wrap_here = node.op in matrix_only_ops
        new_children = []
        for child in node.children:
            child = _walk(child)
            if (
                wrap_here
                and isinstance(child, Leaf)
                and not isinstance(child.value, (int, float))
                and field_types.get(str(child.value)) == "VECTOR"
            ):
                child = Node("vec_avg", [child])
            new_children.append(child)
        node.children = new_children
        return node

    return to_expression(_walk(tree))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _load_cached(field_repo, scope):
    """Nạp fields: có scope -> load_cached(**scope); không -> load_cached().
    Bao try/except chữ ký để tương thích repo cũ không nhận tham số."""
    if scope:
        return list(field_repo.load_cached(**scope))
    try:
        return list(field_repo.load_cached())
    except TypeError:
        return list(field_repo.load_cached(None, None, None))


def _relevant_fields(cached_fields, text: str) -> list[str]:
    """Xếp hạng fields theo độ liên quan với text (hypothesis/idea/mô tả), cắt
    MAX_FIELDS_IN_PROMPT. Text rỗng -> giữ thứ tự gốc (tương thích)."""
    text_low = (text or "").lower()
    text_tokens = _tokens(text_low)
    scored = []
    for idx, f in enumerate(cached_fields):
        fid = getattr(f, "id", None)
        if not fid:
            continue
        dataset = (getattr(f, "dataset_id", "") or "").lower()
        score = 0
        if fid.lower() in text_low:
            score += 100
        if dataset and dataset in text_low:
            score += 20
        score += len(_tokens(fid + " " + (getattr(f, "description", "") or "")) & text_tokens)
        scored.append((score, idx, fid))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [fid for _, _, fid in scored[:MAX_FIELDS_IN_PROMPT]]


# Token gợi ý dataset thay thế (option/news/social/analyst/graph) để độn palette
# khi xếp hạng từ-vựng không đủ min_k field liên quan.
ALT_THEME_TOKENS = {
    "option", "implied", "iv", "put", "call", "skew", "pcr", "news", "event",
    "sentiment", "novelty", "social", "buzz", "scl", "analyst", "revision",
    "target", "recommendation", "supply", "graph", "competitor", "customer",
}


def retrieve_field_palette(field_repo, scope, text, k: int = 20, min_k: int = 8) -> list:
    """Palette field THẬT liên quan `text`, trả đối tượng field. Đảm bảo không rỗng
    khi cache không rỗng: thiếu thì độn theo theme alt-data rồi thứ tự cache."""
    cached = _load_cached(field_repo, scope)
    by_id = {getattr(f, "id", None): f for f in cached if getattr(f, "id", None)}
    ranked_ids = _relevant_fields(cached, text)
    out = [by_id[fid] for fid in ranked_ids[:k] if fid in by_id]
    if len(out) >= min_k:
        return out
    chosen = {getattr(f, "id", None) for f in out}

    def _pad(candidates):
        for f in candidates:
            fid = getattr(f, "id", None)
            if fid and fid not in chosen:
                out.append(f)
                chosen.add(fid)
                if len(out) >= min_k:
                    return True
        return False

    themed = [
        f for f in cached
        if _tokens(f"{getattr(f, 'id', '')} {getattr(f, 'description', '') or ''}") & ALT_THEME_TOKENS
    ]
    if not _pad(themed):
        _pad(cached)
    return out


def _field_type_context(selected_fields) -> str:
    by_type: dict[str, list[str]] = {}
    for field in selected_fields:
        fid = getattr(field, "id", None)
        ftype = (getattr(field, "type", "") or "").strip().upper()
        if fid and ftype:
            by_type.setdefault(ftype, []).append(fid)
    if not by_type:
        return ""

    lines = ["FIELD TYPES (dung de tranh sai kieu input):"]
    for ftype in ("MATRIX", "VECTOR", "GROUP", "EVENT"):
        values = by_type.get(ftype)
        if values:
            lines.append(f"- {ftype}: {', '.join(values[:20])}")
    vector_fields = by_type.get("VECTOR") or []
    if vector_fields:
        sample = vector_fields[0]
        lines.append(
            "QUY TAC VECTOR: khong goi truc tiep ts_zscore/ts_mean/ts_rank/rank tren VECTOR field. "
            "Hay giam VECTOR ve MATRIX bang vec_avg(field) hoac vec_sum(field) truoc. "
            f"Vi du: ts_zscore(vec_avg({sample}), 20), rank(vec_avg({sample}))."
        )
    return "\n".join(lines)


def build_symbol_context(
    field_repo, operator_repo, prefilter, scope, relevance_text: str = "", pinned=None
) -> str:
    operators = [o.name for o in operator_repo.load_cached() if getattr(o, "name", None)]
    cached_fields = _load_cached(field_repo, scope)
    field_by_id = {getattr(f, "id", None): f for f in cached_fields if getattr(f, "id", None)}
    if pinned:
        fields = [fid for fid in dict.fromkeys(pinned) if fid in field_by_id][:MAX_FIELDS_IN_PROMPT]
    else:
        fields = _relevant_fields(cached_fields, relevance_text)
    selected_fields = [field_by_id[fid] for fid in fields if fid in field_by_id]
    type_context = _field_type_context(selected_fields)
    op_line = ", ".join(operators[:80]) or "rank, ts_delta, ts_mean, group_neutralize, ts_corr"
    field_line = ", ".join(fields) or "close, open, high, low, volume, vwap, returns"
    examples = "\n".join(f"- {e}" for e in FEWSHOT_EXAMPLES)
    context = (
        f"OPERATORS hợp lệ: {op_line}\n"
        f"FIELDS khả dụng: {field_line}\n"
        f"Ví dụ TÍN HIỆU LÕI hợp lệ:\n{examples}"
    )
    if type_context:
        context += f"\n{type_context}"
    if pinned and fields:
        context += "\nTUYỆT ĐỐI chỉ dùng field trong danh sách FIELDS trên; KHÔNG bịa tên field mới."
    return context


def build_syntax_constraints(prefilter) -> str:
    """Ràng buộc cú pháp suy ra từ pre-filter để biểu thức qua lọc ngay."""
    max_depth = getattr(prefilter, "max_depth", 6)
    max_nodes = getattr(prefilter, "max_nodes", 30)
    return (
        "RÀNG BUỘC bắt buộc để qua bộ lọc cú pháp:\n"
        "- CHỈ sinh BIỂU THỨC TÍN HIỆU LÕI. KHÔNG bọc scale / ts_decay_linear (decay) / "
        "group_neutralize (neutralize) — các bước này do tầng CẤU HÌNH (sim) xử lý sau; "
        "toàn bộ ngân sách độ sâu dành cho tín hiệu.\n"
        f"- Độ sâu lồng nhau TỐI ĐA {max_depth}; tổng số node TỐI ĐA {max_nodes}. "
        "Ưu tiên biểu thức GỌN và NÔNG, tránh lồng quá nhiều tầng.\n"
        "- CHỈ dùng đối số theo VỊ TRÍ. TUYỆT ĐỐI không dùng đối số có tên kiểu "
        "key=value (vd viết winsorize(x, 3) chứ KHÔNG viết winsorize(x, std=3)).\n"
        "- Đối số chỉ là field/group đã liệt kê, biểu thức con, hoặc SỐ NGUYÊN.\n"
    )


def suggest_fields(field_repo, scope, bad_field: str, limit: int = 5, pinned=None) -> list[str]:
    """Field thật gần 'bad_field' nhất: ưu tiên cùng tiền tố dataset, rồi trùng token.
    Không khớp gì -> fallback pinned hoặc top field cache (không bao giờ rỗng)."""
    cached = _load_cached(field_repo, scope)
    bad_low = (bad_field or "").lower()
    bad_prefix = bad_low.split("_", 1)[0]
    bad_tokens = set(re.findall(r"[a-z0-9]+", bad_low))
    scored = []
    for f in cached:
        fid = getattr(f, "id", None)
        if not fid:
            continue
        fl = fid.lower()
        score = 0
        if bad_prefix and fl.startswith(bad_prefix):
            score += 50
        score += len(set(re.findall(r"[a-z0-9]+", fl)) & bad_tokens)
        if score:
            scored.append((score, fid))
    scored.sort(key=lambda t: -t[0])
    result = [fid for _, fid in scored[:limit]]
    if not result:
        if pinned:
            result = [p for p in pinned if isinstance(p, str)][:limit]
        else:
            result = [getattr(f, "id", None) for f in cached if getattr(f, "id", None)][:limit]
    return result


def repair_to_expression(deepseek, prefilter, field_repo, scope, system, user, task, pinned=None) -> str | None:
    """Vòng LLM -> auto-wrap -> prefilter.check -> retry kèm hint field thay thế."""
    field_types = getattr(prefilter, "field_types", None)
    matrix_only = getattr(prefilter, "matrix_only_ops", None)
    for attempt in range(MAX_REPAIR_ATTEMPTS):
        data = extract_json(deepseek.complete(system, user, json_mode=True, task=task))
        expr = data.get("expression") if isinstance(data, dict) else None
        if not expr:
            user = 'Trả ĐÚNG JSON {"expression": "..."}.'
            continue
        expr = autowrap_vector_fields(expr, field_types, matrix_only)
        ok, reason = prefilter.check(expr)
        if ok:
            return expr
        logger.info("LLM expr lỗi (lần {}): {} — {}", attempt + 1, expr, reason)
        bad = extract_rejected_field(reason)
        suggestions = suggest_fields(field_repo, scope, bad, pinned=pinned) if bad else []
        hint = build_repair_hint(reason, suggestions, pinned)
        user = f'Biểu thức "{expr}" bị lỗi: {reason}.{hint} Sửa lại, trả JSON.'
    return None
