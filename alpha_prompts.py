"""Hợp đồng prompt nghiêm ngặt cho DeepSeek và ánh xạ response thành AlphaDraft."""

from research_models import AlphaDraft


REQUIRED_ALPHA_KEYS = (
    "hypothesis",
    "rationale",
    "expression",
    "dataset_ids",
    "field_ids",
    "operator_names",
    "settings",
)

_COMMON_RULES = (
    "Return one JSON object only.\n"
    "Use only allowed_fields, allowed_operators, and allowed_settings.\n"
    "Do not invent identifiers.\n"
    "Each Alpha must contain hypothesis, rationale, expression, dataset_ids,\n"
    "field_ids, operator_names, and settings."
)


class PromptResponseError(ValueError):
    """Response DeepSeek thiếu khóa bắt buộc hoặc vi phạm hợp đồng."""


def _allowed_context(context):
    return {
        "allowed_dataset_ids": list(context.dataset_ids),
        "allowed_fields": [
            {
                "id": field["id"],
                "field_type": field.get("field_type"),
                "description": field.get("description"),
            }
            for field in context.fields
        ],
        "allowed_operators": [op["name"] for op in context.operators],
        "allowed_settings": {
            "instrumentType": context.scope.instrument_type,
            "region": context.scope.region,
            "delay": context.scope.delay,
            "universe": context.scope.universe,
        },
    }


def build_idea_prompt(catalog, lessons):
    system = (
        f"{_COMMON_RULES}\n"
        "Propose one broad research idea as JSON with keys: title, content,\n"
        "dataset_keywords, field_keywords. Avoid repeating failed lessons."
    )
    payload = {
        "dataset_catalog": catalog,
        "lessons": list(lessons or []),
    }
    return system, payload


def build_root_alpha_prompt(idea, context, count, lessons):
    system = (
        f"{_COMMON_RULES}\n"
        f"Return exactly {count} Alpha objects under key 'alphas'.\n"
        "Each Alpha must express a distinct hypothesis from the others."
    )
    payload = {
        "idea": idea,
        "count": count,
        "lessons": list(lessons or []),
        **_allowed_context(context),
    }
    return system, payload


def build_variant_prompt(parent, direction, context):
    system = (
        f"{_COMMON_RULES}\n"
        "Return exactly one Alpha under key 'alphas'.\n"
        f"Improve the parent Alpha only along: {direction}.\n"
        "Do not change anything else."
    )
    payload = {
        "improvement_direction": direction,
        "parent": {
            "expression": _get(parent, "expression"),
            "hypothesis": _get(parent, "hypothesis"),
            "metrics": _get(parent, "metrics", {}),
        },
        **_allowed_context(context),
    }
    return system, payload


# -- Response mapping ------------------------------------------------------

def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _require_keys(alpha):
    if not isinstance(alpha, dict):
        raise PromptResponseError("Mỗi Alpha phải là một object JSON.")
    missing = [key for key in REQUIRED_ALPHA_KEYS if key not in alpha]
    if missing:
        raise PromptResponseError(f"Alpha thiếu khóa: {missing}")


def _to_draft(alpha, parent_id=None, generation=0, improvement_direction=None):
    _require_keys(alpha)
    return AlphaDraft(
        hypothesis=str(alpha["hypothesis"]).strip(),
        rationale=str(alpha["rationale"]),
        expression=str(alpha["expression"]),
        dataset_ids=list(alpha["dataset_ids"]),
        field_ids=list(alpha["field_ids"]),
        operator_names=list(alpha["operator_names"]),
        settings=dict(alpha["settings"]),
        parent_id=parent_id,
        generation=generation,
        improvement_direction=improvement_direction,
    )


def map_root_response(data, context):
    alphas = data.get("alphas") if isinstance(data, dict) else None
    if not isinstance(alphas, list) or not alphas:
        raise PromptResponseError("Response thiếu danh sách 'alphas'.")

    drafts = []
    seen = set()
    for alpha in alphas:
        draft = _to_draft(alpha, generation=0)
        key = draft.hypothesis.lower()
        if key in seen:
            raise PromptResponseError("Có hypothesis trùng trong cùng một lô.")
        seen.add(key)
        drafts.append(draft)
    return drafts


def map_variant_response(data, parent_alpha_id, direction):
    alphas = data.get("alphas") if isinstance(data, dict) else None
    if not isinstance(alphas, list) or len(alphas) != 1:
        raise PromptResponseError("Variant phải trả về đúng một Alpha.")
    return _to_draft(
        alphas[0],
        parent_id=parent_alpha_id,
        generation=1,
        improvement_direction=direction,
    )
