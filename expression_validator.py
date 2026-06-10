"""Kiểm tra Alpha do DeepSeek sinh dựa trên metadata, scope và lịch sử."""

from dataclasses import dataclass, field
from typing import List

from expression_parser import (
    ExpressionSyntaxError,
    iter_field_contexts,
    parse_expression,
)


GROUP_IDENTIFIERS = {
    "market", "sector", "industry", "subindustry", "country", "exchange",
}
ALLOWED_NEUTRALIZATION = {
    "NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "COUNTRY",
}


@dataclass(frozen=True)
class ValidationError:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error_codes: List[str]
    errors: List[ValidationError] = field(default_factory=list)
    expression_hash: str = None
    fingerprint: str = None
    normalized_expression: str = None

    @classmethod
    def invalid(cls, code, message):
        return cls(False, [code], [ValidationError(code, message)])


class ExpressionValidator:
    def __init__(self, metadata_store, research_store, config):
        self.metadata_store = metadata_store
        self.research_store = research_store
        self.config = config

    def validate(self, draft, context):
        try:
            parsed = parse_expression(draft.expression)
        except ExpressionSyntaxError as exc:
            return ValidationResult.invalid("SYNTAX_ERROR", str(exc))

        errors = []
        errors += self._validate_operators(parsed, context)
        errors += self._validate_identifiers(parsed, context)
        errors += self._validate_types(draft, context)
        errors += self._validate_scope_and_settings(draft, context)
        errors += self._validate_duplicates(parsed)

        seen = set()
        unique_errors = []
        for error in errors:
            if error.code not in seen:
                seen.add(error.code)
                unique_errors.append(error)

        return ValidationResult(
            is_valid=not unique_errors,
            error_codes=[error.code for error in unique_errors],
            errors=unique_errors,
            expression_hash=parsed.expression_hash,
            fingerprint=parsed.fingerprint,
            normalized_expression=parsed.normalized_expression,
        )

    # -- Rules -------------------------------------------------------------

    @staticmethod
    def _validate_operators(parsed, context):
        allowed = {operator["name"] for operator in context.operators}
        return [
            ValidationError("UNKNOWN_OPERATOR", f"Operator không hợp lệ: {name}")
            for name in sorted(parsed.operator_names)
            if name not in allowed
        ]

    @staticmethod
    def _validate_identifiers(parsed, context):
        field_ids = {item["id"] for item in context.fields}
        errors = []
        for name in sorted(parsed.identifiers):
            if name in field_ids or name in GROUP_IDENTIFIERS:
                continue
            errors.append(ValidationError("UNKNOWN_FIELD", f"Field không hợp lệ: {name}"))
        return errors

    @staticmethod
    def _validate_types(draft, context):
        field_types = {item["id"]: item["field_type"] for item in context.fields}
        errors = []
        for usage in iter_field_contexts(draft.expression):
            field_type = field_types.get(usage.name)
            if field_type == "VECTOR":
                if not any(op.startswith("vec_") for op in usage.ancestors):
                    errors.append(ValidationError(
                        "VECTOR_REDUCER_REQUIRED",
                        f"Field VECTOR {usage.name} cần operator vec_* để giảm chiều.",
                    ))
            elif field_type == "GROUP":
                parent = usage.parent_operator or ""
                if not parent.startswith("group_"):
                    errors.append(ValidationError(
                        "GROUP_POSITION_REQUIRED",
                        f"Field GROUP {usage.name} chỉ dùng trong operator group_*.",
                    ))
        return errors

    @staticmethod
    def _validate_scope_and_settings(draft, context):
        scope = context.scope
        settings = draft.settings or {}
        errors = []
        expected = {
            "instrumentType": scope.instrument_type,
            "region": scope.region,
            "delay": scope.delay,
            "universe": scope.universe,
        }
        for key, value in expected.items():
            if str(settings.get(key)) != str(value):
                errors.append(ValidationError(
                    "SETTINGS_SCOPE_MISMATCH",
                    f"Settings {key}={settings.get(key)} không khớp scope {value}.",
                ))
                break
        neutralization = settings.get("neutralization")
        if neutralization is not None and neutralization not in ALLOWED_NEUTRALIZATION:
            errors.append(ValidationError(
                "INVALID_NEUTRALIZATION",
                f"Neutralization không hợp lệ: {neutralization}",
            ))
        return errors

    def _validate_duplicates(self, parsed):
        hashes = self.research_store.find_expression_hashes()
        if parsed.expression_hash in hashes:
            return [ValidationError(
                "DUPLICATE_EXPRESSION", "Biểu thức trùng exact hash với Alpha đã có."
            )]
        threshold = self.config.similarity_threshold
        for existing in self.research_store.find_fingerprints():
            if self._jaccard(parsed.fingerprint, existing) >= threshold:
                return [ValidationError(
                    "SIMILAR_EXPRESSION",
                    "Biểu thức quá giống một Alpha đã có.",
                )]
        return []

    @staticmethod
    def _jaccard(first, second):
        grams_first = _char_ngrams(first, 3)
        grams_second = _char_ngrams(second, 3)
        if not grams_first or not grams_second:
            return 1.0 if first == second else 0.0
        intersection = grams_first & grams_second
        union = grams_first | grams_second
        return len(intersection) / len(union)


def _char_ngrams(text, size):
    text = text or ""
    if len(text) < size:
        return {text} if text else set()
    return {text[index:index + size] for index in range(len(text) - size + 1)}
