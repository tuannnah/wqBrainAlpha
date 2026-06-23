"""Lỗi LLM dùng chung. Tách 'hết quota' khỏi lỗi tạm thời để marathon dừng đúng lúc."""

from __future__ import annotations

# Mẫu nhận diện hết quota/giới hạn dùng (claude-cli/codex-cli) trong stderr. Khớp
# không phân biệt hoa thường. Cố ý hẹp để không nhầm lỗi mạng/cú pháp thành quota.
QUOTA_PATTERNS = (
    "usage limit",
    "rate limit",
    "quota",
    "429",
    "too many requests",
    "limit reached",
)


def is_quota_error(text: str) -> bool:
    """True nếu thông điệp lỗi mang dấu hiệu hết quota/giới hạn dùng."""
    low = (text or "").lower()
    return any(p in low for p in QUOTA_PATTERNS)


class QuotaExhaustedError(RuntimeError):
    """LLM báo hết quota/giới hạn dùng. Kế thừa RuntimeError để code cũ bắt
    RuntimeError vẫn hoạt động; marathon bắt riêng loại này để DỪNG (không retry)."""
