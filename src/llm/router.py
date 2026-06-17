"""Định tuyến đa model theo loại tác vụ (T6.3).

Model rẻ/nhanh cho sinh & đột biến hàng loạt; model mạnh cho suy luận khó (sinh
giả thuyết, chấm nhất quán/độc đáo). Giữ nguyên interface `complete(system, user,
json_mode)` mà các module hiện dùng — chỉ thêm tham số `task` tùy chọn để định
tuyến. Không có model mạnh riêng -> mọi tác vụ về model rẻ.
"""

from __future__ import annotations

from src.llm.deepseek_client import Usage

# Tác vụ cần suy luận khó -> model mạnh. Còn lại -> model rẻ.
STRONG_TASKS = {"hypothesis", "alignment", "originality", "describe"}


class ModelRouter:
    def __init__(self, cheap, strong=None, default: str = "strong"):
        self.cheap = cheap
        self.strong = strong or cheap  # không có model mạnh -> dùng rẻ cho tất cả
        self.default = default
        self._routes: dict[str, str] = {t: "strong" for t in STRONG_TASKS}

    def set_route(self, task: str, target: str) -> None:
        """Ép một tác vụ dùng model 'cheap' hoặc 'strong' (cấu hình được)."""
        self._routes[task] = target

    def _pick(self, task: str | None):
        target = self.default if task is None else self._routes.get(task, "cheap")
        return self.strong if target == "strong" else self.cheap

    def complete(self, system: str, user: str, json_mode: bool = True, task: str | None = None) -> str:
        return self._pick(task).complete(system, user, json_mode=json_mode)

    @property
    def usage(self) -> Usage:
        """Gộp usage từ các model con (nếu cheap is strong thì chỉ tính một lần)."""
        total = Usage()
        seen = set()
        for model in (self.cheap, self.strong):
            if id(model) in seen:
                continue
            seen.add(id(model))
            u = getattr(model, "usage", None)
            if u is not None:
                total.prompt_tokens += u.prompt_tokens
                total.completion_tokens += u.completion_tokens
        return total
