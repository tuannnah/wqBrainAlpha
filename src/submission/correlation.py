"""Kiểm tra self-correlation của alpha trước khi nộp."""

from __future__ import annotations

import time

from loguru import logger


class CorrelationChecker:
    MAX_SELF_CORR = 0.70

    def __init__(self, client, max_self_corr: float | None = None):
        self.client = client
        self.max_self_corr = max_self_corr if max_self_corr is not None else self.MAX_SELF_CORR

    def max_self_correlation(
        self, wq_alpha_id: str, *, max_polls: int = 20, sleep_fn=time.sleep,
    ) -> float:
        """Max self-correlation với pool user. WQ tính BẤT ĐỒNG BỘ: `GET /correlations/self`
        trả HTTP 200 **body RỖNG + header Retry-After** trong lúc tính, chỉ khi xong mới trả
        JSON (schema+records). Phải POLL — trước đây gọi `resp.json()` trên body rỗng ->
        JSONDecodeError làm sập cả bước submit. Poll tới `max_polls` lần rồi mới bỏ cuộc."""
        for _ in range(max(1, max_polls)):
            resp = self.client.get(f"/alphas/{wq_alpha_id}/correlations/self")
            if resp.status_code not in (200, 201):
                logger.warning("Không lấy được correlation cho {}: {}", wq_alpha_id, resp.status_code)
                # Không xác định được → coi như rủi ro cao để an toàn.
                return 1.0
            # Body rỗng + Retry-After = WQ đang tính -> chờ rồi thử lại (fake trong test
            # không set Retry-After nên rơi thẳng xuống _extract_max, giữ hành vi cũ).
            if not (getattr(resp, "text", "") or "").strip() and resp.headers.get("Retry-After"):
                sleep_fn(float(resp.headers.get("Retry-After") or 1.0))
                continue
            return self._extract_max(resp.json())
        logger.warning("Self-corr {} chưa tính xong sau {} lần poll — coi rủi ro cao", wq_alpha_id, max_polls)
        return 1.0

    @staticmethod
    def _extract_max(payload: dict) -> float:
        """Trích max correlation từ nhiều format response có thể gặp."""
        if not isinstance(payload, dict):
            return 1.0
        if "max" in payload and isinstance(payload["max"], (int, float)):
            return float(payload["max"])

        values: list[float] = []
        records = payload.get("records") or payload.get("results") or []
        schema = payload.get("schema", {})
        properties = schema.get("properties") if isinstance(schema, dict) else None
        corr_index = None
        if isinstance(properties, list):
            for idx, prop in enumerate(properties):
                name = prop.get("name", "") if isinstance(prop, dict) else ""
                if "corr" in name.lower():
                    corr_index = idx
                    break
        for row in records:
            if isinstance(row, (list, tuple)):
                if corr_index is not None and corr_index < len(row):
                    values.append(abs(float(row[corr_index])))
            elif isinstance(row, dict):
                for key in ("correlation", "corr", "value"):
                    if key in row and isinstance(row[key], (int, float)):
                        values.append(abs(float(row[key])))
                        break
        return max(values) if values else 0.0

    def is_acceptable(self, wq_alpha_id: str) -> bool:
        return self.max_self_correlation(wq_alpha_id) <= self.max_self_corr
