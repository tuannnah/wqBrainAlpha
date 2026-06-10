"""Nạp và kiểm tra cấu hình nghiên cứu (không chứa secret)."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


class ConfigError(ValueError):
    """Lỗi cấu hình nghiên cứu."""


@dataclass(frozen=True)
class ResearchConfig:
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_timeout_seconds: int = 90
    deepseek_max_retries: int = 2
    deepseek_max_output_tokens: int = 6000
    root_alphas_per_batch: int = 5
    max_batches_per_idea: int = 3
    max_parents: int = 2
    variants_per_parent: int = 5
    quality_gate_ratio: float = 0.8
    sharpe_threshold: float = 1.5
    fitness_threshold: float = 1.0
    turnover_min: float = 0.01
    turnover_hard_limit: float = 0.9
    similarity_threshold: float = 0.9
    candidate_fields_min: int = 20
    candidate_fields_max: int = 50
    target_qualified_per_run: int = 10
    simulation_poll_timeout_seconds: int = 900
    simulation_delay_seconds: float = 5.0
    rate_limit_backoff_seconds: float = 2.0
    raw_response_max_chars: int = 200_000
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5

    def validate(self):
        positive = (
            "deepseek_timeout_seconds",
            "deepseek_max_output_tokens",
            "root_alphas_per_batch",
            "max_batches_per_idea",
            "max_parents",
            "variants_per_parent",
            "candidate_fields_min",
            "candidate_fields_max",
            "target_qualified_per_run",
            "simulation_poll_timeout_seconds",
            "raw_response_max_chars",
            "log_max_bytes",
            "log_backup_count",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} phải lớn hơn 0")
        for name in (
            "quality_gate_ratio",
            "turnover_hard_limit",
            "similarity_threshold",
        ):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ConfigError(f"{name} phải nằm trong khoảng (0, 1]")
        for name in (
            "sharpe_threshold",
            "fitness_threshold",
            "turnover_min",
            "simulation_delay_seconds",
            "rate_limit_backoff_seconds",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} không được âm")
        if self.candidate_fields_min > self.candidate_fields_max:
            raise ConfigError(
                "candidate_fields_min không được lớn hơn candidate_fields_max"
            )
        if self.variants_per_parent > 5:
            raise ConfigError("variants_per_parent không được lớn hơn 5")


def load_config(path):
    path = Path(path)
    defaults = ResearchConfig()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(defaults), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return defaults
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(ResearchConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"Khóa config không hỗ trợ: {sorted(unknown)}")
    try:
        config = ResearchConfig(**raw)
    except TypeError as exc:
        raise ConfigError(str(exc)) from exc
    config.validate()
    return config
