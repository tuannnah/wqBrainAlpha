"""Không gian cấu hình của một alpha, tách khỏi không gian biểu thức (T5.1, T5.2).

Một alpha = (biểu thức + cấu hình). Ở giai đoạn sinh/tinh chỉnh biểu thức, cấu hình
cố định ở `default()`; chỉ quét cấu hình SAU khi đã có biểu thức tốt (T5.3). `key()`
cho cache phân biệt theo cấu hình; `to_settings()` để truyền vào Simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Cấu hình mặc định hợp lý, cố định ở giai đoạn sinh biểu thức (T5.2).
DEFAULT_NEUTRALIZATION = "SUBINDUSTRY"
DEFAULT_DECAY = 0
DEFAULT_TRUNCATION = 0.08
DEFAULT_DELAY = 1
VALID_NEUTRALIZATIONS = {
    "NONE",
    "MARKET",
    "SECTOR",
    "INDUSTRY",
    "SUBINDUSTRY",
    "COUNTRY",
    "EXCHANGE",
}


def _normalize_neutralization(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"neutralization must be a string, got {value!r}")
    normalized = str(value).strip().upper()
    if normalized not in VALID_NEUTRALIZATIONS:
        raise ValueError(f"neutralization must be one of {sorted(VALID_NEUTRALIZATIONS)}, got {value!r}")
    return normalized


@dataclass(frozen=True)
class SimConfig:
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = DEFAULT_DELAY
    neutralization: str = DEFAULT_NEUTRALIZATION
    decay: int = DEFAULT_DECAY
    truncation: float = DEFAULT_TRUNCATION

    def __post_init__(self) -> None:
        if not isinstance(self.decay, int) or isinstance(self.decay, bool) or not 0 <= self.decay <= 512:
            raise ValueError(f"decay must be an int in [0, 512], got {self.decay!r}")
        if (
            not isinstance(self.truncation, (int, float))
            or isinstance(self.truncation, bool)
            or not 0.0 < float(self.truncation) <= 0.5
        ):
            raise ValueError(f"truncation must be numeric in (0, 0.5], got {self.truncation!r}")
        object.__setattr__(self, "truncation", float(self.truncation))
        object.__setattr__(self, "neutralization", _normalize_neutralization(self.neutralization))

    @classmethod
    def default(cls, region: str = "USA", universe: str = "TOP3000", delay: int = DEFAULT_DELAY) -> "SimConfig":
        return cls(region=region, universe=universe, delay=delay)

    def with_overrides(self, **changes) -> "SimConfig":
        """Trả bản sao với một số chiều bị ghi đè (bản gốc không đổi)."""
        return replace(self, **changes)

    def to_settings(self) -> dict:
        """Dict settings truyền vào Simulator.simulate(..., settings=...)."""
        return {
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "neutralization": self.neutralization,
            "decay": self.decay,
            "truncation": self.truncation,
        }

    def key(self) -> str:
        """Khoá cấu hình ổn định, người đọc được — phục vụ cache phân biệt theo config."""
        return (
            f"{self.region}|{self.universe}|delay={self.delay}|"
            f"{self.neutralization}|decay={self.decay}|truncation={self.truncation}"
        )
