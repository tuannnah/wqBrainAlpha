# SimConfig Extra Settings (Sub-project E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm 3 field còn thiếu trong `SimConfig` (`test_period`, `max_trade`,
`max_position`) và tự động bật `max_trade="ON"` cho các region BẮT BUỘC theo tài liệu
(ASI/JPN/HKG/KOR/TWN) khi dùng `SimConfig.default()`.

**Architecture:** `SimConfig` là dataclass thuần (`src/simulation/config.py`) — thêm 3 field
mới cùng kiểu validate như `neutralization` đã có, mở rộng `to_settings()`/`key()`. LƯU Ý:
`pasteurization`/`nanHandling`/`unitHandling` đã có sẵn giá trị đúng mặc định trong
`SIM_DEFAULTS["settings"]` (`src/simulation/simulator.py:103-119`, đã kiểm tra: `pasteurization:
"ON"`, `nanHandling: "OFF"`) — KHÔNG cần thêm vào `SimConfig`, chỉ 3 field thật sự thiếu
(`test_period`/`max_trade`/`max_position` không có trong `SIM_DEFAULTS` lẫn `SimConfig`).

**Tech Stack:** Python dataclass thuần, không I/O.

## Global Constraints

- TDD bắt buộc: test FAIL trước, code tối thiểu, xác nhận PASS.
- Code/comment/commit tiếng Việt có dấu.
- Mỗi task = 1 commit.
- Chạy test: `venv/Scripts/python -m pytest`.
- **Tác dụng phụ đã biết và CHẤP NHẬN**: `key()` đổi format (thêm field mới) → mọi
  `expr_hash(expr, config_key)` cũ trong DB không còn khớp cache mới → lần chạy tiếp theo sẽ
  simulate lại (không dùng nhầm cache của config cũ) thay vì lỗi âm thầm — đánh đổi hợp lý, ghi
  rõ ở đây để không ai coi là bug khi thấy cache "mất".

---

### Task 1: Thêm field `test_period`/`max_trade`/`max_position` + validate + `to_settings()`/`key()`

**Files:**
- Modify: `src/simulation/config.py` (toàn bộ file, xem code đầy đủ bên dưới)
- Test: `tests/test_sim_config.py`

**Interfaces:**
- Consumes: không có (dataclass thuần).
- Produces: `SimConfig.test_period: str` (mặc định `"P0Y0M"`, format ISO-8601 duration WQ Brain
  dùng), `SimConfig.max_trade: str` (`"ON"`/`"OFF"`), `SimConfig.max_position: str`
  (`"ON"`/`"OFF"`) — dùng bởi Task 2 và `to_settings()`/`Simulator.simulate()`.

- [ ] **Step 1: Viết test FAIL**

Thêm vào cuối `tests/test_sim_config.py`:

```python
def test_test_period_max_trade_max_position_mac_dinh():
    c = SimConfig.default()
    assert c.test_period == "P0Y0M"
    assert c.max_trade == "OFF"
    assert c.max_position == "OFF"


def test_to_settings_co_test_period_max_trade_max_position():
    c = SimConfig.default(region="USA").with_overrides(max_position="ON")
    s = c.to_settings()
    assert s["testPeriod"] == "P0Y0M"
    assert s["maxTrade"] == "OFF"
    assert s["maxPosition"] == "ON"


def test_max_trade_normalize_hoa_thuong():
    assert SimConfig(max_trade="on").max_trade == "ON"


def test_max_trade_gia_tri_khong_hop_le_raise():
    with pytest.raises(ValueError, match="max_trade"):
        SimConfig(max_trade="MAYBE")


def test_max_position_gia_tri_khong_hop_le_raise():
    with pytest.raises(ValueError, match="max_position"):
        SimConfig(max_position="MAYBE")


def test_key_phan_biet_theo_max_trade():
    a = SimConfig.default()
    b = SimConfig.default().with_overrides(max_trade="ON")
    assert a.key() != b.key()
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/test_sim_config.py -k "test_period or max_trade or max_position" -v`
Expected: FAIL với `TypeError: __init__() got an unexpected keyword argument 'test_period'`

- [ ] **Step 3: Cài tối thiểu**

Thay toàn bộ nội dung `src/simulation/config.py` bằng:

```python
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
DEFAULT_TEST_PERIOD = "P0Y0M"  # ISO-8601 duration; P0Y0M = không dùng Test Period (mặc định WQ)
VALID_NEUTRALIZATIONS = {
    "NONE",
    "MARKET",
    "SECTOR",
    "INDUSTRY",
    "SUBINDUSTRY",
    "COUNTRY",
    "EXCHANGE",
}
# Region BẮT BUỘC max_trade=ON theo tài liệu consultant-simulation-features/consultant-features
# (sub-project E, docs/superpowers/specs/2026-07-02-submission-compliance-roadmap-design.md).
REQUIRE_MAX_TRADE_REGIONS = {"ASI", "JPN", "HKG", "KOR", "TWN"}


def _normalize_neutralization(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"neutralization must be a string, got {value!r}")
    normalized = str(value).strip().upper()
    if normalized not in VALID_NEUTRALIZATIONS:
        raise ValueError(f"neutralization must be one of {sorted(VALID_NEUTRALIZATIONS)}, got {value!r}")
    return normalized


def _normalize_on_off(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {value!r}")
    normalized = value.strip().upper()
    if normalized not in ("ON", "OFF"):
        raise ValueError(f"{field_name} must be 'ON' or 'OFF', got {value!r}")
    return normalized


@dataclass(frozen=True)
class SimConfig:
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = DEFAULT_DELAY
    neutralization: str = DEFAULT_NEUTRALIZATION
    decay: int = DEFAULT_DECAY
    truncation: float = DEFAULT_TRUNCATION
    test_period: str = DEFAULT_TEST_PERIOD
    max_trade: str = "OFF"
    max_position: str = "OFF"

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
        object.__setattr__(self, "max_trade", _normalize_on_off(self.max_trade, "max_trade"))
        object.__setattr__(self, "max_position", _normalize_on_off(self.max_position, "max_position"))

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
            "testPeriod": self.test_period,
            "maxTrade": self.max_trade,
            "maxPosition": self.max_position,
        }

    def key(self) -> str:
        """Khoá cấu hình ổn định, người đọc được — phục vụ cache phân biệt theo config."""
        return (
            f"{self.region}|{self.universe}|delay={self.delay}|"
            f"{self.neutralization}|decay={self.decay}|truncation={self.truncation}|"
            f"test_period={self.test_period}|max_trade={self.max_trade}|"
            f"max_position={self.max_position}"
        )
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/test_sim_config.py -v`
Expected: PASS toàn bộ (test cũ + 6 test mới)

- [ ] **Step 5: Commit**

```bash
git add src/simulation/config.py tests/test_sim_config.py
git commit -m "feat(simulation): them test_period/max_trade/max_position vao SimConfig"
```

---

### Task 2: `SimConfig.default()` tự bật `max_trade=ON` cho region bắt buộc

**Files:**
- Modify: `src/simulation/config.py` (method `default`, đã sửa ở Task 1 — sửa tiếp)
- Test: `tests/test_sim_config.py`

**Interfaces:**
- Consumes: `REQUIRE_MAX_TRADE_REGIONS` (đã thêm ở Task 1).
- Produces: hành vi mới của `SimConfig.default(region=...)` — không thêm hàm/tên mới.

- [ ] **Step 1: Viết test FAIL**

Thêm vào cuối `tests/test_sim_config.py`:

```python
@pytest.mark.parametrize("region", ["ASI", "JPN", "HKG", "KOR", "TWN", "asi"])
def test_default_tu_bat_max_trade_cho_region_bat_buoc(region):
    c = SimConfig.default(region=region)
    assert c.max_trade == "ON"


@pytest.mark.parametrize("region", ["USA", "EUR", "GLB", "CHN", "AMR"])
def test_default_khong_bat_max_trade_cho_region_khac(region):
    c = SimConfig.default(region=region)
    assert c.max_trade == "OFF"
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/test_sim_config.py -k default_tu_bat_max_trade -v`
Expected: FAIL — `assert 'OFF' == 'ON'` (region ASI/JPN/HKG/KOR/TWN chưa tự bật)

- [ ] **Step 3: Cài tối thiểu**

Trong `src/simulation/config.py`, sửa `default()`:

```python
    @classmethod
    def default(cls, region: str = "USA", universe: str = "TOP3000", delay: int = DEFAULT_DELAY) -> "SimConfig":
        max_trade = "ON" if region.upper() in REQUIRE_MAX_TRADE_REGIONS else "OFF"
        return cls(region=region, universe=universe, delay=delay, max_trade=max_trade)
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/test_sim_config.py -v`
Expected: PASS toàn bộ

- [ ] **Step 5: Chạy toàn bộ suite, xác nhận không vỡ gì**

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS hết, trừ 1 fail có sẵn không liên quan (`test_make_engine_postgres_backend`).
Lưu ý: các test khác dùng `SimConfig.default(region="USA", ...)` không bị ảnh hưởng vì "USA"
không nằm trong `REQUIRE_MAX_TRADE_REGIONS`.

- [ ] **Step 6: Commit**

```bash
git add src/simulation/config.py tests/test_sim_config.py
git commit -m "feat(simulation): SimConfig.default tu bat max_trade=ON cho ASI/JPN/HKG/KOR/TWN"
```

---

## Self-Review (đã chạy)

- **Spec coverage**: mục "Sub-project E" trong roadmap spec — việc 1 (thêm field) = Task 1;
  việc 2 (ràng buộc bắt buộc theo region) = Task 2; việc 3 (max_position khuyến nghị, không bắt
  buộc) — đã thêm field ở Task 1 nhưng KHÔNG tự động bật (đúng tinh thần "khuyến nghị" chứ
  không "bắt buộc" như max_trade — không cần task riêng).
- **Placeholder scan**: sạch, mọi step có code đầy đủ.
- **Type consistency**: `test_period`/`max_trade`/`max_position` cùng tên field xuyên
  `SimConfig`, `to_settings()` (camelCase đúng key WQ dùng, đối chiếu `SIM_DEFAULTS`), `key()`.
