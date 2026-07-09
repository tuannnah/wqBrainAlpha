# Power Pool Theme Sim Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho tool sim alpha đúng ràng buộc Power Pool Theme hiện hành (USA/delay1/TOP1000/risk-neut/∉pv1) để nộp được Pure Power Pool, mặc định đọc theme từ lịch.

**Architecture:** Lớp theme-driven config mỏng. `power_pool_theme.py` parse tập neutralization cho phép + gate enforce; `SimConfig` mở rộng nhận risk-neut; `alt_data_seeds.py` map category→risk-neut giao với tập theme; một helper thuần biến theme hôm nay thành `SimConfig`; wiring ở `main.py` đọc theme làm mặc định; đường Regular giữ nguyên khi không có theme.

**Tech Stack:** Python 3, pytest, typer, dataclasses. Không thêm dependency.

## Global Constraints

- Code, comment, commit message, giao tiếp: **tiếng Việt** (giữ đủ dấu). TDD bắt buộc: đỏ→xanh, mỗi task ≥1 commit.
- KHÔNG bịa field/dataset/theme; theme data cập nhật thủ công (xem docstring `src/scoring/power_pool_theme.py`).
- Enum API neutralization rủi ro (verbatim từ docs): `SLOW`, `FAST`, `SLOW_AND_FAST`, `REVERSION_AND_MOMENTUM`, `STATISTICAL`, `CROWDING`.
- Map token theme→enum: slow→SLOW, fast→FAST, "slow and fast"→SLOW_AND_FAST, ram→REVERSION_AND_MOMENTUM, statistical→STATISTICAL, crowding→CROWDING.
- Theme tuần hiện tại: `date(2026,7,6)`–`date(2026,7,12)`, region=USA, delay=1, universe=TOP1000, datasets_excluded=("pv1",).
- Backward-compat: `matches_theme(...)` thêm tham số `neutralization` phải là keyword có default `None` (test cũ gọi không truyền vẫn chạy).

## File Structure

- `src/simulation/config.py` — mở rộng `VALID_NEUTRALIZATIONS` (Task 1).
- `src/scoring/power_pool_theme.py` — `parse_allowed_neutralizations`, field `allowed_neutralizations`, entry tuần hiện tại (Task 2), enforce neut trong `matches_theme` (Task 3), gate `check_theme_compliance` (Task 8).
- `src/generation/alt_data_seeds.py` — `pp_neutralization_for_expr`, `pp_neut_candidates` (Task 4).
- `src/app/power_pool_config.py` — MỚI: `resolve_theme_sim_config` (Task 5).
- `src/app/closed_loop_adapters.py` — `LocalTunerRefiner` dùng pp-neut khi có tập theme (Task 6).
- `main.py` — `_run_closed_loop_session` đọc theme làm mặc định (Task 7).
- Test: `tests/test_sim_config.py`, `tests/unit/test_power_pool_theme.py`, `tests/unit/test_alt_data_seeds.py`, `tests/unit/test_power_pool_config.py` (mới), `tests/unit/test_power_pool_flag.py`.

---

### Task 1: SimConfig chấp nhận neutralization rủi ro

**Files:**
- Modify: `src/simulation/config.py:18-26`
- Test: `tests/test_sim_config.py`

**Interfaces:**
- Produces: `SimConfig(neutralization="STATISTICAL")` (và 5 giá trị risk khác) không raise; `RISK_NEUTRALIZATIONS: frozenset[str]`.

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/test_sim_config.py`:

```python
import pytest

from src.simulation.config import RISK_NEUTRALIZATIONS, SimConfig


@pytest.mark.parametrize(
    "neut",
    ["STATISTICAL", "CROWDING", "REVERSION_AND_MOMENTUM", "SLOW", "FAST", "SLOW_AND_FAST"],
)
def test_simconfig_chap_nhan_risk_neutralization(neut):
    cfg = SimConfig(neutralization=neut)
    assert cfg.neutralization == neut
    assert cfg.to_settings()["neutralization"] == neut


def test_risk_neutralizations_du_6_gia_tri():
    assert RISK_NEUTRALIZATIONS == frozenset(
        {"STATISTICAL", "CROWDING", "REVERSION_AND_MOMENTUM", "SLOW", "FAST", "SLOW_AND_FAST"}
    )


def test_simconfig_van_tu_choi_neut_bay():
    with pytest.raises(ValueError):
        SimConfig(neutralization="KHONG_TON_TAI")
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `python -m pytest tests/test_sim_config.py -k risk -q`
Expected: FAIL (`ImportError: RISK_NEUTRALIZATIONS` hoặc `ValueError` cho STATISTICAL).

- [ ] **Step 3: Sửa `src/simulation/config.py`**

Thay khối `VALID_NEUTRALIZATIONS = {...}` (dòng 18-26) bằng:

```python
# Neutralization nhóm (group-based) — chia theo phân loại ngành/thị trường.
GROUP_NEUTRALIZATIONS = frozenset(
    {"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "COUNTRY", "EXCHANGE"}
)
# Neutralization rủi ro (risk-based) — bắt buộc cho Power Pool Theme; enum verbatim từ
# docs/worldquantbrain/docs/advanced-topics/{statistical,crowding,ram}-risk-neutralized-alphas.md.
RISK_NEUTRALIZATIONS = frozenset(
    {"STATISTICAL", "CROWDING", "REVERSION_AND_MOMENTUM", "SLOW", "FAST", "SLOW_AND_FAST"}
)
VALID_NEUTRALIZATIONS = GROUP_NEUTRALIZATIONS | RISK_NEUTRALIZATIONS
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `python -m pytest tests/test_sim_config.py -q`
Expected: PASS (toàn bộ file, gồm test cũ).

- [ ] **Step 5: Commit**

```bash
git add src/simulation/config.py tests/test_sim_config.py
git commit -m "feat(sim-config): SimConfig chap nhan neutralization rui ro cho Power Pool"
```

---

### Task 2: parse_allowed_neutralizations + entry tuần hiện tại

**Files:**
- Modify: `src/scoring/power_pool_theme.py:22-76`
- Test: `tests/unit/test_power_pool_theme.py`

**Interfaces:**
- Produces:
  - `parse_allowed_neutralizations(raw: str | None) -> frozenset[str]`
  - `PowerPoolThemeWeek.allowed_neutralizations: frozenset[str]` (default `frozenset()`)
  - entry lịch `date(2026,7,6)–date(2026,7,12)` với filter tuần hiện tại.

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/unit/test_power_pool_theme.py`:

```python
from datetime import date

from src.scoring.power_pool_theme import (
    parse_allowed_neutralizations,
    theme_for_date,
)


def test_parse_allowed_neutralizations_du_6_token():
    raw = "neutralization in (slow, fast, slow and fast, ram, statistical, crowding)"
    assert parse_allowed_neutralizations(raw) == frozenset(
        {"SLOW", "FAST", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING"}
    )


def test_parse_allowed_neutralizations_bo_token_la():
    raw = "neutralization in (statistical, khong_biet, crowding)"
    assert parse_allowed_neutralizations(raw) == frozenset({"STATISTICAL", "CROWDING"})


def test_parse_allowed_neutralizations_none_va_rong():
    assert parse_allowed_neutralizations(None) == frozenset()
    assert parse_allowed_neutralizations("region=USA & delay=1") == frozenset()


def test_theme_tuan_hien_tai_2026_07_09():
    week = theme_for_date(date(2026, 7, 9))
    assert week is not None
    assert week.region == "USA"
    assert week.delay == 1
    assert week.universe == "TOP1000"
    assert week.datasets_excluded == ("pv1",)
    assert week.allowed_neutralizations == frozenset(
        {"SLOW", "FAST", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING"}
    )
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `python -m pytest tests/unit/test_power_pool_theme.py -k "allowed or tuan_hien_tai" -q`
Expected: FAIL (`ImportError: parse_allowed_neutralizations`).

- [ ] **Step 3: Sửa `src/scoring/power_pool_theme.py`**

3a. Thêm bảng map + hàm parse ngay sau `import re` (sau dòng 19):

```python
# Map token tự do trong "neutralization in (...)" của theme -> enum API BRAIN (verbatim từ
# docs/worldquantbrain/docs/advanced-topics/*-risk-neutralized-alphas.md). Token lạ -> bỏ qua.
_NEUT_TOKEN_TO_API: dict[str, str] = {
    "slow": "SLOW",
    "fast": "FAST",
    "slow and fast": "SLOW_AND_FAST",
    "ram": "REVERSION_AND_MOMENTUM",
    "statistical": "STATISTICAL",
    "crowding": "CROWDING",
}


def parse_allowed_neutralizations(raw: str | None) -> frozenset[str]:
    """Từ cụm 'neutralization in (a, b, ...)' trả tập enum API cho phép. Token lạ bị bỏ (không
    đoán). Không có cụm / raw None -> frozenset() (nghĩa: theme không ràng buộc neutralization)."""
    if not raw:
        return frozenset()
    m = re.search(r"neutralization\s+in\s*\(([^)]*)\)", raw)
    if not m:
        return frozenset()
    out: set[str] = set()
    for tok in m.group(1).split(","):
        key = tok.strip().lower()
        if key in _NEUT_TOKEN_TO_API:
            out.add(_NEUT_TOKEN_TO_API[key])
    return frozenset(out)
```

3b. Thêm field vào `PowerPoolThemeWeek` (trong dataclass, sau `unparsed_constraints`):

```python
    allowed_neutralizations: frozenset[str] = frozenset()
```

3c. Thêm entry tuần hiện tại vào cuối `JUNE_JULY_2026_CALENDAR` (trước dấu `]`):

```python
    PowerPoolThemeWeek(
        date(2026, 7, 6), date(2026, 7, 12),
        region="USA", delay=1, universe="TOP1000",
        datasets_excluded=("pv1",),
        unparsed_constraints="neutralization in (slow, fast, slow and fast, ram, statistical, crowding)",
        allowed_neutralizations=parse_allowed_neutralizations(
            "neutralization in (slow, fast, slow and fast, ram, statistical, crowding)"
        ),
    ),
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `python -m pytest tests/unit/test_power_pool_theme.py -q`
Expected: PASS (gồm test cũ).

- [ ] **Step 5: Commit**

```bash
git add src/scoring/power_pool_theme.py tests/unit/test_power_pool_theme.py
git commit -m "feat(power-pool-theme): parse tap neutralization cho phep + entry tuan hien tai"
```

---

### Task 3: matches_theme enforce neutralization

**Files:**
- Modify: `src/scoring/power_pool_theme.py:91-108`
- Test: `tests/unit/test_power_pool_theme.py`

**Interfaces:**
- Consumes: `PowerPoolThemeWeek.allowed_neutralizations` (Task 2).
- Produces: `matches_theme(week, *, region, delay, universe, datasets_used, neutralization=None) -> tuple[bool, list[str]]`.

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/unit/test_power_pool_theme.py`:

```python
from src.scoring.power_pool_theme import matches_theme


def test_matches_theme_chan_khi_neut_ngoai_tap():
    week = theme_for_date(date(2026, 7, 9))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000",
        datasets_used={"option8"}, neutralization="SUBINDUSTRY",
    )
    assert ok is False
    assert any("neutralization" in r.lower() for r in reasons)


def test_matches_theme_cho_qua_khi_neut_trong_tap():
    week = theme_for_date(date(2026, 7, 9))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000",
        datasets_used={"option8"}, neutralization="STATISTICAL",
    )
    assert ok is True
    assert reasons == []


def test_matches_theme_khong_truyen_neut_thi_khong_chan_neut():
    week = theme_for_date(date(2026, 7, 9))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000", datasets_used={"option8"},
    )
    assert ok is True  # neutralization=None -> giữ tương thích ngược, không xét neut
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `python -m pytest tests/unit/test_power_pool_theme.py -k "neut and matches" -q`
Expected: FAIL (`matches_theme() got an unexpected keyword argument 'neutralization'`).

- [ ] **Step 3: Sửa `matches_theme` trong `src/scoring/power_pool_theme.py`**

Đổi chữ ký + thêm khối kiểm neut. Thay dòng 91-108:

```python
def matches_theme(
    week: PowerPoolThemeWeek, *, region: str, delay: int, universe: str,
    datasets_used: set[str], neutralization: str | None = None,
) -> tuple[bool, list[str]]:
    """Kiểm 1 alpha có khớp `week` không — CHỈ kiểm phần đã parse chắc chắn (region/delay/
    universe/datasets_excluded/allowed_neutralizations). Field None của `week` KHÔNG chặn.
    `neutralization=None` -> KHÔNG xét neut (tương thích ngược cho nơi gọi cũ)."""
    reasons: list[str] = []
    if week.region is not None and region.upper() != week.region.upper():
        reasons.append(f"region {region} != theme yêu cầu {week.region}")
    if week.delay is not None and delay != week.delay:
        reasons.append(f"delay {delay} != theme yêu cầu {week.delay}")
    if week.universe is not None and universe.upper() != week.universe.upper():
        reasons.append(f"universe {universe} != theme yêu cầu {week.universe}")
    excluded_used = datasets_used & set(week.datasets_excluded)
    if excluded_used:
        reasons.append(f"dùng dataset bị loại trừ theo theme: {sorted(excluded_used)}")
    if (
        neutralization is not None
        and week.allowed_neutralizations
        and neutralization.upper() not in week.allowed_neutralizations
    ):
        reasons.append(
            f"neutralization {neutralization} không thuộc tập theme cho phép "
            f"{sorted(week.allowed_neutralizations)}"
        )
    return (not reasons, reasons)
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `python -m pytest tests/unit/test_power_pool_theme.py -q`
Expected: PASS (gồm test cũ — chúng không truyền `neutralization` nên không bị ảnh hưởng).

- [ ] **Step 5: Commit**

```bash
git add src/scoring/power_pool_theme.py tests/unit/test_power_pool_theme.py
git commit -m "feat(power-pool-theme): matches_theme enforce neutralization theo tap theme"
```

---

### Task 4: pp_neutralization_for_expr + pp_neut_candidates

**Files:**
- Modify: `src/generation/alt_data_seeds.py:67-82`
- Test: `tests/unit/test_alt_data_seeds.py`

**Interfaces:**
- Consumes: `FieldCollector`, category helper `_is_option/_is_social/_is_analyst` (đã có).
- Produces:
  - `pp_neutralization_for_expr(expr: str, allowed: frozenset[str], registry=None) -> str`
  - `pp_neut_candidates(expr: str, allowed: frozenset[str], registry=None, sweep: bool = False) -> list[str]`

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/unit/test_alt_data_seeds.py`:

```python
import src.operators_local  # noqa: F401
from src.generation.alt_data_seeds import pp_neut_candidates, pp_neutralization_for_expr

_ALLOWED = frozenset(
    {"SLOW", "FAST", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING"}
)


def test_pp_neut_option_ra_statistical():
    expr = "ts_backfill(implied_volatility_call_30, 22)"
    assert pp_neutralization_for_expr(expr, _ALLOWED) == "STATISTICAL"


def test_pp_neut_social_ra_crowding():
    expr = "ts_mean(snt_social_value, 5)"
    assert pp_neutralization_for_expr(expr, _ALLOWED) == "CROWDING"


def test_pp_neut_fallback_khi_lua_chon_ngoai_allowed():
    # allowed KHÔNG có STATISTICAL -> rơi về phần tử đầu (sorted) của allowed
    allowed = frozenset({"CROWDING", "SLOW"})
    expr = "ts_backfill(implied_volatility_call_30, 22)"  # option -> muốn STATISTICAL
    assert pp_neutralization_for_expr(expr, allowed) == "CROWDING"  # sorted(["CROWDING","SLOW"])[0]


def test_pp_neut_candidates_mac_dinh_1x_va_sweep():
    expr = "ts_mean(snt_social_value, 5)"
    assert pp_neut_candidates(expr, _ALLOWED) == ["CROWDING"]
    assert pp_neut_candidates(expr, _ALLOWED, sweep=True) == sorted(_ALLOWED)
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `python -m pytest tests/unit/test_alt_data_seeds.py -k pp_neut -q`
Expected: FAIL (`ImportError: pp_neut_candidates`).

- [ ] **Step 3: Sửa `src/generation/alt_data_seeds.py`**

Thêm sau `neutralization_for_expr` (cuối file):

```python
# Map category dataset -> neutralization RỦI RO ưu tiên (Power Pool Theme chỉ cho risk-neut).
# Khác `neutralization_for_expr` (group-neut cho đường non-PP).
_PP_CATEGORY_DEFAULT = {
    "option": "STATISTICAL",
    "social": "CROWDING",
    "analyst": "SLOW",
}


def pp_neutralization_for_expr(expr: str, allowed: frozenset[str], registry=None) -> str:
    """Chọn 1 neutralization RỦI RO cho biểu thức alt-data theo category dataset, GIAO với tập
    `allowed` của theme. option→STATISTICAL, social/sentiment→CROWDING, analyst/fundamental→SLOW,
    price-derived/mặc định→REVERSION_AND_MOMENTUM/STATISTICAL. Lựa chọn không thuộc `allowed` ->
    phần tử đầu (sorted, ổn định) của `allowed`. `allowed` rỗng -> STATISTICAL (an toàn chung)."""
    reg = registry or default_registry()
    fields = FieldCollector(reg).visit(parse(expr))
    if any(_is_option(f) for f in fields):
        choice = _PP_CATEGORY_DEFAULT["option"]
    elif any(_is_social(f) for f in fields):
        choice = _PP_CATEGORY_DEFAULT["social"]
    elif any(_is_analyst(f) for f in fields):
        choice = _PP_CATEGORY_DEFAULT["analyst"]
    else:
        choice = "STATISTICAL"
    if not allowed:
        return choice
    if choice in allowed:
        return choice
    return sorted(allowed)[0]


def pp_neut_candidates(
    expr: str, allowed: frozenset[str], registry=None, sweep: bool = False
) -> list[str]:
    """Danh sách neutralization để refiner sim. Mặc định 1× (chỉ lựa chọn map theo category);
    `sweep=True` -> toàn bộ `allowed` (sorted ổn định) để quét con giữ config tốt nhất."""
    if sweep and allowed:
        return sorted(allowed)
    return [pp_neutralization_for_expr(expr, allowed, registry)]
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `python -m pytest tests/unit/test_alt_data_seeds.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/generation/alt_data_seeds.py tests/unit/test_alt_data_seeds.py
git commit -m "feat(alt-data): map category->risk-neut giao voi tap theme (pp_neut)"
```

---

### Task 5: resolve_theme_sim_config (helper thuần)

**Files:**
- Create: `src/app/power_pool_config.py`
- Test: `tests/unit/test_power_pool_config.py`

**Interfaces:**
- Consumes: `theme_for_date` (Task 2), `SimConfig` (Task 1).
- Produces: `resolve_theme_sim_config(base: SimConfig, on_date: date, calendar=None) -> ThemeResolution` với `ThemeResolution(sim_config: SimConfig, allowed_neutralizations: frozenset[str], theme, region: str, universe: str, warning: str | None)`.

- [ ] **Step 1: Viết test đỏ**

Tạo `tests/unit/test_power_pool_config.py`:

```python
from datetime import date

from src.app.power_pool_config import resolve_theme_sim_config
from src.simulation.config import SimConfig


def test_resolve_co_theme_override_region_universe_delay():
    base = SimConfig.default(region="USA", universe="TOP3000", delay=1)
    res = resolve_theme_sim_config(base, date(2026, 7, 9))
    assert res.theme is not None
    assert res.sim_config.universe == "TOP1000"
    assert res.sim_config.region == "USA"
    assert res.sim_config.delay == 1
    assert res.region == "USA"
    assert res.universe == "TOP1000"
    assert "STATISTICAL" in res.allowed_neutralizations
    assert res.warning is None


def test_resolve_khong_theme_giu_nguyen_va_canh_bao():
    base = SimConfig.default(region="USA", universe="TOP3000", delay=1)
    res = resolve_theme_sim_config(base, date(2026, 8, 15))  # ngoài lịch
    assert res.theme is None
    assert res.sim_config.universe == "TOP3000"  # giữ nguyên
    assert res.allowed_neutralizations == frozenset()
    assert res.warning is not None
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `python -m pytest tests/unit/test_power_pool_config.py -q`
Expected: FAIL (`ModuleNotFoundError: src.app.power_pool_config`).

- [ ] **Step 3: Tạo `src/app/power_pool_config.py`**

```python
"""Biến Power Pool Theme của một ngày thành SimConfig (override region/delay/universe) + tập
neutralization cho phép. Đọc theme làm MẶC ĐỊNH: có theme -> áp ràng buộc; không có -> giữ
config gốc + cảnh báo (đường Regular). Thuần logic, dễ test — wiring nằm ở main.py."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.scoring.power_pool_theme import PowerPoolThemeWeek, theme_for_date
from src.simulation.config import SimConfig


@dataclass(frozen=True)
class ThemeResolution:
    sim_config: SimConfig
    allowed_neutralizations: frozenset[str]
    theme: PowerPoolThemeWeek | None
    region: str
    universe: str
    warning: str | None


def resolve_theme_sim_config(
    base: SimConfig, on_date: date, calendar: list[PowerPoolThemeWeek] | None = None
) -> ThemeResolution:
    """Có theme cho `on_date` -> override region/delay/universe của `base` theo theme + trả tập
    allowed_neutralizations. Không có theme -> trả `base` nguyên vẹn, allowed rỗng, kèm warning."""
    week = theme_for_date(on_date, calendar)
    if week is None:
        return ThemeResolution(
            sim_config=base, allowed_neutralizations=frozenset(), theme=None,
            region=base.region, universe=base.universe,
            warning=(
                f"Không có Power Pool Theme cho {on_date} trong lịch — giữ config Regular "
                f"({base.region}/{base.universe}/delay={base.delay}). Cập nhật lịch nếu muốn "
                f"nộp Pure Power Pool (xem docstring src/scoring/power_pool_theme.py)."
            ),
        )
    overrides: dict = {}
    if week.region is not None:
        overrides["region"] = week.region
    if week.universe is not None:
        overrides["universe"] = week.universe
    if week.delay is not None:
        overrides["delay"] = week.delay
    sim_config = base.with_overrides(**overrides) if overrides else base
    return ThemeResolution(
        sim_config=sim_config,
        allowed_neutralizations=week.allowed_neutralizations,
        theme=week,
        region=sim_config.region,
        universe=sim_config.universe,
        warning=None,
    )
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `python -m pytest tests/unit/test_power_pool_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/app/power_pool_config.py tests/unit/test_power_pool_config.py
git commit -m "feat(power-pool): helper resolve_theme_sim_config (theme->SimConfig)"
```

---

### Task 6: LocalTunerRefiner dùng pp-neut khi có tập theme

**Files:**
- Modify: `src/app/closed_loop_adapters.py:17` (import), `:88-111` (__init__), `:144-158` (`_sim_direct`)
- Test: `tests/unit/test_power_pool_flag.py`

**Interfaces:**
- Consumes: `pp_neutralization_for_expr` (Task 4).
- Produces: `LocalTunerRefiner(..., pp_allowed_neutralizations: frozenset[str] = frozenset())`; khi non-empty, `_sim_direct` chọn neut qua `pp_neutralization_for_expr`, ngược lại giữ `neutralization_for_expr` cũ.

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/unit/test_power_pool_flag.py`:

```python
def test_sim_direct_dung_risk_neut_khi_co_tap_theme(monkeypatch):
    """Có pp_allowed_neutralizations -> nhánh alt-data sim với risk-neut (STATISTICAL cho option),
    KHÔNG dùng group-neut (SECTOR) như đường non-theme."""
    monkeypatch.setattr("src.backtest.sub_universe.sub_universe_ok", lambda *a, **kw: True)

    captured = {}

    class _SimGhi:
        def simulate(self, expr, settings=None):
            captured["neutralization"] = settings["neutralization"]
            return SimulationResult(
                expression=expr, alpha_id="wq-x", status="passed",
                sharpe=1.2, fitness=1.1, turnover=0.3, drawdown=0.1, raw={},
            )

    r = LocalTunerRefiner(
        simulator=_SimGhi(), repo=_RepoGia(), data=object(),
        local_config=PortfolioConfig(decay=4, truncation=0.08),
        sim_config=SimConfig.default(),
        pp_allowed_neutralizations=frozenset({"STATISTICAL", "CROWDING"}),
    )
    # ép _is_alt_data=True để đi nhánh sim thẳng
    monkeypatch.setattr(r, "_is_alt_data", lambda expr: True)
    r.refine_and_sim(_cand_gia("ts_backfill(implied_volatility_call_30, 22)"))
    assert captured["neutralization"] == "STATISTICAL"
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `python -m pytest tests/unit/test_power_pool_flag.py -k sim_direct_dung_risk -q`
Expected: FAIL (`unexpected keyword argument 'pp_allowed_neutralizations'`).

- [ ] **Step 3: Sửa `src/app/closed_loop_adapters.py`**

3a. Dòng 17 — thêm import `pp_neutralization_for_expr`:

```python
from src.generation.alt_data_seeds import (
    ALT_DATA_CORES,
    neutralization_for_expr,
    pp_neutralization_for_expr,
)
```

3b. `__init__` — thêm tham số (sau `calib_repo=None,` ở dòng 93) và gán:

```python
        max_pool_corr: float = 0.70, calib_repo=None,
        pp_allowed_neutralizations: frozenset[str] = frozenset(),
    ) -> None:
```

Thêm dòng gán (sau `self.calib_repo = calib_repo`):

```python
        # Tập neutralization theo Power Pool Theme (rỗng -> đường non-theme, dùng group-neut cũ).
        self.pp_allowed_neutralizations = pp_allowed_neutralizations
```

3c. `_sim_direct` — thay dòng 149 (`neut = neutralization_for_expr(expr, self.registry)`):

```python
        if self.pp_allowed_neutralizations:
            neut = pp_neutralization_for_expr(
                expr, self.pp_allowed_neutralizations, self.registry
            )
        else:
            neut = neutralization_for_expr(expr, self.registry)
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `python -m pytest tests/unit/test_power_pool_flag.py -q`
Expected: PASS (gồm test cũ).

- [ ] **Step 5: Commit**

```bash
git add src/app/closed_loop_adapters.py tests/unit/test_power_pool_flag.py
git commit -m "feat(refiner): nhanh alt-data dung risk-neut khi co tap Power Pool Theme"
```

---

### Task 7: Wiring main.py — đọc theme làm mặc định

**Files:**
- Modify: `main.py:675-712` (trong `_run_closed_loop_session`)
- Test: `tests/test_auto_command.py` (smoke import + gọi helper)

**Interfaces:**
- Consumes: `resolve_theme_sim_config` (Task 5), `LocalTunerRefiner(pp_allowed_neutralizations=...)` (Task 6).
- Produces: `_run_closed_loop_session` mặc định áp theme hôm nay lên `sim_config`, log rõ; không theme -> log warning, giữ Regular.

- [ ] **Step 1: Viết test đỏ (smoke helper)**

Thêm vào `tests/test_auto_command.py`:

```python
from datetime import date

from src.app.power_pool_config import resolve_theme_sim_config
from src.simulation.config import SimConfig


def test_wiring_theme_ap_top1000_cho_hom_nay_trong_lich():
    base = SimConfig.default(region="USA", universe="TOP3000", delay=1)
    res = resolve_theme_sim_config(base, date(2026, 7, 9))
    # Đây là hợp đồng main.py dựa vào: có theme -> TOP1000 + tập risk-neut không rỗng.
    assert res.sim_config.universe == "TOP1000"
    assert res.allowed_neutralizations
```

- [ ] **Step 2: Chạy test — kỳ vọng PASS ngay (helper đã có từ Task 5)**

Run: `python -m pytest tests/test_auto_command.py -k wiring_theme -q`
Expected: PASS. (Test này chốt hợp đồng; bước 3 nối vào main.py.)

- [ ] **Step 3: Sửa `main.py` trong `_run_closed_loop_session`**

3a. Thêm import trong thân hàm (cạnh các import cục bộ đầu hàm, sau dòng 666):

```python
    from datetime import date as _date

    from src.app.power_pool_config import resolve_theme_sim_config
```

3b. Ngay sau khối `cfg, sim_config = _closed_loop_configs(...)` (kết thúc dòng 678), chèn:

```python
    # MẶC ĐỊNH đọc Power Pool Theme hôm nay: có theme -> sim đúng region/universe/delay theme
    # (nộp được Pure Power Pool); không có -> giữ config Regular + cảnh báo.
    _res = resolve_theme_sim_config(sim_config, _date.today())
    pp_allowed = _res.allowed_neutralizations
    if _res.theme is not None:
        sim_config = _res.sim_config
        region, universe = _res.region, _res.universe
        console.print(
            f"[cyan]Power Pool Theme {_res.theme.start_date}..{_res.theme.end_date}: "
            f"sim {region}/{universe}/delay={sim_config.delay}, "
            f"neutralization ∈ {sorted(pp_allowed)}[/cyan]"
        )
    else:
        console.print(f"[yellow]{_res.warning}[/yellow]")
```

3c. Truyền `pp_allowed` vào `LocalTunerRefiner(...)` (khối dòng 705-712) — thêm dòng cuối trước `)`:

```python
            calib_repo=repo,
            pp_allowed_neutralizations=pp_allowed,
        )
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS + smoke import main**

Run: `python -m pytest tests/test_auto_command.py -q && python -c "import main"`
Expected: PASS và import main không lỗi.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_auto_command.py
git commit -m "feat(closed-loop): doc Power Pool Theme lam mac dinh khi sim vong kin"
```

---

### Task 8: Gate check_theme_compliance trước submit

**Files:**
- Modify: `src/scoring/power_pool_theme.py` (thêm hàm cuối file)
- Test: `tests/unit/test_power_pool_theme.py`

**Interfaces:**
- Consumes: `theme_for_date`, `matches_theme` (Task 2/3).
- Produces: `check_theme_compliance(*, region, delay, universe, neutralization, datasets_used, on_date, calendar=None) -> tuple[bool, list[str]]` — True nếu không có theme (không chặn) hoặc khớp; False + reasons nếu lệch.

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/unit/test_power_pool_theme.py`:

```python
from src.scoring.power_pool_theme import check_theme_compliance


def test_check_theme_compliance_khop():
    ok, reasons = check_theme_compliance(
        region="USA", delay=1, universe="TOP1000", neutralization="STATISTICAL",
        datasets_used={"option8"}, on_date=date(2026, 7, 9),
    )
    assert ok is True and reasons == []


def test_check_theme_compliance_lech_neut_va_universe():
    ok, reasons = check_theme_compliance(
        region="USA", delay=1, universe="TOP3000", neutralization="SUBINDUSTRY",
        datasets_used={"pv1"}, on_date=date(2026, 7, 9),
    )
    assert ok is False
    assert len(reasons) >= 2  # universe + neutralization (+ pv1)


def test_check_theme_compliance_khong_co_theme_khong_chan():
    ok, reasons = check_theme_compliance(
        region="USA", delay=1, universe="TOP3000", neutralization="SUBINDUSTRY",
        datasets_used={"pv1"}, on_date=date(2026, 8, 15),
    )
    assert ok is True and reasons == []
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `python -m pytest tests/unit/test_power_pool_theme.py -k check_theme_compliance -q`
Expected: FAIL (`ImportError: check_theme_compliance`).

- [ ] **Step 3: Thêm hàm cuối `src/scoring/power_pool_theme.py`**

```python
def check_theme_compliance(
    *, region: str, delay: int, universe: str, neutralization: str,
    datasets_used: set[str], on_date: date,
    calendar: list[PowerPoolThemeWeek] | None = None,
) -> tuple[bool, list[str]]:
    """Gate trước khi nộp Pure Power Pool: alpha (region/delay/universe/neutralization/datasets)
    có khớp theme của `on_date` không. Không có theme cho ngày đó -> (True, []) (không chặn ở
    đây; việc có nộp Pure Power Pool hay không do nơi gọi quyết). Lệch -> (False, reasons) để
    log rõ (tránh để Brain trả 'does not match any Power Pool Theme')."""
    week = theme_for_date(on_date, calendar)
    if week is None:
        return (True, [])
    return matches_theme(
        week, region=region, delay=delay, universe=universe,
        datasets_used=datasets_used, neutralization=neutralization,
    )
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `python -m pytest tests/unit/test_power_pool_theme.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scoring/power_pool_theme.py tests/unit/test_power_pool_theme.py
git commit -m "feat(power-pool-theme): gate check_theme_compliance truoc submit"
```

---

### Task 9: Regression — chạy full suite liên quan

**Files:** (không sửa; chỉ chạy)

- [ ] **Step 1: Chạy các suite bị đụng**

Run:
```bash
python -m pytest tests/test_sim_config.py tests/unit/test_power_pool_theme.py \
  tests/unit/test_alt_data_seeds.py tests/unit/test_power_pool_config.py \
  tests/unit/test_power_pool_flag.py tests/test_auto_command.py -q
```
Expected: PASS toàn bộ.

- [ ] **Step 2: Smoke toàn repo (nhanh)**

Run: `python -m pytest -q -x`
Expected: PASS (hoặc chỉ các fail đã có từ trước không liên quan — nếu có, ghi lại, không sửa ngoài phạm vi).

- [ ] **Step 3: Commit (nếu có chỉnh sửa vặt do regression)**

```bash
git add -A && git commit -m "test: xanh full suite Power Pool Theme sim config"
```

---

## Self-Review

**Spec coverage:**
- Component 1 (SimConfig risk-neut) → Task 1 ✅
- Component 2 (parse allowed + calendar entry + matches_theme enforce) → Task 2 + Task 3 ✅
- Component 3 (pp_neutralization_for_expr + pp_neut_candidates) → Task 4 ✅
- Component 4 (theme-driven config mặc định + refiner dùng risk-neut) → Task 5 + Task 6 + Task 7 ✅
- Component 5 (gate trước submit) → Task 8 ✅
- Kiểm thử TDD từng component → mỗi task có test đỏ→xanh ✅

**Placeholder scan:** không có TBD/TODO; mọi step có code/command cụ thể.

**Type consistency:** `allowed_neutralizations: frozenset[str]` nhất quán Task 2→3→5→6; `pp_neutralization_for_expr(expr, allowed, registry)` khớp Task 4↔6; `resolve_theme_sim_config`/`ThemeResolution` khớp Task 5↔7; `matches_theme(..., neutralization=None)` khớp Task 3↔8.
