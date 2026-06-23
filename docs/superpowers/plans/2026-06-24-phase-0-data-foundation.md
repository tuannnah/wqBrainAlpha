# Phase 0 — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development hoặc superpowers:executing-plans. Steps dùng checkbox (`- [ ]`).

**Goal:** Dựng nền dữ liệu cho MiniBrain — `MarketData` panel bất biến `(T,N)`, port
`MarketDataSource`, adapter Parquet, universe mask per-day + sector groups, config/thresholds,
và fixture panel thật-hình-dạng cho mọi test sau.

**Architecture:** Thêm vào `src/data/` (tái dùng `WQBrainClient` sẵn có). Panel là
`dict[str, np.ndarray(T,N)]` + axes dates/assets; out-of-universe = NaN. Adapter đọc Parquet;
`market_fetch` kéo từ WQ Brain (probe — rủi ro Gap #3).

**Tech Stack:** numpy 2.4, pandas 3.0, pyarrow 24, pydantic-settings, pytest.

## Global Constraints

(Kế thừa master plan) Python 3.12; full type hints; `mypy --strict`/`ruff` clean; out-of-universe=NaN;
no look-ahead; thresholds chỉ ở `config/thresholds.py`; TDD; nhánh `phase-0-data-foundation`.

---

### Task 0.1: Tạo nhánh

**Files:** —

- [ ] **Step 1: Tạo nhánh từ main sạch**

```bash
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b phase-0-data-foundation
git status
```
Expected: "On branch phase-0-data-foundation", working tree clean (file rác đã xóa ở commit trước).

---

### Task 0.2: `config/thresholds.py`

**Files:**
- Create: `config/thresholds.py`
- Test: `tests/unit/test_thresholds.py`

**Interfaces:**
- Produces: module constants `MAX_DEPTH:int`, `SELF_CORR_MAX:float`, `TURNOVER_FLOOR:float`,
  `WEIGHT_CONCENTRATION_CAP:float`, `SHARPE_MIN:float`, `PER_YEAR_SHARPE_MIN:float`,
  `TURNOVER_BAND:tuple[float,float]`, `CALIBRATION_RHO_BAR:float`.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_thresholds.py
from config import thresholds


def test_threshold_values_present_and_sane():
    assert thresholds.MAX_DEPTH == 7
    assert thresholds.SELF_CORR_MAX == 0.70
    assert thresholds.TURNOVER_FLOOR == 0.125
    assert 0.0 < thresholds.WEIGHT_CONCENTRATION_CAP <= 1.0
    assert thresholds.CALIBRATION_RHO_BAR == 0.5
    lo, hi = thresholds.TURNOVER_BAND
    assert 0.0 <= lo < hi
```

- [ ] **Step 2: Chạy test — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_thresholds.py -v`
Expected: FAIL `ModuleNotFoundError: config.thresholds`.

- [ ] **Step 3: Tạo file**

```python
# config/thresholds.py
"""Mọi ngưỡng gate/submission của MiniBrain ở MỘT nơi (Gap #7/R9 master spec).

Không hardcode các số này ở call site — đổi theo cuộc thi thì sửa tại đây.
"""

from __future__ import annotations

# --- Cấu trúc biểu thức ---
MAX_DEPTH: int = 7  # trần độ sâu AST (gồm wrapper config khi đếm cho gate)

# --- Self-correlation (cổng chặn submission thật sự) ---
SELF_CORR_MAX: float = 0.70  # PnL self-corr >= ngưỡng này -> hard fail

# --- Metrics ---
TURNOVER_FLOOR: float = 0.125  # sàn turnover trong công thức fitness
WEIGHT_CONCENTRATION_CAP: float = 0.10  # |weight| 1 mã tối đa (gate tập trung)
SHARPE_MIN: float = 1.0  # sàn Sharpe (soft score)
PER_YEAR_SHARPE_MIN: float = 0.0  # sàn Sharpe năm tệ nhất (regime robustness)
TURNOVER_BAND: tuple[float, float] = (0.01, 0.70)  # dải turnover hợp lệ (soft)

# --- Calibration (tin cậy cả tool) ---
CALIBRATION_RHO_BAR: float = 0.5  # Spearman ρ tối thiểu để tin ranking local
```

- [ ] **Step 4: Chạy test — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_thresholds.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/thresholds.py tests/unit/test_thresholds.py
git commit -m "feat(config): thresholds tập trung cho MiniBrain gates"
```

---

### Task 0.3: Mở rộng `config/settings.py`

**Files:**
- Modify: `config/settings.py`
- Test: `tests/unit/test_settings_market.py`

**Interfaces:**
- Produces: `settings.market_data_dir:str` (default `"data/market"`), `settings.global_seed:int` (default `42`).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_settings_market.py
from config.settings import Settings


def test_market_settings_defaults():
    s = Settings()
    assert s.market_data_dir == "data/market"
    assert s.global_seed == 42
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_settings_market.py -v`
Expected: FAIL `AttributeError: market_data_dir`.

- [ ] **Step 3: Thêm field**

Trong `config/settings.py`, sau dòng `default_delay: int = 1` thêm:

```python
    # --- MiniBrain (local backtester) ---
    market_data_dir: str = "data/market"  # nơi chứa parquet panel cục bộ
    global_seed: int = 42  # seed toàn cục (determinism, R8)
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_settings_market.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/unit/test_settings_market.py
git commit -m "feat(config): settings market_data_dir + global_seed"
```

---

### Task 0.4: Type aliases

**Files:**
- Create: `src/local_types.py`
- Test: `tests/unit/test_local_types.py`

**Interfaces:**
- Produces: `Panel`, `Mask`, `Dates`, `Assets` (numpy.typing aliases).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_local_types.py
import numpy as np

from src.local_types import Assets, Dates, Mask, Panel


def test_aliases_usable_as_annotations():
    p: Panel = np.zeros((2, 3), dtype=np.float64)
    m: Mask = np.ones((2, 3), dtype=np.bool_)
    d: Dates = np.array(["2020-01-01"], dtype="datetime64[D]")
    a: Assets = np.array(["AAPL"], dtype=np.str_)
    assert p.shape == (2, 3) and m.dtype == np.bool_ and d.dtype.kind == "M" and a.dtype.kind == "U"
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_local_types.py -v`
Expected: FAIL `ModuleNotFoundError: src.local_types`.

- [ ] **Step 3: Tạo file**

```python
# src/local_types.py
"""Type alias dùng chung cho tầng local backtester (panel (T,N), mask, axes)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

type Panel = npt.NDArray[np.float64]  # (T, N); NaN = missing / ngoài universe
type Mask = npt.NDArray[np.bool_]  # (T, N); True = trong universe ngày đó
type Dates = npt.NDArray[np.datetime64]  # (T,)
type Assets = npt.NDArray[np.str_]  # (N,)
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_local_types.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_types.py tests/unit/test_local_types.py
git commit -m "feat(local): type aliases Panel/Mask/Dates/Assets"
```

---

### Task 0.5: `MarketData` panel

**Files:**
- Create: `src/data/market_panel.py`
- Test: `tests/unit/test_market_panel.py`

**Interfaces:**
- Consumes: `Panel, Mask, Dates, Assets` từ `src.local_types`.
- Produces: `MarketData(dates, assets, fields, universe, returns, groups)` frozen+slots;
  `.field(name)->Panel`, `.years()->dict[int,slice]`; `__post_init__` validate shape/axis.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_market_panel.py
import numpy as np
import pytest

from src.data.market_panel import MarketData


def _toy() -> MarketData:
    dates = np.array(["2020-01-01", "2020-01-02", "2021-01-04"], dtype="datetime64[D]")
    assets = np.array(["AAA", "BBB"], dtype=np.str_)
    close = np.array([[1.0, 2.0], [1.1, np.nan], [1.2, 2.2]], dtype=np.float64)
    universe = np.array([[True, True], [True, False], [True, True]], dtype=np.bool_)
    returns = np.array([[np.nan, np.nan], [0.1, np.nan], [0.0909, 0.0]], dtype=np.float64)
    groups = {"sector": np.array([[0, 1], [0, 1], [0, 1]])}
    return MarketData(dates=dates, assets=assets, fields={"close": close},
                      universe=universe, returns=returns, groups=groups)


def test_field_returns_panel():
    md = _toy()
    assert md.field("close").shape == (3, 2)


def test_field_unknown_raises():
    md = _toy()
    with pytest.raises(KeyError):
        md.field("nope")


def test_years_slices_by_calendar_year():
    md = _toy()
    years = md.years()
    assert set(years) == {2020, 2021}
    assert md.dates[years[2020]].tolist() == np.array(
        ["2020-01-01", "2020-01-02"], dtype="datetime64[D]").tolist()


def test_post_init_rejects_shape_mismatch():
    dates = np.array(["2020-01-01"], dtype="datetime64[D]")
    assets = np.array(["AAA", "BBB"], dtype=np.str_)
    bad = np.zeros((2, 2), dtype=np.float64)  # T=2 nhưng dates T=1
    with pytest.raises(ValueError):
        MarketData(dates=dates, assets=assets, fields={"close": bad},
                   universe=np.ones((1, 2), dtype=np.bool_),
                   returns=np.zeros((1, 2), dtype=np.float64), groups={})
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_market_panel.py -v`
Expected: FAIL `ModuleNotFoundError: src.data.market_panel`.

- [ ] **Step 3: Tạo file**

```python
# src/data/market_panel.py
"""Panel thị trường bất biến: các mảng field/universe/returns cùng trục (T,N).

MarketData là nguồn dữ liệu duy nhất cho Evaluator. Out-of-universe = NaN (không phải 0);
universe là mask per-day (an toàn look-ahead/survivorship).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.local_types import Assets, Dates, Mask, Panel


@dataclass(frozen=True, slots=True)
class MarketData:
    dates: Dates  # (T,)
    assets: Assets  # (N,)
    fields: dict[str, Panel]  # name -> (T, N)
    universe: Mask  # (T, N)
    returns: Panel  # (T, N) close-to-close simple returns
    groups: dict[str, np.ndarray]  # "sector" -> (T, N) int codes

    def __post_init__(self) -> None:
        t, n = len(self.dates), len(self.assets)
        shape = (t, n)
        for name, arr in self.fields.items():
            if arr.shape != shape:
                raise ValueError(f"field {name!r} shape {arr.shape} != {shape}")
        if self.universe.shape != shape:
            raise ValueError(f"universe shape {self.universe.shape} != {shape}")
        if self.returns.shape != shape:
            raise ValueError(f"returns shape {self.returns.shape} != {shape}")
        for name, arr in self.groups.items():
            if arr.shape != shape:
                raise ValueError(f"group {name!r} shape {arr.shape} != {shape}")

    def field(self, name: str) -> Panel:
        """Mảng (T,N) của field; KeyError nếu không có."""
        return self.fields[name]

    def years(self) -> dict[int, slice]:
        """Slice hàng theo từng năm dương lịch (cho per-year Sharpe)."""
        yrs = self.dates.astype("datetime64[Y]").astype(int) + 1970
        out: dict[int, slice] = {}
        for y in np.unique(yrs):
            idx = np.nonzero(yrs == y)[0]
            out[int(y)] = slice(int(idx[0]), int(idx[-1]) + 1)
        return out
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_market_panel.py -v`
Expected: PASS (4 test).

- [ ] **Step 5: Commit**

```bash
git add src/data/market_panel.py tests/unit/test_market_panel.py
git commit -m "feat(data): MarketData panel bất biến + validate shape/years"
```

---

### Task 0.6: `MarketDataSource` port

**Files:**
- Create: `src/data/market_source.py`
- Test: `tests/unit/test_market_source.py`

**Interfaces:**
- Consumes: `MarketData`.
- Produces: `MarketDataSource` Protocol — `load(start:str,end:str,universe:str="TOP3000")->MarketData`,
  `available_fields()->list[str]`.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_market_source.py
import numpy as np

from src.data.market_panel import MarketData
from src.data.market_source import MarketDataSource


class _Fake:
    def load(self, start: str, end: str, universe: str = "TOP3000") -> MarketData:
        d = np.array(["2020-01-01"], dtype="datetime64[D]")
        a = np.array(["AAA"], dtype=np.str_)
        z = np.zeros((1, 1), dtype=np.float64)
        return MarketData(dates=d, assets=a, fields={"close": z},
                          universe=np.ones((1, 1), dtype=np.bool_), returns=z, groups={})

    def available_fields(self) -> list[str]:
        return ["close"]


def test_fake_satisfies_protocol():
    src: MarketDataSource = _Fake()
    md = src.load("2020-01-01", "2020-01-02")
    assert md.field("close").shape == (1, 1)
    assert src.available_fields() == ["close"]
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_market_source.py -v`
Expected: FAIL `ModuleNotFoundError: src.data.market_source`.

- [ ] **Step 3: Tạo file**

```python
# src/data/market_source.py
"""Port (Protocol) nguồn dữ liệu thị trường — MiniBrain phụ thuộc cái NÀY, không phụ thuộc
feed cụ thể. Adapter chịu trách nhiệm PIT correctness, lịch sử universe, quy ước delay."""

from __future__ import annotations

from typing import Protocol

from src.data.market_panel import MarketData


class MarketDataSource(Protocol):
    def load(self, start: str, end: str, universe: str = "TOP3000") -> MarketData: ...

    def available_fields(self) -> list[str]: ...
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_market_source.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/market_source.py tests/unit/test_market_source.py
git commit -m "feat(data): MarketDataSource port (Protocol)"
```

---

### Task 0.7: `universe.py` — mask per-day + sector groups

**Files:**
- Create: `src/data/universe.py`
- Test: `tests/unit/test_universe.py`

**Interfaces:**
- Consumes: `Panel, Mask`.
- Produces: `build_universe_mask(valid: Panel) -> Mask` (True khi giá trị hữu hạn & >0 volume...);
  `sector_codes(raw_sector: np.ndarray, assets) -> np.ndarray` (int codes (T,N)).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_universe.py
import numpy as np

from src.data.universe import build_universe_mask, sector_codes


def test_mask_true_only_for_finite_positive():
    vol = np.array([[10.0, 0.0], [np.nan, 5.0]], dtype=np.float64)
    mask = build_universe_mask(vol)
    assert mask.tolist() == [[True, False], [False, True]]


def test_mask_changes_per_day_no_survivorship():
    vol = np.array([[1.0, 1.0], [1.0, np.nan]], dtype=np.float64)
    mask = build_universe_mask(vol)
    # mã thứ 2 rời universe ngày 2 -> mask per-day khác nhau
    assert mask[0].tolist() == [True, True]
    assert mask[1].tolist() == [True, False]


def test_sector_codes_dense_ints():
    raw = np.array([["10", "20"], ["10", "30"]], dtype=object)
    codes = sector_codes(raw)
    assert codes.shape == (2, 2)
    assert codes.dtype.kind in ("i", "u")
    # mã cùng sector "10" -> cùng code
    assert codes[0, 0] == codes[1, 0]
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_universe.py -v`
Expected: FAIL `ModuleNotFoundError: src.data.universe`.

- [ ] **Step 3: Tạo file**

```python
# src/data/universe.py
"""Xây universe mask per-day (an toàn survivorship) và mã sector dạng int dày đặc."""

from __future__ import annotations

import numpy as np

from src.local_types import Mask, Panel


def build_universe_mask(tradable: Panel) -> Mask:
    """True khi cell hữu hạn và > 0 (vd volume/giá hợp lệ). Mỗi ngày một mask riêng."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(tradable) & (tradable > 0.0)


def sector_codes(raw_sector: np.ndarray) -> np.ndarray:
    """Map nhãn sector (chuỗi) -> int code dày đặc, giữ shape (T,N). NaN/None -> -1."""
    flat = raw_sector.ravel()
    labels = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
              for v in flat]
    uniq = {lab: i for i, lab in enumerate(sorted({x for x in labels if x is not None}))}
    codes = np.array([uniq[lab] if lab is not None else -1 for lab in labels], dtype=np.int64)
    return codes.reshape(raw_sector.shape)
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_universe.py -v`
Expected: PASS (3 test).

- [ ] **Step 5: Commit**

```bash
git add src/data/universe.py tests/unit/test_universe.py
git commit -m "feat(data): universe mask per-day + sector codes"
```

---

### Task 0.8: `ParquetSource` adapter

**Files:**
- Create: `src/data/adapters/__init__.py`
- Create: `src/data/adapters/parquet_source.py`
- Test: `tests/unit/test_parquet_source.py`

**Interfaces:**
- Consumes: `MarketData`, `MarketDataSource`, settings.
- Produces: `ParquetSource(root: str)` thỏa `MarketDataSource`; `save(md, root)` ghi parquet để round-trip.

> Layout parquet: `root/fields/<name>.parquet` (index=dates, cols=assets), `root/universe.parquet`,
> `root/returns.parquet`, `root/groups/<g>.parquet`, `root/axes.parquet` (dates+assets).

- [ ] **Step 1: Viết test đỏ (round-trip)**

```python
# tests/unit/test_parquet_source.py
import numpy as np

from src.data.adapters.parquet_source import ParquetSource, save
from src.data.market_panel import MarketData


def _toy() -> MarketData:
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    assets = np.array(["AAA", "BBB"], dtype=np.str_)
    close = np.array([[1.0, 2.0], [1.1, np.nan]], dtype=np.float64)
    return MarketData(dates=dates, assets=assets, fields={"close": close},
                      universe=np.array([[True, True], [True, False]]),
                      returns=np.array([[np.nan, np.nan], [0.1, np.nan]]),
                      groups={"sector": np.array([[0, 1], [0, 1]])})


def test_round_trip(tmp_path):
    md = _toy()
    save(md, str(tmp_path))
    src = ParquetSource(str(tmp_path))
    out = src.load("2020-01-01", "2020-01-02")
    np.testing.assert_array_equal(out.dates, md.dates)
    np.testing.assert_array_equal(out.assets, md.assets)
    np.testing.assert_allclose(out.field("close"), md.field("close"), equal_nan=True)
    assert out.universe.tolist() == md.universe.tolist()
    assert "close" in src.available_fields()
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_parquet_source.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Tạo `__init__.py` rỗng + adapter**

```python
# src/data/adapters/__init__.py
"""Adapter cụ thể cho MarketDataSource."""
```

```python
# src/data/adapters/parquet_source.py
"""Adapter Parquet cho MarketDataSource: đọc/ghi panel (T,N) ra đĩa để tái dùng.

Layout: root/axes.parquet (dates,assets), root/fields/<name>.parquet,
root/universe.parquet, root/returns.parquet, root/groups/<g>.parquet.
Mỗi bảng field: index=dates, columns=assets, value=float64 (NaN giữ nguyên).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.market_panel import MarketData


def _to_frame(arr: np.ndarray, dates: np.ndarray, assets: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(arr, index=pd.Index(dates, name="date"), columns=list(assets))


def save(md: MarketData, root: str) -> None:
    """Ghi MarketData ra parquet partitioned dưới `root`."""
    base = Path(root)
    (base / "fields").mkdir(parents=True, exist_ok=True)
    (base / "groups").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": md.dates, "asset_idx": range(len(md.dates))}).to_parquet(
        base / "axes_dates.parquet")
    pd.DataFrame({"asset": md.assets}).to_parquet(base / "axes_assets.parquet")
    for name, arr in md.fields.items():
        _to_frame(arr, md.dates, md.assets).to_parquet(base / "fields" / f"{name}.parquet")
    _to_frame(md.universe.astype(np.int8), md.dates, md.assets).to_parquet(base / "universe.parquet")
    _to_frame(md.returns, md.dates, md.assets).to_parquet(base / "returns.parquet")
    for g, arr in md.groups.items():
        _to_frame(arr, md.dates, md.assets).to_parquet(base / "groups" / f"{g}.parquet")


class ParquetSource:
    """Đọc panel đã lưu. Thỏa Protocol MarketDataSource."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def available_fields(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "fields").glob("*.parquet"))

    def load(self, start: str, end: str, universe: str = "TOP3000") -> MarketData:
        assets = pd.read_parquet(self.root / "axes_assets.parquet")["asset"].to_numpy().astype(np.str_)
        fields: dict[str, np.ndarray] = {}
        dates_ref: np.ndarray | None = None
        for name in self.available_fields():
            df = pd.read_parquet(self.root / "fields" / f"{name}.parquet")
            mask = (df.index >= np.datetime64(start)) & (df.index <= np.datetime64(end))
            df = df.loc[mask]
            dates_ref = df.index.to_numpy().astype("datetime64[D]")
            fields[name] = df.to_numpy(dtype=np.float64)
        assert dates_ref is not None, "không có field nào để suy ra trục dates"

        def _load(name: str) -> np.ndarray:
            df = pd.read_parquet(self.root / name)
            m = (df.index >= np.datetime64(start)) & (df.index <= np.datetime64(end))
            return df.loc[m].to_numpy()

        universe_arr = _load("universe.parquet").astype(bool)
        returns_arr = _load("returns.parquet").astype(np.float64)
        groups: dict[str, np.ndarray] = {}
        gdir = self.root / "groups"
        if gdir.exists():
            for p in gdir.glob("*.parquet"):
                df = pd.read_parquet(p)
                m = (df.index >= np.datetime64(start)) & (df.index <= np.datetime64(end))
                groups[p.stem] = df.loc[m].to_numpy()
        return MarketData(dates=dates_ref, assets=assets, fields=fields,
                          universe=universe_arr, returns=returns_arr, groups=groups)
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_parquet_source.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/adapters/ tests/unit/test_parquet_source.py
git commit -m "feat(data): ParquetSource adapter round-trip MarketData"
```

---

### Task 0.9: `market_fetch.py` — kéo WQ Brain → parquet (SPIKE có kiểm soát)

**Files:**
- Create: `src/data/market_fetch.py`
- Test: `tests/unit/test_market_fetch.py`

**Interfaces:**
- Consumes: `WQBrainClient` (`src.data.client`), `save` (0.8), `build_universe_mask`/`sector_codes` (0.7).
- Produces: `fetch_to_parquet(client, fields, start, end, universe, out_dir) -> str` (đường dẫn root);
  `_assemble_panel(raw_by_field, sector_raw) -> MarketData` (hàm thuần, test được không cần mạng).

> **Rủi ro Gap #3 (master spec):** WQ Brain KHÔNG cấp bulk OHLCV sạch cho TOP3000 qua một
> endpoint. Task này tách 2 phần: (a) `_assemble_panel` — logic ghép raw→MarketData, test
> bằng dữ liệu giả thuần (không mạng); (b) `fetch_to_parquet` — gọi client thật, chạy tay khi
> có phiên. Nếu probe endpoint thất bại, ghi rõ cách lấy data thực tế vào `PROGRESS.md` và để
> `fetch_to_parquet` raise `NotImplementedError` với thông điệp chỉ dẫn — KHÔNG giả vờ thành công.

- [ ] **Step 1: Viết test đỏ (chỉ test `_assemble_panel`, không mạng)**

```python
# tests/unit/test_market_fetch.py
import numpy as np

from src.data.market_fetch import _assemble_panel


def test_assemble_panel_builds_aligned_marketdata():
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    assets = np.array(["AAA", "BBB"], dtype=np.str_)
    raw = {
        "close": (dates, assets, np.array([[10.0, 20.0], [11.0, np.nan]])),
        "volume": (dates, assets, np.array([[100.0, 50.0], [0.0, 5.0]])),
    }
    sector_raw = np.array([["10", "20"], ["10", "20"]], dtype=object)
    md = _assemble_panel(raw, sector_raw, tradable_field="volume")
    assert md.field("close").shape == (2, 2)
    # universe = volume hữu hạn & >0
    assert md.universe.tolist() == [[True, True], [False, True]]
    # returns ngày 0 = NaN (không look-ahead)
    assert np.isnan(md.returns[0]).all()
    assert "sector" in md.groups
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_market_fetch.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Tạo file**

```python
# src/data/market_fetch.py
"""Kéo OHLCV + universe + sector từ WQ Brain → MarketData → parquet.

Tách logic ghép (thuần, test được) khỏi I/O mạng. Phần mạng (`fetch_to_parquet`) gọi
WQBrainClient thật; vì WQ không cấp bulk OHLCV sạch (Gap #3), endpoint thực tế phải probe khi
chạy tay — nếu chưa xác định, raise NotImplementedError có chỉ dẫn thay vì giả vờ thành công.
"""

from __future__ import annotations

import numpy as np

from src.data.adapters.parquet_source import save
from src.data.market_panel import MarketData
from src.data.universe import build_universe_mask, sector_codes

RawField = tuple[np.ndarray, np.ndarray, np.ndarray]  # (dates, assets, values(T,N))


def _simple_returns(close: np.ndarray) -> np.ndarray:
    """Close-to-close simple returns; hàng đầu = NaN (không look-ahead)."""
    prev = np.empty_like(close)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = (close - prev) / prev
    return ret


def _assemble_panel(
    raw_by_field: dict[str, RawField],
    sector_raw: np.ndarray,
    tradable_field: str = "volume",
) -> MarketData:
    """Ghép raw theo field thành MarketData căn trục. Giả định mọi field cùng (dates,assets)."""
    if not raw_by_field:
        raise ValueError("raw_by_field rỗng")
    dates, assets, _ = next(iter(raw_by_field.values()))
    fields = {name: vals for name, (_, _, vals) in raw_by_field.items()}
    tradable = fields[tradable_field]
    universe = build_universe_mask(tradable)
    returns = _simple_returns(fields["close"])
    groups = {"sector": sector_codes(sector_raw)}
    return MarketData(dates=dates.astype("datetime64[D]"),
                      assets=assets.astype(np.str_), fields=fields,
                      universe=universe, returns=returns, groups=groups)


def fetch_to_parquet(
    client,  # WQBrainClient
    fields: list[str],
    start: str,
    end: str,
    universe: str = "TOP3000",
    out_dir: str = "data/market",
) -> str:
    """Kéo `fields` cho cửa sổ [start,end] → ghi parquet, trả root path.

    CHƯA chốt endpoint bulk: probe khi chạy tay (xem docstring module). Tới khi xác định,
    nâng NotImplementedError có chỉ dẫn để không tạo data sai âm thầm.
    """
    raise NotImplementedError(
        "Endpoint bulk OHLCV của WQ Brain chưa được xác định (Gap #3). "
        "Probe API khi có phiên rồi điền logic; tạm thời nạp panel qua ParquetSource.save(). "
        "Ghi cách lấy data thực tế vào PROGRESS.md."
    )
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_market_fetch.py -v`
Expected: PASS (test `_assemble_panel`; `fetch_to_parquet` chưa test mạng).

- [ ] **Step 5: Commit**

```bash
git add src/data/market_fetch.py tests/unit/test_market_fetch.py
git commit -m "feat(data): assemble panel + khung fetch WQ->parquet (spike Gap#3)"
```

---

### Task 0.10: Fixture `small_panel`

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_fixture_small_panel.py`

**Interfaces:**
- Produces: pytest fixture `small_panel -> MarketData` (T=120, N=30, reproducible seed) cho mọi phase sau.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_fixture_small_panel.py
import numpy as np

from src.data.market_panel import MarketData


def test_small_panel_shape_and_determinism(small_panel: MarketData):
    assert isinstance(small_panel, MarketData)
    assert small_panel.field("close").shape == (120, 30)
    assert small_panel.universe.dtype == np.bool_
    # ngày 0 của returns là NaN (không look-ahead)
    assert np.isnan(small_panel.returns[0]).all()


def test_small_panel_has_out_of_universe_nan(small_panel: MarketData):
    # có ít nhất 1 cell ngoài universe để các phase sau test NaN-propagation
    assert (~small_panel.universe).any()
```

- [ ] **Step 2: Chạy — đỏ**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_fixture_small_panel.py -v`
Expected: FAIL `fixture 'small_panel' not found`.

- [ ] **Step 3: Append fixture vào `tests/conftest.py`**

Thêm vào cuối `tests/conftest.py`:

```python
# --- MiniBrain: panel nhỏ thật-hình-dạng cho test backtester (Phase 0+) ---
@pytest.fixture
def small_panel():
    """Panel (T=120, N=30) reproducible: close = random walk; volume dương;
    universe per-day (vài mã rời giữa kỳ); sector groups; returns close-to-close."""
    import numpy as np

    from src.data.market_panel import MarketData

    rng = np.random.default_rng(42)
    t, n = 120, 30
    dates = (np.datetime64("2020-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array([f"A{i:02d}" for i in range(n)], dtype=np.str_)
    steps = rng.normal(0.0, 0.02, size=(t, n))
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    volume = rng.uniform(1e5, 1e6, size=(t, n))
    universe = np.ones((t, n), dtype=bool)
    universe[: t // 2, -3:] = False  # 3 mã cuối chỉ vào universe nửa sau
    prev = np.empty_like(close)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = (close - prev) / prev
    sector = np.tile(np.arange(n) % 5, (t, 1)).astype(np.int64)
    return MarketData(dates=dates, assets=assets,
                      fields={"close": close, "volume": volume},
                      universe=universe, returns=returns, groups={"sector": sector})
```

- [ ] **Step 4: Chạy — xanh**

Run: `venv/Scripts/python.exe -m pytest tests/unit/test_fixture_small_panel.py -v`
Expected: PASS (2 test).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/unit/test_fixture_small_panel.py
git commit -m "test(data): fixture small_panel real-shaped reproducible"
```

---

### Task 0.11: Review + gate + merge

**Files:** —

- [ ] **Step 1: Full test + ruff**

```bash
venv/Scripts/python.exe -m pytest tests/unit -v
venv/Scripts/python.exe -m ruff check src/data src/local_types.py config tests/unit 2>&1 || true
```
Expected: tất cả test Phase 0 xanh; ruff không lỗi (sửa nếu có).

- [ ] **Step 2: Cập nhật PROGRESS.md** (skill session-journal — append entry Phase 0: done, quyết định, rủi ro Gap#3, next=Phase 1).

- [ ] **Step 3: Merge + push**

```bash
git checkout main
git merge --no-ff phase-0-data-foundation -m "merge: Phase 0 — data foundation"
git push origin main
```

**DoD Phase 0:** MarketData load + validate; universe per-day; returns reconcile; fixture
`small_panel` tồn tại; ParquetSource round-trip; rủi ro fetch (Gap#3) ghi rõ; test xanh.

---

## Self-review

- **Spec coverage:** config/thresholds (0.2), settings (0.3), types (0.4), MarketData (0.5),
  port (0.6), universe (0.7), adapter (0.8), fetch (0.9), fixture (0.10) — khớp Phase 0 master plan ✔
- **Placeholder scan:** không TBD; `fetch_to_parquet` raise có chỉ dẫn (cố ý, không phải placeholder) ✔
- **Type consistency:** `MarketData(dates,assets,fields,universe,returns,groups)` nhất quán mọi task; `save`/`ParquetSource`/`_assemble_panel` dùng đúng chữ ký ✔
