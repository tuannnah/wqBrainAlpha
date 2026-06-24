# Phase 9 — Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> (khuyến nghị) hoặc superpowers:executing-plans để thực thi từng task. Task 9.1 (logic
> calibration view) và Task 9.2 (logic pool heatmap) độc lập nhau (file khác nhau, không đụng
> chung) → có thể chia 2 sub-agent song song. Task 9.3 (export shortlist — logic thuần)
> nên làm sau hoặc song song với 9.1/9.2 (độc lập về file: `src/pipeline/shortlist_export.py`
> mới). Task 9.4 (render Streamlit — thêm 3 tab vào `dashboard/app.py`) **phải chạy sau**
> 9.1–9.3 vì nó import các hàm thuần từ cả ba. Task 9.5 (submit helper logic thuần) độc lập
> file, có thể làm song song 9.1–9.3. Task 9.6 (review + merge + push) luôn cuối cùng, tuần
> tự.

**Goal:** Thêm vào dashboard Streamlit sẵn có (`dashboard/app.py`) ba khả năng mới của
MiniBrain — (1) xem báo cáo calibration (Spearman ρ sharpe/fitness + theo năm) từ
`CalibrationReport` (Phase 4.5), (2) xem bản đồ self-correlation của pool alpha đã pass
(Phase 6 `PoolCorrelation`/`load_pool`), (3) xuất shortlist (Phase 8) ra file để đem đi sim
Brain, kèm submit helper tùy chọn dùng client/login sẵn có (không hardcode secret) — **mà
không phá** 5 tab cũ (`Overview/Explorer/Tiến trình/Submissions/Correlation`). Vì UI
Streamlit khó unit-test, toàn bộ logic được tách thành hàm thuần (`build_calibration_view`,
`build_pool_heatmap_data`, `export_shortlist`, `build_submit_payload`) sống ở `src/` (không
phải `dashboard/`), có TDD đầy đủ; phần render trong `dashboard/app.py` chỉ gọi các hàm này
và hiển thị — verify bằng smoke test thủ công (`streamlit run`), không unit test UI.

**Architecture:** Logic thuần đặt tại `src/dashboard_logic.py` (calibration view +
pool heatmap — đọc-chỉ, không phụ thuộc `streamlit`) và `src/pipeline/shortlist_export.py`
(export + submit payload — mở rộng package `src/pipeline` đã có từ Phase 8, cùng tầng với
`shortlist.py`/`runner.py`). `src/dashboard_logic.py` import `src/calibration/report.py`
(Phase 4.5: `CalibrationReport`) — chỉ kiểu dữ liệu, không gọi lại harness — và nhận
`pool: dict[int, tuple[Dates, Panel1D]]` (kiểu trả về của `AlphaRepository.load_pool()`,
Phase 5/6) làm tham số, **không** tự mở DB. `dashboard/app.py` là nơi **duy nhất** được phép
import `streamlit` — toàn bộ logic thuần test được mà không cần streamlit cài/sandbox UI.
Dependency rule giữ nguyên: `dashboard/` (app layer) phụ thuộc `src/dashboard_logic.py` +
`src/pipeline/shortlist_export.py` + `src/storage/repository.py` + `src/calibration/*` —
một chiều, không có cạnh ngược.

**Lưu ý phụ thuộc-tới-trước (forward dependency):** Tại thời điểm viết plan này, **Phase 5
(Database mở rộng), Phase 6 (Pool correlation), Phase 7 (GP), Phase 8 (Shortlist+CLI) đã có
plan chi tiết** (`docs/superpowers/plans/2026-06-24-phase-{4.5,5,6}-*.md` tồn tại; Phase 7/8
theo master plan nhưng **Phase 8 chưa có file plan step-by-step riêng** — chỉ mục master
plan). Phase 9 **giả định các chữ ký sau đây** (khớp B9/B10/B11 + các quyết định đã ghi
trong Phase 4.5/6 plan), và nếu lúc thực thi Phase 9 mà thực tế khác, người thực thi phải
sửa import ở `src/dashboard_logic.py`/`shortlist_export.py` cho khớp **trước khi** viết
test (không đổi hợp đồng hàm thuần `build_*`/`export_*` — chúng nhận tham số đã là kiểu dữ
liệu thuần, tách khỏi nguồn DB cụ thể):

- `src.calibration.report.CalibrationReport` — `n: int`, `spearman_sharpe: float`,
  `spearman_fitness: float`, `self_corr_agreement: float`, `decile_hit_rate: float`,
  `by_year: dict[int, float]` (đã chốt ở Phase 4.5 plan, dòng 610-617).
- `src.storage.repository.AlphaRepository.load_pool() -> dict[int, tuple[Dates,
  npt.NDArray[np.float64]]]` (đã chốt ở Phase 6 plan Task 6.1/6.3, dòng 576-577) — key là
  `evaluation_id`.
- `src.pipeline.shortlist.shortlist(...) -> list[...]` (Phase 8, chưa có chữ ký step-by-step
  cố định) — Task 9.3 ở plan này **không** gọi `shortlist()` trực tiếp; nó nhận
  `items: Sequence[ShortlistRow]` đã được tầng gọi (CLI/dashboard) chuẩn bị, với
  `ShortlistRow` là `Protocol` tối thiểu định nghĩa tại `shortlist_export.py` (xem Task 9.3)
  — cách này tách Phase 9 khỏi việc phải biết chính xác kiểu trả về nội bộ của
  `pipeline.shortlist`, chỉ cần nó có các trường `expr_string`, `sharpe`, `fitness`,
  `turnover`, `self_corr_max` (duck-typing qua Protocol, không ép import ngược).

**Tech Stack:** Python 3.12, streamlit (đã trong `requirements.txt`), pandas, numpy, pytest,
ruff, mypy --strict. Không thêm dependency mới (không cần `plotly`/`matplotlib` — heatmap
hiển thị qua `st.dataframe` với `pandas.DataFrame.style.background_gradient` hoặc
`st.dataframe` thường, vì dữ liệu trả về là ma trận số — đủ cho MVP).

## Global Constraints

- Python 3.12; cú pháp hiện đại (`match`, `X | None`, `type` alias, `@dataclass(frozen=True, slots=True)`, `Protocol`).
- Full type hints; `mypy --strict` clean; `ruff` clean; không unused import.
- **No look-ahead:** time-series ops chỉ đọc rows ≤ t; thiếu lịch sử → NaN.
- **No survivorship:** universe mask per-day; out-of-universe = NaN (không phải 0).
- **Delay-1:** `pnl_t = nansum(weights_{t-1} * returns_t)`.
- **Stage separation:** expression = signal core; neut/decay/trunc/scale/delay ở `PortfolioConfig`.
- **Thresholds chỉ ở `config/thresholds.py`** — không hardcode gate number ở call site.
- **Determinism:** randomness qua seed inject; ghi seed vào DB.
- **WQ operator fidelity:** tra skill `worldquant-brain` trước khi viết FASTEXPR/operator.
- **TDD:** test trước, đỏ → code tối thiểu → xanh → commit. Mỗi phase = 1 nhánh git → merge → push.
- **Per-phase ritual:** Design → Implement → Explain → Review (test+ruff+mypy) → Gate → Journal (`PROGRESS.md`).

## Pre-condition (đọc trước khi bắt đầu)

Kiểm tra các module Phase 9 tiêu thụ đã tồn tại; nếu chưa, Task tương ứng vẫn viết được với
fake/stub kiểu dữ liệu thuần (dataclass) trong test — **không** chặn Phase 9, vì hợp đồng
hàm thuần ở plan này nhận dữ liệu đã-tải (`CalibrationReport`, `dict[int, tuple[Dates,
ndarray]]`), không tự gọi DB/harness thật bên trong hàm `build_*`/`export_*`:

```bash
venv/Scripts/python.exe -c "from src.calibration.report import CalibrationReport; print('calib ok')"
venv/Scripts/python.exe -c "from src.storage.repository import AlphaRepository; print('repo ok')"
venv/Scripts/python.exe -c "import streamlit; print('streamlit', streamlit.__version__)"
```

- Nếu `calib ok`/`repo ok` lỗi `ModuleNotFoundError` → Phase 4.5/5/6 chưa merge vào `main`.
  Task 9.1/9.2 **vẫn làm được** vì test dùng `CalibrationReport`/tuple giả lập cùng hình
  dạng (dataclass tự định nghĩa trong test nếu import thật chưa có) — nhưng **không** xóa
  import thật khỏi `src/dashboard_logic.py`; nếu import thật chưa tồn tại lúc code, dừng lại
  báo cáo block thay vì viết lại `CalibrationReport` ở đây (đó là lấn phạm vi Phase 4.5).
- Nếu `streamlit` chưa cài: `venv/Scripts/pip.exe install -r requirements.txt`.

---

### Task 9.1: `build_calibration_view` — logic thuần cho tab Calibration

**Files:**
- Create: `src/dashboard_logic.py`
- Test: `tests/unit/test_dashboard_logic_calibration.py`

**Interfaces:**
- Consumes: `src.calibration.report.CalibrationReport` (Phase 4.5).
- Produces:
  ```python
  def build_calibration_view(report: CalibrationReport) -> pd.DataFrame: ...
  ```
  Trả về `DataFrame` 2 phần ghép dọc, cột `["metric", "value"]`:
  - 4 dòng đầu: `("n", report.n)`, `("spearman_sharpe", report.spearman_sharpe)`,
    `("spearman_fitness", report.spearman_fitness)`,
    `("self_corr_agreement", report.self_corr_agreement)`,
    `("decile_hit_rate", report.decile_hit_rate)` (5 dòng, đếm lại: n + 4 metric ρ/agreement
    = 5 dòng tổng).
  - Sau đó N dòng `("year_<năm>", spearman_sharpe_theo_năm)` lấy từ `report.by_year`, sort
    theo năm tăng dần (key `int`).
  - `NaN` (`math.isnan`) giữ nguyên là `float("nan")` trong cột `value` (không ép về 0) —
    để dashboard hiển thị rõ "chưa đủ dữ liệu" thay vì số giả.
  - `report.n == 0` → trả `DataFrame` chỉ có dòng `("n", 0)`, không có dòng ρ nào (tránh suy
    diễn từ rỗng).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_dashboard_logic_calibration.py
"""Test build_calibration_view: chuyen CalibrationReport thanh DataFrame hien thi duoc."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.calibration.report import CalibrationReport
from src.dashboard_logic import build_calibration_view


def _report(**overrides: object) -> CalibrationReport:
    base = dict(
        n=50,
        spearman_sharpe=0.62,
        spearman_fitness=0.55,
        self_corr_agreement=0.80,
        decile_hit_rate=0.40,
        by_year={2023: 0.5, 2022: 0.6},
    )
    base.update(overrides)
    return CalibrationReport(**base)  # type: ignore[arg-type]


def test_build_calibration_view_returns_dataframe_with_metric_value_columns() -> None:
    df = build_calibration_view(_report())

    assert list(df.columns) == ["metric", "value"]


def test_build_calibration_view_includes_headline_metrics() -> None:
    df = build_calibration_view(_report())
    by_metric = df.set_index("metric")["value"]

    assert by_metric["n"] == 50
    assert by_metric["spearman_sharpe"] == pytest.approx(0.62)
    assert by_metric["spearman_fitness"] == pytest.approx(0.55)
    assert by_metric["self_corr_agreement"] == pytest.approx(0.80)
    assert by_metric["decile_hit_rate"] == pytest.approx(0.40)


def test_build_calibration_view_by_year_sorted_ascending() -> None:
    df = build_calibration_view(_report())
    year_rows = df[df["metric"].str.startswith("year_")]

    assert list(year_rows["metric"]) == ["year_2022", "year_2023"]
    assert list(year_rows["value"]) == [pytest.approx(0.6), pytest.approx(0.5)]


def test_build_calibration_view_preserves_nan() -> None:
    df = build_calibration_view(_report(spearman_sharpe=float("nan")))
    by_metric = df.set_index("metric")["value"]

    assert math.isnan(by_metric["spearman_sharpe"])


def test_build_calibration_view_empty_report_has_only_n_row() -> None:
    df = build_calibration_view(_report(n=0, by_year={}))

    assert list(df["metric"]) == ["n"]
    assert df.set_index("metric")["value"]["n"] == 0
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_dashboard_logic_calibration.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.dashboard_logic'` (hoặc lỗi tương tự —
file chưa tồn tại).

- [ ] **Step 3: Code tối thiểu — `src/dashboard_logic.py` (phần calibration)**

```python
# src/dashboard_logic.py
"""Logic thuan cho dashboard Streamlit (KHONG import streamlit o day).

Tach rieng khoi dashboard/app.py de unit-test duoc: cac ham build_* nhan du lieu da
tai (CalibrationReport, pool tu load_pool()) va tra ve cau truc thuan (DataFrame /
ma tran) ma streamlit chi can render, khong tu mo DB / goi lai harness.
"""

from __future__ import annotations

import pandas as pd

from src.calibration.report import CalibrationReport


def build_calibration_view(report: CalibrationReport) -> pd.DataFrame:
    """Chuyen CalibrationReport thanh DataFrame 2 cot (metric, value) de hien thi.

    n=0 -> chi tra dong "n" (khong suy dien ro tu tap rong). Gia tri NaN giu nguyen
    de dashboard hien ro "chua du du lieu" thay vi so gia.
    """
    rows: list[tuple[str, float]] = [("n", float(report.n))]
    if report.n > 0:
        rows.extend(
            [
                ("spearman_sharpe", report.spearman_sharpe),
                ("spearman_fitness", report.spearman_fitness),
                ("self_corr_agreement", report.self_corr_agreement),
                ("decile_hit_rate", report.decile_hit_rate),
            ]
        )
        for year in sorted(report.by_year):
            rows.append((f"year_{year}", report.by_year[year]))
    return pd.DataFrame(rows, columns=["metric", "value"])
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_dashboard_logic_calibration.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Lint + type-check**

```bash
venv/Scripts/python.exe -m ruff check src/dashboard_logic.py tests/unit/test_dashboard_logic_calibration.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/dashboard_logic.py
```
Expected: cả hai sạch (test file có thể bỏ qua mypy --strict theo convention dự án nếu
`pyproject`/`mypy.ini` exclude `tests/` — kiểm tra; nếu không exclude, sửa cho strict-clean).

- [ ] **Step 6: Commit**

```bash
git add src/dashboard_logic.py tests/unit/test_dashboard_logic_calibration.py
git commit -m "feat(dashboard): build_calibration_view — DataFrame thuan tu CalibrationReport"
```

---

### Task 9.2: `build_pool_heatmap_data` — logic thuần cho tab Pool-corr

**Files:**
- Modify: `src/dashboard_logic.py`
- Test: `tests/unit/test_dashboard_logic_pool_heatmap.py`

**Interfaces:**
- Consumes: `pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]]` (kiểu trả về của
  `AlphaRepository.load_pool()`, Phase 5/6 — `Dates = npt.NDArray[np.datetime64]` từ
  `src.local_types`); dùng `PoolCorrelation.max_corr` (Phase 6, `src.backtest.pool_corr`)
  pairwise để dựng ma trận — **không** viết lại logic Pearson ở đây, tái dùng Phase 6.
- Produces:
  ```python
  def build_pool_heatmap_data(
      pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]],
  ) -> pd.DataFrame: ...
  ```
  Trả về `DataFrame` vuông `(n, n)` với `index`/`columns` = các `evaluation_id` (key của
  `pool`, sort tăng dần để hiển thị ổn định), giá trị `[i, j]` = `max|ρ|` Pearson giữa PnL
  của `i` và `j` trên ngày chung (dùng `PoolCorrelation(pool).max_corr(pnl_i, dates_i)` lấy
  riêng so với từng `j`, hoặc tính trực tiếp pairwise — chọn cách đơn giản và đúng: với mỗi
  cặp `(i, j)`, `i != j`, dựng `PoolCorrelation({j: pool[j]})` rồi gọi
  `.max_corr(pool[i][1], pool[i][0])` để lấy đúng giá trị ρ giữa `i` và `j` cụ thể, không
  phải max toàn pool). Đường chéo `[i, i] = 1.0`. Pool rỗng hoặc 1 phần tử → `DataFrame`
  rỗng/`(1,1)` với giá trị `1.0` tương ứng — không lỗi.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_dashboard_logic_pool_heatmap.py
"""Test build_pool_heatmap_data: ma tran self-correlation pool tu load_pool()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dashboard_logic import build_pool_heatmap_data


def _dates(n: int) -> np.ndarray:
    return np.array(
        [np.datetime64("2023-01-01") + np.timedelta64(i, "D") for i in range(n)]
    )


def test_build_pool_heatmap_data_empty_pool_returns_empty_dataframe() -> None:
    df = build_pool_heatmap_data({})

    assert df.empty


def test_build_pool_heatmap_data_single_entry_diagonal_one() -> None:
    dates = _dates(10)
    pool = {1: (dates, np.random.default_rng(0).normal(size=10))}

    df = build_pool_heatmap_data(pool)

    assert df.shape == (1, 1)
    assert df.loc[1, 1] == pytest.approx(1.0)


def test_build_pool_heatmap_data_perfectly_correlated_pair() -> None:
    dates = _dates(20)
    base = np.random.default_rng(1).normal(size=20)
    pool = {1: (dates, base), 2: (dates, base * 2.0)}  # ty le -> rho = 1.0

    df = build_pool_heatmap_data(pool)

    assert df.shape == (2, 2)
    assert df.loc[1, 1] == pytest.approx(1.0)
    assert df.loc[2, 2] == pytest.approx(1.0)
    assert df.loc[1, 2] == pytest.approx(1.0, abs=1e-6)
    assert df.loc[2, 1] == pytest.approx(1.0, abs=1e-6)


def test_build_pool_heatmap_data_index_sorted_by_evaluation_id() -> None:
    dates = _dates(10)
    rng = np.random.default_rng(2)
    pool = {30: (dates, rng.normal(size=10)), 5: (dates, rng.normal(size=10))}

    df = build_pool_heatmap_data(pool)

    assert list(df.index) == [5, 30]
    assert list(df.columns) == [5, 30]


def test_build_pool_heatmap_data_uncorrelated_pair_near_zero() -> None:
    dates = _dates(500)
    rng = np.random.default_rng(3)
    pool = {1: (dates, rng.normal(size=500)), 2: (dates, rng.normal(size=500))}

    df = build_pool_heatmap_data(pool)

    assert abs(df.loc[1, 2]) < 0.3
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_dashboard_logic_pool_heatmap.py -v
```
Expected: `ImportError`/`AttributeError` — `build_pool_heatmap_data` chưa tồn tại.

- [ ] **Step 3: Code tối thiểu — thêm vào `src/dashboard_logic.py`**

```python
# them vao src/dashboard_logic.py (sau import o dau file, mo rong import)
import numpy as np
import numpy.typing as npt

from src.backtest.pool_corr import PoolCorrelation
from src.local_types import Dates


def build_pool_heatmap_data(
    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]],
) -> pd.DataFrame:
    """Ma tran (n, n) max|rho| Pearson giua PnL cua moi cap alpha trong pool.

    Tai dung PoolCorrelation.max_corr (Phase 6) cho tung cap thay vi viet lai Pearson.
    Duong cheo = 1.0 (alpha tu tuong quan voi chinh no). Pool rong -> DataFrame rong.
    """
    ids = sorted(pool)
    n = len(ids)
    if n == 0:
        return pd.DataFrame()

    matrix = np.eye(n, dtype=np.float64)
    for i, id_i in enumerate(ids):
        dates_i, pnl_i = pool[id_i]
        for j, id_j in enumerate(ids):
            if i == j:
                continue
            corr_engine = PoolCorrelation({id_j: pool[id_j]})
            rho, _ = corr_engine.max_corr(pnl_i, dates_i)
            matrix[i, j] = rho
    return pd.DataFrame(matrix, index=ids, columns=ids)
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_dashboard_logic_pool_heatmap.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Lint + type-check**

```bash
venv/Scripts/python.exe -m ruff check src/dashboard_logic.py tests/unit/test_dashboard_logic_pool_heatmap.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/dashboard_logic.py
```
Expected: sạch.

- [ ] **Step 6: Commit**

```bash
git add src/dashboard_logic.py tests/unit/test_dashboard_logic_pool_heatmap.py
git commit -m "feat(dashboard): build_pool_heatmap_data — ma tran self-corr pool tai dung PoolCorrelation"
```

---

### Task 9.3: `export_shortlist` — xuất shortlist ra file để sim Brain

**Files:**
- Create: `src/pipeline/shortlist_export.py`
- Test: `tests/unit/test_pipeline_shortlist_export.py`

**Interfaces:**
- Consumes: không import `src.pipeline.shortlist` trực tiếp (Phase 8 chưa có chữ ký step-
  by-step cố định) — định nghĩa `Protocol` tối thiểu tại chỗ để decoupling:
  ```python
  class ShortlistRow(Protocol):
      expr_string: str
      sharpe: float
      fitness: float
      turnover: float
      self_corr_max: float
  ```
- Produces:
  ```python
  def export_shortlist(
      items: Sequence[ShortlistRow], path: Path, *, fmt: Literal["csv", "json"] = "csv",
  ) -> Path: ...
  ```
  - `fmt="csv"`: ghi file CSV với header
    `expr_string,sharpe,fitness,turnover,self_corr_max`, một dòng mỗi item, **không** ghi
    index pandas. Định dạng này đem trực tiếp đi "sim trên Brain" được — mỗi dòng là một
    expression kèm metric local để người dùng ưu tiên thứ tự nộp.
  - `fmt="json"`: ghi file JSON là **list** các object (`{"expr_string": ..., "sharpe":
    ..., "fitness": ..., "turnover": ..., "self_corr_max": ...}`), `ensure_ascii=False` (giữ
    dấu tiếng Việt nếu có trong các trường mở rộng tương lai), `indent=2`.
  - Tạo thư mục cha (`path.parent`) nếu chưa có (`mkdir(parents=True, exist_ok=True)`).
  - `items` rỗng → vẫn ghi file hợp lệ (CSV chỉ có header; JSON là `[]`), không lỗi —
    dashboard có thể gọi export trước khi có dữ liệu mà không crash.
  - Trả về `path` (cho caller log/hiển thị đường dẫn).
  - `fmt` khác `"csv"/"json"` → `ValueError` rõ nghĩa.

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_pipeline_shortlist_export.py
"""Test export_shortlist: xuat shortlist ra CSV/JSON de dem di sim Brain."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.pipeline.shortlist_export import export_shortlist


@dataclass(frozen=True, slots=True)
class _Row:
    expr_string: str
    sharpe: float
    fitness: float
    turnover: float
    self_corr_max: float


def _rows() -> list[_Row]:
    return [
        _Row("rank(close)", 1.2, 0.9, 0.15, 0.30),
        _Row("ts_mean(volume, 20)", 0.8, 0.5, 0.40, 0.55),
    ]


def test_export_shortlist_csv_writes_header_and_rows(tmp_path: Path) -> None:
    out = export_shortlist(_rows(), tmp_path / "shortlist.csv", fmt="csv")

    assert out == tmp_path / "shortlist.csv"
    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "expr_string", "sharpe", "fitness", "turnover", "self_corr_max",
        ]
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["expr_string"] == "rank(close)"
    assert rows[0]["sharpe"] == "1.2"


def test_export_shortlist_json_writes_list_of_objects(tmp_path: Path) -> None:
    out = export_shortlist(_rows(), tmp_path / "shortlist.json", fmt="json")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["expr_string"] == "rank(close)"
    assert data[0]["self_corr_max"] == pytest.approx(0.30)


def test_export_shortlist_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "dir" / "shortlist.csv"
    out = export_shortlist(_rows(), nested, fmt="csv")

    assert out.exists()


def test_export_shortlist_empty_items_writes_header_only(tmp_path: Path) -> None:
    out = export_shortlist([], tmp_path / "empty.csv", fmt="csv")

    with out.open(encoding="utf-8") as f:
        content = f.read()
    assert "expr_string" in content
    assert "\n" not in content.strip()  # chi header, khong co dong du lieu


def test_export_shortlist_invalid_fmt_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fmt"):
        export_shortlist(_rows(), tmp_path / "x.txt", fmt="xml")  # type: ignore[arg-type]
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_pipeline_shortlist_export.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.pipeline.shortlist_export'`.

- [ ] **Step 3: Code tối thiểu — `src/pipeline/shortlist_export.py`**

```python
# src/pipeline/shortlist_export.py
"""Xuat shortlist (Phase 8) ra file CSV/JSON de dem di simulate tren WQ Brain.

Logic thuan: nhan Sequence[ShortlistRow] da chuan bi san (tu pipeline.shortlist hoac
dashboard), khong tu goi DB/GP. ShortlistRow la Protocol toi thieu (duck-typing) de
khong ep Phase 9 phai biet chinh xac kieu noi bo cua pipeline.shortlist (Phase 8 chua
co chu ky step-by-step co dinh tai thoi diem viet module nay).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal, Protocol, Sequence


class ShortlistRow(Protocol):
    expr_string: str
    sharpe: float
    fitness: float
    turnover: float
    self_corr_max: float


_FIELDS = ("expr_string", "sharpe", "fitness", "turnover", "self_corr_max")


def export_shortlist(
    items: Sequence[ShortlistRow],
    path: Path,
    *,
    fmt: Literal["csv", "json"] = "csv",
) -> Path:
    """Ghi shortlist ra `path` theo `fmt`; tra ve `path` de caller log/hien thi."""
    if fmt not in ("csv", "json"):
        raise ValueError(f"fmt khong ho tro: {fmt!r} (chi nhan 'csv' hoac 'json')")

    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(_FIELDS))
            writer.writeheader()
            for item in items:
                writer.writerow({field: getattr(item, field) for field in _FIELDS})
    else:
        payload = [{field: getattr(item, field) for field in _FIELDS} for item in items]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return path
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_pipeline_shortlist_export.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Lint + type-check**

```bash
venv/Scripts/python.exe -m ruff check src/pipeline/shortlist_export.py tests/unit/test_pipeline_shortlist_export.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/pipeline/shortlist_export.py
```
Expected: sạch.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/shortlist_export.py tests/unit/test_pipeline_shortlist_export.py
git commit -m "feat(pipeline): export_shortlist CSV/JSON — dem shortlist di sim Brain"
```

---

### Task 9.4: `build_submit_payload` — submit helper logic thuần (không hardcode secret)

**Files:**
- Modify: `src/pipeline/shortlist_export.py`
- Test: `tests/unit/test_pipeline_shortlist_export.py` (mở rộng, hoặc file mới
  `tests/unit/test_pipeline_submit_payload.py` — chọn file mới để giữ Task 9.3 tách biệt)
- Test (mới): `tests/unit/test_pipeline_submit_payload.py`

**Interfaces:**
- Mục đích: tool **đã có** client/login Brain sẵn (`src/simulation`/`src/submission` theo
  cấu trúc repo hiện tại — submit thật KHÔNG được hardcode token/cookie/email/password ở
  đây). Hàm thuần ở đây **không gọi network** — nó chỉ chuẩn bị payload (danh sách
  expression cần submit) từ shortlist đã lọc theo một ngưỡng tối thiểu, để tầng gọi (CLI
  hoặc dashboard, dùng client/login đã đăng nhập sẵn có trong `src/simulation`/
  `src/submission`) truyền tiếp vào hàm submit thật đã tồn tại trong codebase. Việc gọi
  client thật **ngoài phạm vi Task này** (đó là network I/O, không phải hàm thuần testable;
  dashboard render ở Task 9.5 mô tả cách gọi).
  ```python
  def build_submit_payload(
      items: Sequence[ShortlistRow], *, min_sharpe: float = 0.0, max_self_corr: float = 1.0,
  ) -> list[str]: ...
  ```
  Trả về danh sách `expr_string` (theo thứ tự `items` truyền vào, không tự sort lại) thỏa
  `sharpe >= min_sharpe` **và** `self_corr_max <= max_self_corr`. Không có item nào thỏa →
  `[]`. Đây là danh sách "ứng viên để submit" — **không** gọi API submit thật (rule "không
  hardcode secret" được giữ bằng cách hàm này không biết gì về credential/network).

- [ ] **Step 1: Viết test đỏ**

```python
# tests/unit/test_pipeline_submit_payload.py
"""Test build_submit_payload: loc shortlist theo nguong truoc khi dua vao submit thuc."""

from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.shortlist_export import build_submit_payload


@dataclass(frozen=True, slots=True)
class _Row:
    expr_string: str
    sharpe: float
    fitness: float
    turnover: float
    self_corr_max: float


def _rows() -> list[_Row]:
    return [
        _Row("a", sharpe=1.5, fitness=1.0, turnover=0.1, self_corr_max=0.2),
        _Row("b", sharpe=0.3, fitness=0.2, turnover=0.1, self_corr_max=0.9),
        _Row("c", sharpe=1.1, fitness=0.8, turnover=0.1, self_corr_max=0.6),
    ]


def test_build_submit_payload_filters_by_min_sharpe_and_max_self_corr() -> None:
    payload = build_submit_payload(_rows(), min_sharpe=1.0, max_self_corr=0.70)

    assert payload == ["a", "c"]


def test_build_submit_payload_preserves_input_order() -> None:
    rows = list(reversed(_rows()))
    payload = build_submit_payload(rows, min_sharpe=1.0, max_self_corr=0.70)

    assert payload == ["c", "a"]


def test_build_submit_payload_no_match_returns_empty_list() -> None:
    payload = build_submit_payload(_rows(), min_sharpe=10.0)

    assert payload == []


def test_build_submit_payload_default_thresholds_accept_all_positive() -> None:
    payload = build_submit_payload(_rows())

    assert payload == ["a", "b", "c"]
```

- [ ] **Step 2: Chạy test — FAIL**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_pipeline_submit_payload.py -v
```
Expected: `ImportError` — `build_submit_payload` chưa tồn tại.

- [ ] **Step 3: Code tối thiểu — thêm vào `src/pipeline/shortlist_export.py`**

```python
# them vao cuoi src/pipeline/shortlist_export.py
def build_submit_payload(
    items: Sequence[ShortlistRow],
    *,
    min_sharpe: float = 0.0,
    max_self_corr: float = 1.0,
) -> list[str]:
    """Loc shortlist theo nguong toi thieu, tra ve danh sach expr_string ung vien submit.

    KHONG goi network/API submit thuc o day (giu logic thuan, testable). Tang goi
    (CLI/dashboard) dung client/login Brain da co san trong src/submission de submit
    tung expr_string trong danh sach tra ve — KHONG hardcode credential trong module nay.
    """
    return [
        item.expr_string
        for item in items
        if item.sharpe >= min_sharpe and item.self_corr_max <= max_self_corr
    ]
```

- [ ] **Step 4: Chạy test — PASS**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_pipeline_submit_payload.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Lint + type-check**

```bash
venv/Scripts/python.exe -m ruff check src/pipeline/shortlist_export.py tests/unit/test_pipeline_submit_payload.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/pipeline/shortlist_export.py
```
Expected: sạch.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/shortlist_export.py tests/unit/test_pipeline_submit_payload.py
git commit -m "feat(pipeline): build_submit_payload — loc shortlist theo nguong truoc submit thuc"
```

---

### Task 9.5: Render Streamlit — 3 tab mới trong `dashboard/app.py`

**Files:**
- Modify: `dashboard/app.py`

**Mô tả (không phải hàm thuần — chỉ render, không unit test):**

Mở rộng `st.tabs(...)` hiện có (dòng 37, 5 tab: `Overview, Explorer, Tiến trình,
Submissions, Correlation`) thành 8 tab, **giữ nguyên thứ tự và nội dung 5 tab cũ không đổi
một dòng nào** — chỉ thêm 3 tab mới ở cuối: `Calibration`, `Pool-corr`, `Shortlist`.

```python
# sua dong "tab_overview, tab_explorer, tab_ga, tab_subs, tab_corr = st.tabs([...])"
# thanh (them 3 tab moi, GIU NGUYEN 5 tab cu):
(
    tab_overview, tab_explorer, tab_ga, tab_subs, tab_corr,
    tab_calib, tab_pool, tab_shortlist,
) = st.tabs(
    [
        "Overview", "Explorer", "Tiến trình", "Submissions", "Correlation",
        "Calibration", "Pool-corr", "Shortlist",
    ]
)

# ... (5 block `with tab_overview:` ... `with tab_corr:` GIU NGUYEN khong sua) ...

with tab_calib:
    # Phase 9.1: hien thi CalibrationReport (Phase 4.5) qua build_calibration_view.
    from src.calibration.harness import CalibrationHarness  # noqa: E402  (forward dep Phase 4.5)
    from src.calibration.loader import load_brain_records  # noqa: E402
    from src.dashboard_logic import build_calibration_view  # noqa: E402

    st.caption(
        "Spearman rho giua metric local va Brain tren alpha da sim thuc. "
        "Khong tin ranking local cho toi khi spearman_sharpe >= CALIBRATION_RHO_BAR."
    )
    try:
        records = load_brain_records(engine)
        report = CalibrationHarness(scorer=...).run(records)  # xem ghi chu duoi
        st.dataframe(build_calibration_view(report), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Khong tinh duoc calibration report: {exc}")

with tab_pool:
    # Phase 9.2: heatmap self-correlation pool (Phase 6 load_pool + build_pool_heatmap_data).
    from src.dashboard_logic import build_pool_heatmap_data  # noqa: E402
    from src.storage.repository import AlphaRepository  # noqa: E402

    st.caption("Max|rho| Pearson PnL giua cac alpha da pass trong pool — local proxy, "
               "KHONG thay the self-corr checker thuc cua Brain truoc submit.")
    try:
        repo = AlphaRepository(engine)
        pool = repo.load_pool()
        if pool:
            st.dataframe(
                build_pool_heatmap_data(pool).style.background_gradient(
                    cmap="Reds", vmin=0.0, vmax=1.0
                ),
                use_container_width=True,
            )
        else:
            st.info("Pool rong — chua co alpha pass nao duoc luu PnL.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Khong doc duoc pool: {exc}")

with tab_shortlist:
    # Phase 9.3/9.4: export shortlist + submit helper (KHONG hardcode secret).
    from pathlib import Path  # noqa: E402

    from src.pipeline.shortlist_export import build_submit_payload, export_shortlist  # noqa: E402

    st.caption("Xuat shortlist ra file de dem di simulate tren Brain.")
    # `shortlist_rows` lay tu pipeline.shortlist (Phase 8) — tang goi chuan bi du lieu,
    # KHONG goi truc tiep trong block nay vi chu ky Phase 8 chua chot (xem dau plan).
    shortlist_rows = _load_current_shortlist()  # ham noi bo dashboard, xem ghi chu duoi
    fmt = st.selectbox("Dinh dang xuat", ["csv", "json"])
    export_path = st.text_input("Duong dan file xuat", value="exports/shortlist.csv")
    if st.button("Xuat shortlist"):
        out = export_shortlist(shortlist_rows, Path(export_path), fmt=fmt)
        st.success(f"Da xuat {len(shortlist_rows)} dong vao {out}")

    st.divider()
    st.caption("Submit helper (tuy chon) — dung client/login Brain da dang nhap san "
               "trong src/submission, KHONG nhap credential o day.")
    min_sharpe = st.number_input("Sharpe toi thieu", value=1.0)
    max_self_corr = st.number_input("Self-corr toi da", value=0.70)
    payload = build_submit_payload(
        shortlist_rows, min_sharpe=min_sharpe, max_self_corr=max_self_corr
    )
    st.write(f"{len(payload)} expression dat nguong, san sang submit:")
    st.code("\n".join(payload) or "(khong co)")
    if payload and st.button("Submit cac expression tren (dung client da login)"):
        # Goi client/login da co san (vd src.submission.submit_client.submit_many) —
        # KHONG implement lai logic submit o day; chi wire toi ham da ton tai trong repo.
        st.warning(
            "Wire toi ham submit thuc cua src/submission tai day khi module do co API "
            "phu hop (vd submit_many(expr_list, client)) — KHONG hardcode token/cookie."
        )
```

**Ghi chú quan trọng (người thực thi PHẢI xử lý, không phải placeholder bỏ ngỏ):**

1. `CalibrationHarness(scorer=...)` ở trên cần một `LocalScorer` thật (Phase 4.5 định nghĩa
   `LocalScorer = Callable[[str], LocalScore | None]`) — wiring cụ thể (dùng
   `src.pipeline.runner.score_one` nếu Phase 8 đã có, hoặc gọi trực tiếp
   parser→evaluator→backtester→metrics nếu Phase 8 chưa merge) **phải được điền khi thực
   thi Task này**, không để `...`. Nếu tại thời điểm thực thi Phase 7/8 đã merge, dùng
   `score_one` làm scorer (khớp dependency rule — `dashboard` được phép phụ thuộc `pipeline`
   ở layer trên cùng). Nếu chưa, viết một adapter cục bộ trong `dashboard/app.py` gọi trực
   tiếp `parse → Evaluator → PortfolioBuilder → Backtester → MetricsCalculator` (đã có từ
   Phase 1-4) — đây vẫn là code thật, không phải mock.
2. `_load_current_shortlist()` tương tự — phải gọi `src.pipeline.shortlist.shortlist(...)`
   thật (Phase 8) nếu đã có; nếu Phase 8 chưa merge tại thời điểm thực thi Phase 9, dừng lại
   và báo cáo block cho Task 9.5 phần Shortlist-tab (Task 9.1/9.2 vẫn merge được độc lập) —
   **không** tự ý viết lại `pipeline.shortlist` ở đây (lấn phạm vi Phase 8).
3. Toàn bộ import `from src...` trong các block `with tab_*:` đặt **trong** block (không ở
   đầu file) là pattern đã có sẵn ở tab Correlation cũ (dòng 128 file gốc) — giữ nguyên quy
   ước này, không refactor sang import đầu file (giảm cost import khi tab không được mở,
   và giữ patch tối giản so với code cũ).

- [ ] **Step 1: Implement** theo mô tả trên, điền đầy đủ phần "Ghi chú quan trọng" (không để
  `...`/placeholder nào còn sót trong code thật).
- [ ] **Step 2: Lint**

```bash
venv/Scripts/python.exe -m ruff check dashboard/app.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/dashboard_logic.py src/pipeline/shortlist_export.py
```
(mypy --strict chỉ chạy trên module logic mới theo yêu cầu — `dashboard/app.py` dùng
`streamlit` không có stub đầy đủ, không bắt buộc strict-clean, nhưng `ruff check` vẫn áp
dụng.)

- [ ] **Step 3: Smoke test thủ công**

```bash
venv/Scripts/python.exe -m streamlit run dashboard/app.py
```
Mở trình duyệt (Streamlit tự mở `localhost:8501`), kiểm tra thủ công:
- 5 tab cũ (`Overview/Explorer/Tiến trình/Submissions/Correlation`) vẫn hiển thị đúng như
  trước (không có lỗi mới xuất hiện so với trước khi sửa).
- Tab `Calibration` hiển thị bảng metric (hoặc cảnh báo rõ nghĩa nếu chưa có
  `brain_record` nào trong DB — không crash trắng trang).
- Tab `Pool-corr` hiển thị heatmap (hoặc "Pool rỗng" nếu DB chưa có alpha pass nào).
- Tab `Shortlist` cho xuất file CSV/JSON thật vào `exports/` (kiểm tra file được tạo) và
  liệt kê danh sách submit-ready theo ngưỡng nhập.
- Dừng server (`Ctrl+C`) sau khi xác nhận.

- [ ] **Step 4: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): them tab Calibration/Pool-corr/Shortlist, giu nguyen 5 tab cu"
```

---

### Task 9.6: Review tổng + merge + push

**Files:** không tạo file mới — chỉ review/merge.

- [ ] **Step 1: Chạy toàn bộ test suite + lint + mypy trên các module Phase 9**

```bash
venv/Scripts/python.exe -m pytest tests/unit/test_dashboard_logic_calibration.py tests/unit/test_dashboard_logic_pool_heatmap.py tests/unit/test_pipeline_shortlist_export.py tests/unit/test_pipeline_submit_payload.py -v
venv/Scripts/python.exe -m pytest -q
venv/Scripts/python.exe -m ruff check src/dashboard_logic.py src/pipeline/shortlist_export.py dashboard/app.py
venv/Scripts/python.exe -m mypy --strict --follow-imports=silent src/dashboard_logic.py src/pipeline/shortlist_export.py
```
Expected: toàn bộ test Phase 9 pass; full suite (`pytest -q`) không có regression trên 5
tab/test cũ; ruff sạch; mypy --strict sạch trên 2 module logic mới.

- [ ] **Step 2: Self-review (checklist)**

  - [ ] Logic thuần (`build_calibration_view`, `build_pool_heatmap_data`,
    `export_shortlist`, `build_submit_payload`) sống ở `src/`, có test, **không** import
    `streamlit`.
  - [ ] `dashboard/app.py` là nơi duy nhất import `streamlit`; 5 tab cũ không bị sửa nội
    dung (chỉ thêm tab mới vào `st.tabs(...)`).
  - [ ] Không hàm nào trong `src/dashboard_logic.py`/`shortlist_export.py` mở kết nối
    network hoặc đọc credential (`os.environ`, file `.env`, token) — submit thật ủy quyền
    cho client/login đã có sẵn trong `src/submission`.
  - [ ] `export_shortlist` xuất đúng 5 cột (`expr_string,sharpe,fitness,turnover,
    self_corr_max`) — định dạng đủ để đem đi sim Brain (mỗi dòng 1 expression + metric local
    để người dùng ưu tiên).
  - [ ] Không còn `...`/placeholder nào trong code đã merge (Task 9.5 ghi chú 1/2 đã được
    điền cụ thể, không để treo).
  - [ ] `PoolCorrelation`/`CalibrationReport` được **tái dùng** (import), không viết lại
    logic Pearson/Spearman ở Phase 9.
  - [ ] Tiếng Việt trong docstring/comment giữ dấu đầy đủ (kiểm tra bằng mắt các file vừa
    tạo/sửa).
  - [ ] Nếu Task 9.5 phần Shortlist-tab bị block do Phase 8 chưa merge (ghi chú 2) — ghi rõ
    trạng thái block trong PROGRESS.md, không che giấu bằng code giả.

- [ ] **Step 3: Cập nhật PROGRESS.md** (journal cuối phase theo Per-phase ritual) — ghi rõ:
  module đã tạo, ρ/heatmap đã verify bằng cách nào (smoke test thủ công), risk còn mở (Phase
  8 chưa merge → Shortlist-tab phần `_load_current_shortlist` còn placeholder thật hay đã
  wire xong).

- [ ] **Step 4: Merge + push**

```bash
git checkout main
git merge --no-ff phase-9-dashboard
git push origin main
git branch -d phase-9-dashboard
```
