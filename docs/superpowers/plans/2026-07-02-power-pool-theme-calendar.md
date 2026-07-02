# Power Pool Theme Calendar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lưu lịch Power Pool Theme (dữ liệu THỦ CÔNG từ bài
"Current month Power Pool Themes", support.worldquantbrain.com, xác nhận 2026-07-02 với tài
khoản `tuananhpo13@gmail.com` — GOLD Genius, CONSULTANT_APPROVED), tra được theme đang áp dụng
cho 1 ngày, và kiểm 1 alpha có khớp theme đó không (region/delay/universe/dataset loại trừ).

**Dữ liệu nguồn (chép nguyên văn, KHÔNG suy diễn thêm):**
```
June
1-7    : USA D1 Fast Datasets   (chỉ có tên, KHÔNG có filter chi tiết trong bản đã nhận)
8-14   : USA D1 Fast Datasets   (chỉ có tên, KHÔNG có filter chi tiết)
15-21  : GLB D1 Datasets        (chỉ có tên, KHÔNG có filter chi tiết)
22-28  : GLB D1 Datasets        (chỉ có tên, KHÔNG có filter chi tiết)
29/6-5/7: region=USA & delay=1 & universe=TOP1000 and neutralization in (slow, fast,
          slow and fast, ram, statistical, crowding) and datasets not in ['pv1']
          (có filter đầy đủ, KHÔNG có tên riêng trong bản đã nhận)
```
Tuần cuối (29/6-5/7) **trùng hôm nay** (2026-07-02) — đây là theme ĐANG áp dụng.

**Điểm KHÔNG chắc chắn, xử lý rõ ràng thay vì đoán:**
- Cụm `neutralization in (slow, fast, slow and fast, ram, statistical, crowding)` — các giá trị
  này KHÔNG khớp `VALID_NEUTRALIZATIONS` của `SimConfig` (NONE/MARKET/SECTOR/INDUSTRY/
  SUBINDUSTRY/COUNTRY/EXCHANGE) — đây có thể là 1 khái niệm khác của riêng hệ thống Theme (có
  thể liên quan "tốc độ" dataset, không phải setting `neutralization` của simulation). **KHÔNG
  cố map vào `SimConfig`** — giữ nguyên văn trong `unparsed_constraints`, hàm kiểm khớp
  (`matches_theme`) KHÔNG dùng field này để pass/fail, chỉ để hiển thị cảnh báo "cần tự xem lại".
- 2 theme đầu ("USA D1 Fast Datasets"/"GLB D1 Datasets") không có filter chi tiết —
  `region/delay/universe/datasets_excluded` để `None`/rỗng, `theme_for_date()` vẫn trả về được
  (có `name`), nhưng `matches_theme()` với các field `None` sẽ COI LÀ KHÔNG RÀNG BUỘC (không
  chặn gì) vì không có dữ liệu để so — không phải "match tất cả", chỉ là "chưa biết để kiểm".

**Architecture:** `src/scoring/dataset_usage.py` (sub-project D) thêm hàm `datasets_used()`
(khác `dataset_of_alpha` — trả TẤT CẢ dataset dùng, không chỉ khi single-dataset).
`src/scoring/power_pool_theme.py` (module mới) — `PowerPoolThemeWeek`, `parse_theme_filter()`,
lịch `JUNE_JULY_2026_CALENDAR` (dữ liệu thủ công, CẦN CẬP NHẬT khi sang tháng/theme mới — ghi rõ
trong docstring cách cập nhật), `theme_for_date()`, `matches_theme()`.

**Phạm vi KHÔNG làm:** KHÔNG tự động nộp "pure Power Pool" dựa trên theme này (vẫn quá rủi ro
cho 1 lần chạy chưa kiểm chứng — để dành đợt sau khi có dữ liệu nhiều tháng hơn để tin cậy hơn).
Đây chỉ là hạ tầng tra cứu + kiểm khớp, sẵn sàng dùng khi bạn quyết định nối vào luồng nộp thật.

## Global Constraints

- TDD bắt buộc: test FAIL trước, code tối thiểu, xác nhận PASS.
- Code/comment/commit tiếng Việt có dấu.
- Mỗi task = 1 commit.
- Chạy test: `venv/Scripts/python -m pytest`.
- Chép ĐÚNG NGUYÊN VĂN dữ liệu lịch nêu trên vào code — không tự thêm/đoán field còn thiếu.

---

### Task 1: `datasets_used()` — tất cả dataset dùng trong 1 alpha (không giới hạn single-dataset)

**Files:**
- Modify: `src/scoring/dataset_usage.py` (thêm hàm mới cuối file)
- Test: `tests/unit/test_dataset_usage.py`

**Interfaces:**
- Consumes: `FieldCollector`, `OperatorCollector`, `_GROUPING_FIELDS`, `_PV1_OPERATORS` (đã có
  sẵn trong file, sub-project D).
- Produces: `datasets_used(expr: str, field_dataset: dict[str, str]) -> set[str]` — dùng bởi
  Task 2 (`matches_theme`).

- [ ] **Step 1: Viết test FAIL**

Thêm vào cuối `tests/unit/test_dataset_usage.py`:

```python
from src.scoring.dataset_usage import datasets_used


def test_datasets_used_tra_tat_ca_dataset_khong_gioi_han_single():
    fd = {"close": "pv1", "eps": "fundamental6"}
    assert datasets_used("rank(add(close, eps))", fd) == {"pv1", "fundamental6"}


def test_datasets_used_bo_qua_grouping_field():
    fd = {"close": "pv1"}
    assert datasets_used("group_rank(close, sector)", fd) == {"pv1"}


def test_datasets_used_field_khong_ro_dataset_bi_bo_qua_khong_loi():
    fd = {"close": "pv1"}
    assert datasets_used("rank(add(close, unknown_field))", fd) == {"pv1"}


def test_datasets_used_inst_pnl_them_pv1():
    fd = {"eps": "fundamental6"}
    assert datasets_used("inst_pnl(eps, 5)", fd) == {"fundamental6", "pv1"}
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/unit/test_dataset_usage.py -k datasets_used -v`
Expected: FAIL với `ImportError: cannot import name 'datasets_used'`

- [ ] **Step 3: Cài tối thiểu**

Thêm vào cuối `src/scoring/dataset_usage.py`:

```python
def datasets_used(expr: str, field_dataset: dict[str, str]) -> set[str]:
    """Tập TẤT CẢ dataset_id dùng trong `expr` (bỏ qua grouping field và field không rõ
    dataset) — khác `dataset_of_alpha` (chỉ trả kết quả khi DUY NHẤT 1 dataset). Dùng để kiểm
    khớp Power Pool Theme (loại trừ dataset cụ thể, không yêu cầu single-dataset)."""
    node = parse_expression(expr)
    fields = FieldCollector().visit(node)
    operators = OperatorCollector().visit(node)

    datasets: set[str] = set()
    for field_id in fields:
        if field_id in _GROUPING_FIELDS:
            continue
        dataset_id = field_dataset.get(field_id)
        if dataset_id is not None:
            datasets.add(dataset_id)
    if operators & _PV1_OPERATORS:
        datasets.add("pv1")
    return datasets
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/unit/test_dataset_usage.py -v`
Expected: PASS toàn bộ (6 test cũ + 4 test mới = 10)

- [ ] **Step 5: Commit**

```bash
git add src/scoring/dataset_usage.py tests/unit/test_dataset_usage.py
git commit -m "feat(scoring): them datasets_used (tat ca dataset dung, khong gioi han single)"
```

---

### Task 2: `src/scoring/power_pool_theme.py` — lịch + parser + kiểm khớp theme

**Files:**
- Create: `src/scoring/power_pool_theme.py`
- Test: `tests/unit/test_power_pool_theme.py`

**Interfaces:**
- Consumes: không phụ thuộc module khác (nhận `datasets_used` từ caller qua tham số
  `datasets_used: set[str]`, không tự import `src.scoring.dataset_usage` — giữ module thuần,
  dễ test).
- Produces: `PowerPoolThemeWeek`, `parse_theme_filter(raw: str) -> dict`,
  `JUNE_JULY_2026_CALENDAR: list[PowerPoolThemeWeek]`,
  `theme_for_date(d, calendar=None) -> PowerPoolThemeWeek | None`,
  `matches_theme(week, *, region, delay, universe, datasets_used) -> tuple[bool, list[str]]`.

- [ ] **Step 1: Viết test FAIL**

Tạo file `tests/unit/test_power_pool_theme.py`:

```python
"""Test lịch Power Pool Theme — dữ liệu thủ công từ bài 'Current month Power Pool Themes'
(support.worldquantbrain.com), xác nhận 2026-07-02. Xem docstring
src/scoring/power_pool_theme.py để biết cách cập nhật khi sang tháng/theme mới."""

from __future__ import annotations

from datetime import date

from src.scoring.power_pool_theme import (
    JUNE_JULY_2026_CALENDAR,
    matches_theme,
    parse_theme_filter,
    theme_for_date,
)


def test_parse_theme_filter_tach_dung_region_delay_universe_datasets():
    raw = (
        "region=USA & delay=1 & universe=TOP1000 and neutralization in "
        "(slow, fast, slow and fast, ram, statistical, crowding) and datasets not in ['pv1']"
    )
    result = parse_theme_filter(raw)
    assert result["region"] == "USA"
    assert result["delay"] == 1
    assert result["universe"] == "TOP1000"
    assert result["datasets_excluded"] == ("pv1",)
    assert result["unparsed_constraints"] is not None
    assert "neutralization" in result["unparsed_constraints"]


def test_theme_for_date_tuan_co_filter_day_du():
    week = theme_for_date(date(2026, 7, 2))  # hôm nay, thuộc tuần 29/6-5/7
    assert week is not None
    assert week.region == "USA"
    assert week.delay == 1
    assert week.universe == "TOP1000"
    assert week.datasets_excluded == ("pv1",)


def test_theme_for_date_tuan_chi_co_ten():
    week = theme_for_date(date(2026, 6, 3))
    assert week is not None
    assert week.name == "USA D1 Fast Datasets"
    assert week.region is None  # không có filter chi tiết trong dữ liệu đã nhận


def test_theme_for_date_ngoai_lich_tra_none():
    assert theme_for_date(date(2026, 8, 1)) is None


def test_matches_theme_dat_het_dieu_kien():
    week = theme_for_date(date(2026, 7, 2))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000", datasets_used={"fundamental6"},
    )
    assert ok is True
    assert reasons == []


def test_matches_theme_dung_dataset_bi_loai_tru():
    week = theme_for_date(date(2026, 7, 2))
    ok, reasons = matches_theme(
        week, region="USA", delay=1, universe="TOP1000", datasets_used={"pv1"},
    )
    assert ok is False
    assert any("pv1" in r for r in reasons)


def test_matches_theme_sai_region():
    week = theme_for_date(date(2026, 7, 2))
    ok, reasons = matches_theme(
        week, region="EUR", delay=1, universe="TOP1000", datasets_used=set(),
    )
    assert ok is False
    assert any("region" in r for r in reasons)


def test_matches_theme_tuan_khong_co_filter_chi_tiet_khong_chan_gi():
    week = theme_for_date(date(2026, 6, 3))  # chỉ có tên, region/delay/universe đều None
    ok, reasons = matches_theme(
        week, region="ASI", delay=0, universe="TOP500", datasets_used={"pv1"},
    )
    assert ok is True  # không có field nào để so -> không chặn
    assert reasons == []
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool_theme.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.scoring.power_pool_theme'`

- [ ] **Step 3: Cài tối thiểu**

Tạo `src/scoring/power_pool_theme.py`:

```python
"""Lịch Power Pool Theme — dữ liệu THỦ CÔNG lấy từ bài 'Current month Power Pool Themes'
(https://support.worldquantbrain.com/hc/en-us/articles/38927747787031), WQ Brain cập nhật
HÀNG THÁNG/HÀNG TUẦN. Đây KHÔNG phải dữ liệu tự fetch được (chưa tìm ra endpoint/trang đọc
được tự động qua tool hiện có — đã thử WebFetch (403) và read_forum_post qua wqb-mcp (timeout))
— CẦN CẬP NHẬT THỦ CÔNG mỗi khi có tháng/theme mới, bằng cách thêm `PowerPoolThemeWeek` mới vào
CALENDAR bên dưới (copy nguyên văn từ bài viết, KHÔNG suy diễn field còn thiếu).

Nguồn xác nhận 2026-07-02 (tài khoản tuananhpo13@gmail.com, GOLD Genius, CONSULTANT_APPROVED):
2 theme đầu tháng 6 ("USA D1 Fast Datasets", "GLB D1 Datasets") chỉ có TÊN, không có filter chi
tiết trong bản đã nhận từ người dùng — để None/rỗng thay vì đoán. Theme tuần 29/6-5/7 có filter
đầy đủ. Cụm "neutralization in (slow, fast, slow and fast, ram, statistical, crowding)" KHÔNG
khớp VALID_NEUTRALIZATIONS chuẩn của SimConfig — giữ nguyên văn trong `unparsed_constraints`,
KHÔNG dùng để pass/fail trong `matches_theme()`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PowerPoolThemeWeek:
    start_date: date
    end_date: date
    name: str | None = None
    region: str | None = None
    delay: int | None = None
    universe: str | None = None
    datasets_excluded: tuple[str, ...] = ()
    unparsed_constraints: str | None = None

    def contains(self, d: date) -> bool:
        return self.start_date <= d <= self.end_date


def parse_theme_filter(raw: str) -> dict:
    """Trích region/delay/universe/datasets_excluded từ chuỗi filter dạng
    "region=USA & delay=1 & universe=TOP1000 and neutralization in (...) and datasets not in
    ['pv1']". Phần "neutralization in (...)" giữ nguyên văn trong `unparsed_constraints`."""
    result: dict = {
        "region": None, "delay": None, "universe": None,
        "datasets_excluded": (), "unparsed_constraints": None,
    }
    m = re.search(r"region\s*=\s*([A-Za-z0-9_]+)", raw)
    if m:
        result["region"] = m.group(1)
    m = re.search(r"delay\s*=\s*(\d+)", raw)
    if m:
        result["delay"] = int(m.group(1))
    m = re.search(r"universe\s*=\s*([A-Za-z0-9_]+)", raw)
    if m:
        result["universe"] = m.group(1)
    m = re.search(r"datasets\s+not\s+in\s*\[([^\]]*)\]", raw)
    if m:
        items = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
        result["datasets_excluded"] = tuple(items)
    m = re.search(r"(neutralization\s+in\s*\([^)]*\))", raw)
    if m:
        result["unparsed_constraints"] = m.group(1)
    return result


# CALENDAR — dữ liệu THỦ CÔNG, xem docstring module để biết cách cập nhật.
JUNE_JULY_2026_CALENDAR: list[PowerPoolThemeWeek] = [
    PowerPoolThemeWeek(date(2026, 6, 1), date(2026, 6, 7), name="USA D1 Fast Datasets"),
    PowerPoolThemeWeek(date(2026, 6, 8), date(2026, 6, 14), name="USA D1 Fast Datasets"),
    PowerPoolThemeWeek(date(2026, 6, 15), date(2026, 6, 21), name="GLB D1 Datasets"),
    PowerPoolThemeWeek(date(2026, 6, 22), date(2026, 6, 28), name="GLB D1 Datasets"),
    PowerPoolThemeWeek(
        date(2026, 6, 29), date(2026, 7, 5),
        region="USA", delay=1, universe="TOP1000",
        datasets_excluded=("pv1",),
        unparsed_constraints="neutralization in (slow, fast, slow and fast, ram, statistical, crowding)",
    ),
]


def theme_for_date(
    d: date, calendar: list[PowerPoolThemeWeek] | None = None
) -> PowerPoolThemeWeek | None:
    """Trả theme đang áp dụng cho ngày `d`, hoặc None nếu ngoài lịch đã biết (lịch cần cập nhật
    thủ công khi sang tháng mới/theme mới — xem docstring module)."""
    cal = calendar if calendar is not None else JUNE_JULY_2026_CALENDAR
    for week in cal:
        if week.contains(d):
            return week
    return None


def matches_theme(
    week: PowerPoolThemeWeek, *, region: str, delay: int, universe: str, datasets_used: set[str],
) -> tuple[bool, list[str]]:
    """Kiểm 1 alpha có khớp `week` không — CHỈ kiểm phần đã parse chắc chắn (region/delay/
    universe/datasets_excluded). Field nào của `week` là None thì KHÔNG chặn (chưa biết để so,
    không phải "match tất cả"). `week.unparsed_constraints` (nếu có) KHÔNG ảnh hưởng kết quả ở
    đây — người gọi tự đọc `week.unparsed_constraints` để xem lại thủ công."""
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
    return (not reasons, reasons)
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/unit/test_power_pool_theme.py -v`
Expected: PASS (8/8)

- [ ] **Step 5: Chạy toàn bộ suite, xác nhận không vỡ gì**

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS hết, trừ 1 fail có sẵn không liên quan (`test_make_engine_postgres_backend`).

- [ ] **Step 6: Commit**

```bash
git add src/scoring/power_pool_theme.py tests/unit/test_power_pool_theme.py
git commit -m "feat(scoring): them lich Power Pool Theme + parser + kiem khop (du lieu 2026-07)"
```

---

## Self-Review (đã chạy)

- **Spec coverage**: dữ liệu lịch chép đúng nguyên văn 5 dòng đã xác nhận; parser xử lý đúng
  chuỗi filter thật đã cho; `theme_for_date`/`matches_theme` phủ cả 2 trường hợp (tuần có
  filter đầy đủ và tuần chỉ có tên).
- **Placeholder scan**: sạch — không có field nào "TBD", chỗ thiếu dữ liệu dùng `None`/`()` có
  chủ đích, giải thích rõ trong docstring.
- **Type consistency**: `datasets_used()` (Task 1) trả `set[str]`, đúng kiểu tham số
  `datasets_used` của `matches_theme()` (Task 2).
- **Cảnh báo rõ ràng cho người bảo trì sau này**: docstring module ghi rõ đây là dữ liệu thủ
  công, đã thử 2 cách tự fetch đều thất bại, cách cập nhật khi có tháng mới.
