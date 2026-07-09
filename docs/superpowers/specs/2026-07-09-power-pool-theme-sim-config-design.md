# Thiết kế: Sim theo Power Pool Theme để nộp được Pure Power Pool

Ngày: 2026-07-09
Trạng thái: đã duyệt (chờ review spec)

## Bối cảnh / Vấn đề

Nộp **Pure Power Pool** trên WQ Brain đòi hỏi alpha khớp một **Power Pool Theme** đang hiệu
lực; nếu không sẽ bị từ chối: *"Pure Power Pool submission does not match any Power Pool
Theme."*

Theme tuần hiện tại (nguồn: bài "Current month Power Pool Themes", người dùng xác nhận
2026-07-09):

```
region=USA & delay=1 & universe=TOP1000
& neutralization in (slow, fast, slow and fast, ram, statistical, crowding)
& datasets not in ['pv1']
```

Hiện trạng codebase **không thể** sinh config khớp theme:

1. `SimConfig.VALID_NEUTRALIZATIONS` chỉ có nhóm {NONE, MARKET, SECTOR, INDUSTRY,
   SUBINDUSTRY, COUNTRY, EXCHANGE} — **không có** neutralization rủi ro. Ép
   `STATISTICAL` → `ValueError`.
2. Đường alt-data sim với `universe=TOP3000` và neutralization **nhóm** (qua
   `neutralization_for_expr` → SECTOR/SUBINDUSTRY/INDUSTRY) → không đời nào khớp theme
   (cần TOP1000 + risk-neut).
3. `src/scoring/power_pool_theme.py` bắt được ràng buộc nhưng để `neutralization` ở dạng
   `unparsed_constraints` **không enforce**; `matches_theme` bỏ qua neutralization.
4. `JUNE_JULY_2026_CALENDAR` chỉ tới 2026-07-05 → `theme_for_date(2026-07-09)` trả `None`.

## Giá trị enum API (tra từ `docs/worldquantbrain/docs/advanced-topics/*`)

| Theme token       | Enum API BRAIN            |
|-------------------|---------------------------|
| slow              | `SLOW`                    |
| fast              | `FAST`                    |
| slow and fast     | `SLOW_AND_FAST`           |
| ram               | `REVERSION_AND_MOMENTUM`  |
| statistical       | `STATISTICAL`             |
| crowding          | `CROWDING`                |

## Quyết định (người dùng chốt)

- **Neutralization:** map theo category dataset + cho phép sweep con (mặc định 1×).
- **Phạm vi:** đọc theme từ lịch là **hành vi mặc định** (không cần cờ). Có theme → áp
  ràng buộc theme; không có theme → cảnh báo + rơi về default hiện tại (Regular), không chặn.
- **Cách tiếp cận:** A — lớp theme-driven config mỏng (tách bạch, dễ test), thay vì sửa
  cứng default toàn cục (B) hay chỉ vá nhánh alt-data (C).

## Thiết kế

### Component 1 — `src/simulation/config.py`

Thêm `RISK_NEUTRALIZATIONS = {"STATISTICAL", "CROWDING", "REVERSION_AND_MOMENTUM", "SLOW",
"FAST", "SLOW_AND_FAST"}`; gộp vào `VALID_NEUTRALIZATIONS`. Không đổi default (`SUBINDUSTRY`)
— chỉ mở rộng tập hợp lệ để `SimConfig` chấp nhận risk-neut khi được override.

### Component 2 — `src/scoring/power_pool_theme.py`

- `parse_allowed_neutralizations(unparsed_constraints: str | None) -> frozenset[str]`:
  parse cụm `neutralization in (...)`, map từng token tự do → enum API (bảng trên).
  Token lạ → bỏ qua (không đoán). Trả `frozenset()` nếu không có cụm.
- `PowerPoolThemeWeek` thêm field `allowed_neutralizations: frozenset[str] = frozenset()`,
  điền sẵn khi dựng calendar (từ `parse_allowed_neutralizations`).
- Thêm entry tuần hiện tại: `date(2026,7,6)–date(2026,7,12)`, region=USA, delay=1,
  universe=TOP1000, datasets_excluded=("pv1",),
  unparsed_constraints="neutralization in (slow, fast, slow and fast, ram, statistical, crowding)",
  allowed_neutralizations = {SLOW, FAST, SLOW_AND_FAST, REVERSION_AND_MOMENTUM, STATISTICAL, CROWDING}.
- `matches_theme(...)` nhận thêm tham số `neutralization: str | None = None`; nếu theme có
  `allowed_neutralizations` không rỗng và `neutralization` không thuộc tập đó → thêm reason.
  Giữ nguyên hành vi region/delay/universe/datasets_excluded.

### Component 3 — `src/generation/alt_data_seeds.py`

- `pp_neutralization_for_expr(expr, allowed: frozenset[str], registry=None) -> str`:
  map category→risk-neut rồi **giao với `allowed`**:
  - option → `STATISTICAL`
  - social/sentiment → `CROWDING`
  - price-derived → `REVERSION_AND_MOMENTUM`
  - analyst/fundamental → `SLOW`
  - mặc định → `STATISTICAL`
  - nếu lựa chọn không thuộc `allowed` → phần tử đầu (sắp xếp ổn định) của `allowed`.
- `pp_neut_candidates(expr, allowed, registry=None, sweep=False) -> list[str]`: mặc định
  trả `[pp_neutralization_for_expr(...)]` (1×); `sweep=True` trả toàn bộ `allowed` (ổn định)
  để refiner sweep con.
- Giữ nguyên `neutralization_for_expr` (group-neut) cho đường non-PP.

### Component 4 — Wiring (`closed_loop_adapters` + `closed_loop` + `main.py`)

- Mặc định đọc `theme_for_date(date.today())`.
- Có theme → dựng `SimConfig` override region/delay/universe theo theme; nhánh alt-data sim
  thẳng chọn neutralization qua Component 3 (giao với `theme.allowed_neutralizations`);
  nếu bật sweep → lặp qua `pp_neut_candidates`, giữ config Sharpe/fitness cao nhất.
- Không có theme cho hôm nay → `logger.warning` + rơi về default hiện tại (Regular),
  không chặn.

### Component 5 — Gate trước submit (`src/submission/…` / power_pool path)

Trước khi nộp Pure Power Pool: gọi `matches_theme` (đã gồm neutralization) trên
(region, delay, universe, datasets_used, neutralization) của alpha. Không khớp → chặn nộp,
log rõ từng reason (đúng thông điệp Brain sẽ trả) thay vì để Brain từ chối.

## Kiểm thử (TDD — đỏ→xanh từng component)

- `test_power_pool_theme.py`: `parse_allowed_neutralizations` (đủ 6 token, token lạ bị bỏ,
  chuỗi None → rỗng); `matches_theme` chặn khi neut ngoài tập, cho qua khi trong tập, không
  chặn khi theme không khai báo neut; entry tuần hiện tại tồn tại + đúng filter.
- `test_sim_config.py`: `SimConfig` chấp nhận từng risk-neut; vẫn từ chối chuỗi bậy.
- `test_alt_data_seeds.py`: `pp_neutralization_for_expr` map đúng theo category + giao với
  allowed + fallback; `pp_neut_candidates` 1× vs sweep.
- Integration: theme→config wiring (có theme set TOP1000 + risk-neut; không theme → giữ
  default); gate submit chặn alpha lệch theme.

## Ngoài phạm vi (YAGNI)

- Tự động fetch lịch theme từ web (vẫn cập nhật thủ công — xem docstring module).
- Đổi cách chấm/tune local cho alt-data (panel local vẫn chỉ price/volume).
- Neutralization rủi ro cho đường Regular/PV.
