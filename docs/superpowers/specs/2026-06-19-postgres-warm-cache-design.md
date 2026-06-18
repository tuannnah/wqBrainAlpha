# Thiết kế: Chuyển sang Postgres + tải sẵn toàn bộ data WQB

- **Ngày:** 2026-06-19
- **Trạng thái:** Đã duyệt thiết kế, chờ review spec
- **Bối cảnh:** Tool WQ Brain Auto-Alpha hiện cache data-fields/operators vào SQLite
  (`sqlite:///wq_alpha.db`) qua `FieldRepository`/`OperatorRepository` với TTL 30
  ngày. Người dùng đã cài PostgreSQL và muốn (a) chuyển backend sang Postgres,
  (b) tải sẵn toàn bộ data tài khoản có quyền để lần sau không phải chờ tải.

## Mục tiêu

1. Đổi backend lưu trữ từ SQLite sang PostgreSQL, không sửa logic nghiệp vụ.
2. Di trú toàn bộ dữ liệu SQLite hiện có sang Postgres (không mất lịch sử
   alpha/simulation/failure/submission).
3. Chủ động fetch toàn bộ datafields + operators + datasets cho mọi tổ hợp
   `region × universe × delay` tài khoản có quyền, nạp vào Postgres, có thể chạy
   lại (resume) để chỉ làm phần còn thiếu.

## Phi mục tiêu (YAGNI)

- Không mở rộng `_migrate_add_columns` (vá ADD COLUMN) sang Postgres — Postgres bắt
  đầu sạch nên `create_all` là đủ.
- Không xây endpoint khám phá tổ hợp động từ WQB (dùng bảng hằng + probe).
- Không đồng bộ hai chiều SQLite ↔ Postgres; migrate là thao tác một lần.
- Không tự động chạy `warm-cache` theo lịch; là lệnh chạy tay.

## Kiến trúc tổng thể

Ba phần độc lập, triển khai tuần tự. Mỗi phần có thể test riêng.

```
Phần 1: Hạ tầng Postgres   ->  Phần 2: Migrate dữ liệu  ->  Phần 3: warm-cache
(driver + DATABASE_URL)        (lệnh migrate-sqlite)        (lệnh warm-cache)
```

---

## Phần 1 — Backend Postgres

**Mục tiêu:** chạy được trên Postgres chỉ bằng đổi `DATABASE_URL`.

### Thay đổi
- `requirements.txt`: thêm `psycopg[binary]>=3`.
- `DATABASE_URL` dạng `postgresql+psycopg://user:pass@host:port/dbname`.
- `src/storage/db.py`:
  - `make_engine()` đã xử lý nhánh non-sqlite (không set `check_same_thread`,
    không gắn WAL pragma) — giữ nguyên.
  - `init_db()` đã chỉ gọi `_migrate_add_columns` khi backend là sqlite — giữ
    nguyên; Postgres chỉ chạy `create_all`.
- `.env.example`: thêm dòng mẫu URL Postgres (comment), giữ SQLite làm mặc định.

### Tiêu chí hoàn thành
- `init_db()` trên URL Postgres tạo đủ bảng từ `Base.metadata`.
- Test hiện có vẫn xanh khi `DATABASE_URL` là SQLite (không hồi quy).

---

## Phần 2 — Migrate SQLite → Postgres (một lần)

**Mục tiêu:** copy toàn bộ bảng từ `wq_alpha.db` sang Postgres, không mất lịch sử.

### Thành phần
- Hàm `migrate_all(source_url, dest_url)` trong `src/storage/migrate.py`:
  - Mở engine nguồn (SQLite) và đích (Postgres).
  - `init_db(dest_engine)` để tạo schema đích.
  - Với mỗi model theo **thứ tự tôn trọng khóa ngoại**:
    `DataFieldModel`, `FetchStateModel`, `OperatorModel`, `InvalidFieldModel`,
    `AlphaModel` (trước, vì simulations/submissions tham chiếu),
    `SimulationModel`, `FailureModel`, `SubmissionModel`.
  - Đọc toàn bộ rows từ nguồn → ghi sang đích bằng `session.merge()` (idempotent:
    chạy lại không nhân đôi, cập nhật theo khóa chính).
  - Trả về dict `{table_name: số_rows_đã_copy}` để in tổng kết.
- Lệnh Typer `migrate-sqlite` trong `main.py`:
  - Tham số `--source` (mặc định `sqlite:///wq_alpha.db`), `--dest` (mặc định
    `settings.database_url`).
  - In bảng tổng kết số rows mỗi bảng (rich).

### Xử lý lỗi
- Nguồn không tồn tại → báo lỗi rõ ràng, không tạo bảng rỗng vô nghĩa.
- Chạy lại an toàn nhờ `merge` (upsert theo PK).

### Tiêu chí hoàn thành
- Sau migrate, mỗi bảng ở đích có số rows bằng nguồn.
- Chạy `migrate-sqlite` lần hai không làm tăng số rows (idempotent).

---

## Phần 3 — Lệnh `warm-cache` (tải sẵn toàn bộ)

**Mục tiêu:** fetch hết datafields + operators + datasets cho mọi tổ hợp tài khoản
có quyền, nạp vào DB, resume được.

### Thành phần

**1. Ma trận tổ hợp — `src/data/universe_matrix.py`**
- Hằng `WQB_MATRIX`: ánh xạ `region -> {universes: [...], delays: [...]}` theo ma
  trận WQB đã biết (USA, EUR, ASI, CHN, GLB, JPN, KOR, TWN, HKG, AMR... với các
  universe TOP3000/TOP1000/TOP500/TOP200... và delay 0,1).
- Hàm `iter_scopes(regions=None, delays=None) -> Iterator[tuple[region, universe, delay]]`
  sinh ra mọi tổ hợp, có thể lọc theo `regions`/`delays` truyền vào.
- Ghi chú: bảng hằng là nguồn sự thật; tổ hợp tài khoản không có quyền sẽ được
  phát hiện qua probe (xem dưới), không cần bảng phải khớp tuyệt đối với quyền.

**2. Bộ chạy warm-cache — `src/data/warm_cache.py`**
- Hàm `warm_cache(field_repo, operator_repo, scopes, *, force=False, sleep_s=2.0,
  on_event=None) -> WarmCacheReport`:
  - Operators: gọi `operator_repo.ensure(force=force)` một lần (không theo scope).
  - Với mỗi scope:
    - Nếu không `force` và `field_repo._is_cached(...)` → bỏ qua (resume), đếm
      `skipped`.
    - Ngược lại gọi `field_repo.get_fields(region, universe, delay, force_reload=force)`.
    - Nếu trả 0 field hoặc lỗi quyền → ghi `fetch_state.status="no_access"` và đếm
      `no_access` (không coi là lỗi cứng).
    - Nghỉ `sleep_s` giây giữa các scope để giảm 429.
  - Phát sự kiện tiến độ qua `on_event` (giống mẫu `AutoEvent` trong pipeline).
  - Trả `WarmCacheReport(fetched, skipped, no_access, errors)`.
- Tận dụng retry 429 sẵn có ở `WQBrainClient._send_with_rate_limit` (tối đa 5 lần,
  tôn trọng `Retry-After`) — không lặp lại logic backoff.

**3. Trạng thái `no_access`**
- `FieldRepository` thêm helper `mark_no_access(region, universe, delay)` ghi
  `FetchStateModel.status="no_access"` để lần sau bỏ qua nhanh (resume).
- `_is_cached` giữ nguyên (chỉ true khi `status=="complete"`).

**4. Lệnh Typer `warm-cache` trong `main.py`**
- Cờ:
  - `--regions` (CSV, mặc định: tất cả region trong `WQB_MATRIX`).
  - `--delays` (CSV, mặc định: `0,1`).
  - `--force` (bỏ qua cache, tải lại hết).
  - `--sleep` (giây nghỉ giữa scope, mặc định 2.0).
- In tiến độ realtime (rich) và bảng tổng kết cuối: số scope fetched/skipped/
  no_access/errors.

### Luồng dữ liệu
```
warm-cache CLI
  -> iter_scopes(WQB_MATRIX, regions, delays)
  -> với mỗi scope: FieldRepository.get_fields() (resume nếu đã cache)
       -> WQBrainClient.get(/data-fields) [retry 429 tự động]
       -> _replace_in_db + _update_state(status=complete)
  -> OperatorRepository.ensure() (một lần)
  -> WarmCacheReport -> bảng tổng kết
```

### Xử lý lỗi
- 429: backoff sẵn ở client.
- 403/empty: đánh dấu `no_access`, tiếp tục.
- Lỗi mạng giữa chừng: state đã lưu từng scope nên chạy lại tiếp tục được.
- Một scope lỗi không làm dừng toàn bộ; gom vào `errors` của report.

### Tiêu chí hoàn thành
- Chạy `warm-cache` lần hai (không `--force`) bỏ qua toàn bộ scope đã complete.
- Scope không có quyền được đánh dấu `no_access` và bỏ qua ở lần sau.
- Report phản ánh đúng số fetched/skipped/no_access/errors.

---

## Chiến lược test (TDD, không gọi mạng thật)

Dùng fake `WQBrainClient` (trả response giả) và session factory in-memory SQLite.

- **Phần 1:** test `init_db()` tạo đủ bảng; test không hồi quy trên SQLite.
- **Phần 2:** test `migrate_all` copy đúng số rows mọi bảng; test idempotent (chạy
  hai lần không nhân đôi); test thứ tự FK (alphas trước simulations).
- **Phần 3:**
  - `iter_scopes` sinh đúng tổ hợp, lọc theo `regions`/`delays`.
  - `warm_cache` resume: scope đã complete bị skip, không gọi client.
  - `warm_cache` probe: scope trả empty/403 → đánh dấu `no_access`.
  - `warm_cache` report đếm đúng các nhóm.
  - Mỗi task một commit; code/commit/giao tiếp bằng tiếng Việt.

## Thứ tự triển khai

1. Phần 1 (hạ tầng) → 2. Phần 2 (migrate) → 3. Phần 3 (warm-cache).
Mỗi phần xanh test trước khi sang phần sau.
