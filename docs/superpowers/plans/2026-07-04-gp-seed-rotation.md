# Xoay vòng seed family/novel theo batch (round-robin) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Khi `GPIdeaSource.next_batch()` được gọi nhiều lần (mỗi lần dựng một `GPEngine` mới),
toàn bộ seed hợp lệ từ `all_seed_cores()` (family + novel, hiện 150 seed) phải lần lượt được dùng
làm quần thể ban đầu — không chỉ mãi mãi 30 seed đầu tiên theo thứ tự cố định như hiện tại.

**Architecture:** Thêm tham số `seed_offset: int = 0` xuyên 3 lớp
(`GPIdeaSource.next_batch()` → `GPEngine` → `init_population()`), mặc định `0` để không đổi hành
vi ở mọi call site khác. `init_population()` dùng hàm mới `_rotating_slice()` để cắt một lô
`population_size` seed bắt đầu từ `seed_offset % len(valid_seeds)`, nối vòng lại đầu danh sách
nếu tràn cuối. `GPIdeaSource` tính `seed_offset = self._batch * self.pop_size` — mỗi batch dùng
đúng 1 lô kế tiếp, quay lại từ đầu sau khi đã dùng hết toàn bộ danh sách.

**Tech Stack:** Python 3.12, numpy (RNG), pytest. Không thêm dependency mới.

## Global Constraints

- Mọi tham số mới đều có default giữ nguyên hành vi cũ (`seed_offset: int = 0`) — không được phá
  bất kỳ call site/test hiện có nào (CLI `generate`, `local_engine_test.py`, toàn bộ test cũ của
  `init_population`/`GPEngine`/`GPIdeaSource`).
- Test và code chạy bằng venv của project: `./venv/Scripts/python.exe -m pytest ...` (python hệ
  thống thiếu dependency `lark`/`psycopg`, KHÔNG dùng để chạy test/script trong plan này).
- Code/comment/commit message bằng tiếng Việt có dấu đầy đủ, theo đúng văn phong hiện có trong
  từng file (xem docstring/comment gốc trước khi thêm).
- TDD bắt buộc: viết test trước, chạy thấy FAIL, rồi mới sửa code cho PASS.
- Mỗi Task kết thúc bằng 1 commit riêng (không gộp nhiều Task vào 1 commit).
- Spec gốc: `docs/superpowers/specs/2026-07-04-gp-seed-rotation-design.md`.

---

## Task 1: `_rotating_slice` + `init_population(seed_offset=...)`

**Files:**
- Modify: `src/gp/init.py`
- Test: `tests/unit/test_gp_init.py`

**Interfaces:**
- Consumes: không phụ thuộc task khác (đây là task đầu tiên, thuần nội bộ `src/gp/init.py`).
- Produces: `_rotating_slice(items: list[Node], offset: int, count: int) -> list[Node]` (hàm
  module-level, không export ra ngoài file, dùng nội bộ). `init_population(...)` có thêm tham số
  keyword `seed_offset: int = 0` — Task 2 sẽ gọi hàm này với `seed_offset=self.seed_offset`.

- [ ] **Step 1: Viết test cho `_rotating_slice` (RED)**

Mở `tests/unit/test_gp_init.py`, thêm vào cuối file:

```python
def test_rotating_slice_offset_0_giu_nguyen_lat_cat_thuong():
    from src.gp.init import _rotating_slice
    items = [1, 2, 3, 4, 5]
    assert _rotating_slice(items, offset=0, count=3) == [1, 2, 3]


def test_rotating_slice_offset_giua_danh_sach_khong_tran():
    from src.gp.init import _rotating_slice
    items = [1, 2, 3, 4, 5]
    assert _rotating_slice(items, offset=1, count=3) == [2, 3, 4]


def test_rotating_slice_offset_gay_wrap_around():
    from src.gp.init import _rotating_slice
    items = [1, 2, 3, 4, 5]
    assert _rotating_slice(items, offset=4, count=3) == [5, 1, 2]


def test_rotating_slice_offset_boi_so_do_dai_quay_lai_offset_0():
    from src.gp.init import _rotating_slice
    items = [1, 2, 3, 4, 5]
    assert _rotating_slice(items, offset=10, count=3) == [1, 2, 3]


def test_rotating_slice_danh_sach_rong_tra_rong():
    from src.gp.init import _rotating_slice
    assert _rotating_slice([], offset=5, count=3) == []
```

- [ ] **Step 2: Chạy test, xác nhận FAIL vì `_rotating_slice` chưa tồn tại**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_gp_init.py -k rotating_slice -v`
Expected: 5 lỗi `ImportError: cannot import name '_rotating_slice'`.

- [ ] **Step 3: Thêm `_rotating_slice` vào `src/gp/init.py`**

Chèn ngay trước hàm `init_population` (sau `ramped_half_and_half`, dòng 106 hiện tại):

```python
def _rotating_slice(items: list[Node], offset: int, count: int) -> list[Node]:
    """Lát cắt xoay vòng bắt đầu tại offset % len(items), nối vòng lại đầu danh sách nếu
    tràn cuối. Dùng để mỗi batch GP (xem GPIdeaSource) dùng một lô seed khác nhau thay vì
    luôn cố định items[:count] — qua nhiều batch, toàn bộ seed lần lượt được dùng."""
    n = len(items)
    if n == 0:
        return []
    start = offset % n
    end = start + count
    if end <= n:
        return items[start:end]
    return items[start:] + items[: end - n]
```

- [ ] **Step 4: Chạy lại test `_rotating_slice`, xác nhận PASS**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_gp_init.py -k rotating_slice -v`
Expected: 5 passed.

- [ ] **Step 5: Viết test cho `init_population(seed_offset=...)` (RED)**

Thêm vào cuối `tests/unit/test_gp_init.py`:

```python
def test_init_population_seed_offset_mac_dinh_0_giu_nguyen_hanh_vi_cu():
    seeds = [Call(op="rank", args=(Field(f),)) for f in _FIELDS]
    rng = np.random.default_rng(5)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=2, seed_cores=seeds, fields=_FIELDS, max_depth=5,
    )
    assert [ind.expr for ind in pop] == seeds[:2]


def test_init_population_seed_offset_xoay_sang_lo_ke_tiep():
    seeds = [Call(op="rank", args=(Field(f),)) for f in _FIELDS]  # 3 seed (close/volume/returns)
    rng = np.random.default_rng(5)
    registry = default_registry()
    pop = init_population(
        registry, rng, population_size=2, seed_cores=seeds, fields=_FIELDS, max_depth=5,
        seed_offset=2,
    )
    # offset=2, count=2, len=3 -> wrap: seeds[2:3] + seeds[0:1]
    assert [ind.expr for ind in pop] == [seeds[2], seeds[0]]
```

- [ ] **Step 6: Chạy test, xác nhận FAIL**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_gp_init.py -k seed_offset -v`
Expected: `TypeError: init_population() got an unexpected keyword argument 'seed_offset'` (test thứ
2), và test thứ nhất PASS "tình cờ" vì default hiện tại đã đúng — không sao, mục đích chính là
test thứ 2 phải FAIL trước khi sửa.

- [ ] **Step 7: Sửa `init_population` dùng `_rotating_slice`**

Trong `src/gp/init.py`, sửa chữ ký và thân hàm `init_population` (dòng 108-129 hiện tại):

```python
def init_population(
    registry: OperatorRegistry,
    rng: np.random.Generator,
    population_size: int,
    seed_cores: list[Node],
    fields: tuple[str, ...],
    max_depth: int,
    seed_offset: int = 0,
) -> list[Individual]:
    """Quần thể ban đầu: ưu tiên seed kinh nghiệm, lấp đầy phần còn lại bằng ramped
    half-and-half. Seed/cây vượt max_depth bị loại + log warning (không crash).

    ``seed_offset`` chọn lô seed nào được dùng khi số seed hợp lệ nhiều hơn
    ``population_size`` (xoay vòng qua ``_rotating_slice`` thay vì luôn cố định
    seed_cores[:population_size]) — caller (GPEngine/GPIdeaSource) tăng dần offset mỗi
    batch để qua nhiều lần gọi, toàn bộ seed hợp lệ đều được dùng."""
    valid_seeds = [t for t in seed_cores if DepthVisitor().visit(t) <= max_depth]
    dropped = len(seed_cores) - len(valid_seeds)
    if dropped:
        logger.warning("init_population: bỏ qua %d seed vượt max_depth=%d", dropped, max_depth)

    if len(valid_seeds) >= population_size:
        chosen = _rotating_slice(valid_seeds, seed_offset, population_size)
        return [Individual(expr=t) for t in chosen]

    remaining = population_size - len(valid_seeds)
    filler = ramped_half_and_half(registry, rng, remaining, min_depth=2, max_depth=max_depth, fields=fields)
    return [Individual(expr=t) for t in valid_seeds + filler]
```

- [ ] **Step 8: Chạy lại toàn bộ test file, xác nhận PASS hết (kể cả test cũ)**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_gp_init.py -v`
Expected: toàn bộ PASS (bao gồm các test cũ `test_init_population_uses_all_seeds_...`,
`test_init_population_caps_seeds_...`, `test_init_population_all_individuals_...` — không được có
cái nào vỡ, vì `seed_offset` mặc định 0 giữ nguyên hành vi).

- [ ] **Step 9: Commit**

```bash
git add src/gp/init.py tests/unit/test_gp_init.py
git commit -m "$(cat <<'EOF'
feat(gp): them seed_offset xoay vong seed khi khoi tao quan the

init_population() truoc gio luon lay dung seed_cores[:population_size]
theo thu tu co dinh - qua bao nhieu lan goi cung chi dung dung 1 lo seed
dau tien, bo phi phan con lai. Them _rotating_slice() + tham so
seed_offset (mac dinh 0, khong doi hanh vi cu) de caller (GPEngine o
Task sau) chon lo seed khac nhau moi lan goi.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `GPEngine` truyền `seed_offset` xuống `init_population`

**Files:**
- Modify: `src/gp/engine.py`
- Test: `tests/unit/test_gp_engine.py`

**Interfaces:**
- Consumes: `init_population(..., seed_offset: int = 0)` từ Task 1 (đã có, giữ nguyên chữ ký).
- Produces: `GPEngine.__init__(..., seed_offset: int = 0)` — Task 3 (`GPIdeaSource`) sẽ gọi
  `GPEngine(..., seed_offset=<int>)`.

- [ ] **Step 1: Viết test xác nhận `seed_offset` được truyền xuống `init_population` (RED)**

Thêm vào cuối `tests/unit/test_gp_engine.py`:

```python
def test_engine_passes_seed_offset_to_init_population(small_panel, repo, cfg, monkeypatch) -> None:  # noqa: ANN001
    """seed_offset (round-robin seed family qua batch, xem GPIdeaSource) phải truyền
    nguyên vẹn xuống init_population() — spy bằng monkeypatch thay vì dựng seed_cores
    thật để test nhanh, không phụ thuộc nội dung families.py."""
    captured: dict = {}

    def _fake_init_population(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.gp.engine.init_population", _fake_init_population)
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=0, seed=42, seed_offset=8,
    )
    eng.run()
    assert captured["seed_offset"] == 8


def test_engine_seed_offset_mac_dinh_0(small_panel, repo, cfg, monkeypatch) -> None:  # noqa: ANN001
    captured: dict = {}

    def _fake_init_population(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.gp.engine.init_population", _fake_init_population)
    eng = GPEngine(
        data=small_panel, repo=repo, config=cfg, registry=default_registry(),
        pop_size=4, n_generations=0, seed=42,
    )
    eng.run()
    assert captured["seed_offset"] == 0
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py -k seed_offset -v`
Expected: `TypeError: GPEngine.__init__() got an unexpected keyword argument 'seed_offset'`.

- [ ] **Step 3: Sửa `GPEngine.__init__` và `run()`**

Trong `src/gp/engine.py`, sửa `__init__` (dòng 78-107 hiện tại) — thêm `seed_offset` ngay sau
`seed`:

```python
    def __init__(
        self,
        data: MarketData,
        repo: MiniBrainRepository,
        config: PortfolioConfig,
        registry: OperatorRegistry,
        *,
        pop_size: int = 50,
        n_generations: int = 5,
        max_depth: int = 7,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.3,
        seed: int = 42,
        seed_offset: int = 0,
        data_window: str = "default",
        with_llm_seeds: bool = False,
        n_jobs: int = 1,
    ) -> None:
        self.data = data
        self.repo = repo
        self.config = config
        self.registry = registry
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.seed_offset = seed_offset
        self.data_window = data_window
        self.with_llm_seeds = with_llm_seeds
        self.n_jobs = n_jobs
```

Sửa lời gọi `init_population` trong `run()` (dòng 296-303 hiện tại):

```python
        population = init_population(
            registry=self.registry,
            rng=rng,
            population_size=self.pop_size,
            seed_cores=seed_cores,
            fields=fields,
            max_depth=self.max_depth,
            seed_offset=self.seed_offset,
        )
```

- [ ] **Step 4: Chạy lại test, xác nhận PASS**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py -k seed_offset -v`
Expected: 2 passed.

- [ ] **Step 5: Chạy toàn bộ test file `test_gp_engine.py`, xác nhận không hồi quy**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_gp_engine.py -v`
Expected: toàn bộ PASS (test cũ không đổi vì `seed_offset` mặc định 0).

- [ ] **Step 6: Commit**

```bash
git add src/gp/engine.py tests/unit/test_gp_engine.py
git commit -m "$(cat <<'EOF'
feat(gp): GPEngine nhan va truyen seed_offset xuong init_population

Them tham so seed_offset (mac dinh 0, khong doi hanh vi cu) vao
GPEngine.__init__, truyen nguyen ven xuong init_population() trong
run(). Chuan bi cho GPIdeaSource (Task sau) tinh offset tang dan moi
batch de xoay vong seed family/novel.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `GPIdeaSource.next_batch()` tính `seed_offset` tăng dần theo `pop_size`

**Files:**
- Modify: `src/app/closed_loop_adapters.py`
- Test: `tests/unit/test_closed_loop_adapters.py`

**Interfaces:**
- Consumes: `GPEngine(..., seed_offset: int = 0)` từ Task 2.
- Produces: không có consumer nào khác trong repo hiện tại — đây là điểm cuối của chuỗi thay đổi
  (hành vi quan sát được qua chạy thật `main.py` mục 5 / CLI `closed-loop`).

- [ ] **Step 1: Viết test xác nhận `seed_offset` tăng đúng theo `pop_size` mỗi batch (RED)**

Thêm vào cuối `tests/unit/test_closed_loop_adapters.py`:

```python
def test_gp_idea_source_seed_offset_tang_theo_pop_size(small_panel, repo) -> None:  # noqa: ANN001
    """Moi batch phai dung 1 lo seed khac nhau (round-robin) - offset tang dung pop_size,
    khop cach seed=base_seed+batch da co san."""
    from unittest.mock import patch
    cfg = PortfolioConfig(neutralization=Neutralization.NONE, decay=0, truncation=0.10,
                          scale_book=1.0, delay=1)
    src = GPIdeaSource(small_panel, repo, cfg, default_registry(),
                       pop_size=6, n_generations=0, base_seed=42, top_k=5, max_corr=0.99)
    offsets_seen: list[int] = []

    class _StubEngine:
        def __init__(self, *a, seed_offset: int, **k) -> None:
            offsets_seen.append(seed_offset)
        def run(self):
            from src.gp.engine import GPRunResult
            return GPRunResult(generations_run=0, final_population=[], best_by_sharpe=None,
                               n_evaluated=0, n_passed=0, seed=42)

    with patch("src.app.closed_loop_adapters.GPEngine", _StubEngine):
        src.next_batch()
        src.next_batch()
        src.next_batch()
    assert offsets_seen == [0, 6, 12]  # tang dung pop_size (6) moi batch
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop_adapters.py -k seed_offset -v`
Expected: `TypeError: _StubEngine.__init__() missing 1 required keyword-only argument: 'seed_offset'`
(vì `GPIdeaSource.next_batch()` hiện tại chưa truyền `seed_offset` cho `GPEngine`).

- [ ] **Step 3: Sửa `GPIdeaSource.next_batch()`**

Trong `src/app/closed_loop_adapters.py`, sửa hàm `next_batch` (dòng 75-89 hiện tại):

```python
    def next_batch(self) -> list[ShortlistCandidate]:
        seed = self.base_seed + self._batch
        seed_offset = self._batch * self.pop_size
        self._batch += 1
        engine = GPEngine(
            data=self._data, repo=self._repo, config=self._config, registry=self._registry,
            pop_size=self.pop_size, n_generations=self.n_generations, seed=seed,
            seed_offset=seed_offset,
        )
        pool: Any = self._repo.load_pool() or None
        # GPEngine.run() -> GPRunResult; Protocol _RunsGP đòi _GPRunResultLike với
        # list[_GPIndividualLike] — list là invariant nên cast qua Any để truyền qua.
        engine_any: Any = engine
        return generate_many(
            gp_engine=engine_any, cfg=self._config, data=self._data,
            top_k=self.top_k, max_corr=self.max_corr, pool=pool,
        )
```

- [ ] **Step 4: Chạy lại test, xác nhận PASS**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop_adapters.py -k seed_offset -v`
Expected: 1 passed.

- [ ] **Step 5: Chạy toàn bộ test file, xác nhận không hồi quy (đặc biệt test seed cũ)**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_closed_loop_adapters.py -v`
Expected: toàn bộ PASS, bao gồm `test_gp_idea_source_yields_candidates_and_advances_seed` (kiểm
`seed` — KHÔNG phải `seed_offset` — vẫn tăng đúng `[42, 43]` như cũ, không bị ảnh hưởng).

- [ ] **Step 6: Commit**

```bash
git add src/app/closed_loop_adapters.py tests/unit/test_closed_loop_adapters.py
git commit -m "$(cat <<'EOF'
feat(app): GPIdeaSource xoay vong seed family/novel qua tung batch

next_batch() gio tinh seed_offset = batch * pop_size, truyen vao
GPEngine moi lan goi. Ket hop voi _rotating_slice (Task truoc), moi
batch dung dung 1 lo seed ke tiep thay vi mai mai chi dung 30 seed dau
tien co dinh - qua du batch, toan bo seed hop le trong
all_seed_cores() (family + novel) deu duoc dung it nhat 1 lan.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Kiểm tra hồi quy toàn cục + xác nhận thực tế trên seed pool thật

**Files:**
- Không sửa file mới — chỉ chạy kiểm chứng.

**Interfaces:**
- Consumes: toàn bộ thay đổi Task 1-3.
- Produces: xác nhận cuối cùng để đóng plan.

- [ ] **Step 1: Chạy toàn bộ test suite**

Run: `./venv/Scripts/python.exe -m pytest -q`
Expected: tất cả PASS trừ `tests/test_db_postgres.py::test_make_engine_postgres_backend` (lỗi môi
trường có sẵn, thiếu module `psycopg`, KHÔNG liên quan plan này — nếu thấy lỗi nào khác, dừng lại
điều tra trước khi qua bước tiếp).

- [ ] **Step 2: Xác nhận thực tế bằng seed pool thật (150 seed gia đình đã sửa)**

Run:
```bash
./venv/Scripts/python.exe -c "
import numpy as np
import src.operators_local  # noqa: F401
from src.gp.init import init_population
from src.gp.seeds import all_seed_cores
from src.lang.registry import default_registry

seeds = all_seed_cores(with_llm=False)
print('tong so seed hop le:', len(seeds))
registry = default_registry()
seen_ids = set()
pop_size = 30
n_batches = -(-len(seeds) // pop_size)  # ceil division
for batch in range(n_batches + 1):
    offset = batch * pop_size
    pop = init_population(
        registry, np.random.default_rng(42), population_size=pop_size,
        seed_cores=seeds, fields=('close', 'volume', 'returns'), max_depth=7,
        seed_offset=offset,
    )
    for ind in pop:
        seen_ids.add(id(ind.expr))
print(f'so seed KHAC NHAU da dung qua {n_batches + 1} batch:', len(seen_ids))
"
```

Expected: dòng cuối in ra một số RẤT gần hoặc bằng `tổng số seed hợp lệ` (ví dụ ~150), CHỨNG MINH
qua đủ batch, round-robin đã phủ hết toàn bộ seed — khác hẳn hành vi cũ (mãi mãi chỉ 30 seed cố
định, con số này sẽ luôn là 30 nếu bug còn tồn tại).

- [ ] **Step 3: Không commit gì ở Task này** (chỉ là bước xác nhận, không đổi code).

---

## Ghi chú cho người review / thực thi plan

- 3 Task đầu độc lập về mặt logic (mỗi Task 1 file nguồn + 1 file test) nhưng PHẢI làm tuần tự
  (Task 2 cần `init_population(seed_offset=...)` từ Task 1 đã tồn tại để không phải sửa lại;
  Task 3 cần `GPEngine(seed_offset=...)` từ Task 2). Không chạy song song 3 Task này bằng subagent
  độc lập.
- Không có Task nào đụng `src/generation/families.py`, `src/backtest/gate.py`/`gates.py` — đúng
  như phạm vi đã chốt trong spec ("Vấn đề A" gate kép KHÔNG nằm trong plan này).
