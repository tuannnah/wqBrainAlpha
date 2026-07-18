# Plan: Cải thiện tốc độ + chất lượng vòng kín

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Giảm ≥50% thời gian mỗi vòng "Sinh batch ý tưởng" và chống bão hoà nguồn ý tưởng, KHÔNG giảm chất lượng tìm kiếm (spec: `docs/superpowers/specs/2026-07-18-toc-do-chat-luong-vong-kin-design.md`).

**Architecture:** 3 pha — A: cắt công việc mà kết quả chắc chắn bị vứt (GP khi budget cạn, cá thể degenerate, backtest trùng); B: reseed epoch + đa dạng field; C: song song hoá phần backtest thuần. Mọi thay đổi giữ bất biến: cùng seed → cùng kết quả tìm kiếm (chốt chặn = test C2).

**Tech Stack:** Python 3.12, pytest, numpy, SQLite (repo), concurrent.futures (Pha C).

## Global Constraints

- Code/comment/log/commit message: **tiếng Việt** (giữ nguyên thuật ngữ kỹ thuật).
- TDD bắt buộc: test fail trước, implement sau. Mỗi task đúng 1 commit.
- KHÔNG đổi ngưỡng gate local/Brain, KHÔNG đổi logic submit.
- Toàn bộ suite hiện có (~1 218 test) phải xanh sau mỗi task: `python -m pytest -q`.
- Windows: multiprocessing dùng spawn; mọi hàm worker phải ở module-level (picklable).
- Determinism: mọi randomness qua `rng` inject; không gọi `np.random.default_rng()` nội bộ mới.

---

### Task A1: Tắt tiến hoá GP khi gp_budget cạn

**Files:**
- Modify: `src/pipeline/closed_loop.py` (constructor ~dòng 190, nhánh gp_budget ~dòng 382)
- Modify: `src/app/closed_loop_adapters.py` (GPIdeaSource ~842, các wrapper Curated ~654 / AltData ~772 / Combiner ~947, NearMiss ở `src/generation/near_miss_variants.py:194`, wiring `build_closed_loop` ~1242)
- Test: `tests/unit/test_closed_loop.py`, `tests/unit/test_closed_loop_adapters.py`

**Interfaces:**
- Produces: `ClosedLoop.__init__(..., on_gp_budget_exhausted=None)` — callable `(bool) -> None`, gọi ĐÚNG MỘT LẦN với `True` khi `gp_sims_used >= max_gp_sims` lần đầu.
- Produces: `set_gp_budget_exhausted(flag: bool) -> None` trên GPIdeaSource + mọi wrapper (ủy quyền xuống fallback, pattern y hệt `set_saturated_families`).
- Produces: khi cờ bật, `GPIdeaSource.next_batch()` trả `[]` NGAY (không chạy `_run_one_batch`). Task B1 sẽ gọi lại với `False` khi reseed.

- [ ] **Bước 1: Viết test fail**

Thêm vào `tests/unit/test_closed_loop_adapters.py`:

```python
class _EngineNoTrongTest:
    """GPIdeaSource khi cờ budget-cạn bật KHÔNG được đụng tới GPEngine."""


def test_gp_idea_source_bo_qua_tien_hoa_khi_budget_can(monkeypatch):
    # GPIdeaSource với cờ bật: next_batch trả [] và KHÔNG dựng GPEngine
    from src.app import closed_loop_adapters as m

    goi_engine = []
    monkeypatch.setattr(
        m, "GPEngine",
        lambda **kw: goi_engine.append(kw) or (_ for _ in ()).throw(AssertionError("không được dựng GPEngine")),
    )
    src = m.GPIdeaSource(data=None, repo=None, config=None, registry=None)
    src.set_gp_budget_exhausted(True)
    assert src.next_batch() == []
    assert goi_engine == []
    # Gọi lại False -> chạy bình thường (sẽ nổ AssertionError từ fake ở trên)
    src.set_gp_budget_exhausted(False)
    import pytest
    with pytest.raises(AssertionError):
        src.next_batch()


def test_wrapper_uy_quyen_set_gp_budget_exhausted():
    from src.app.closed_loop_adapters import CuratedIdeaSource, GPIdeaSource

    gp = GPIdeaSource(data=None, repo=None, config=None, registry=None)
    wrapper = CuratedIdeaSource(fallback=gp)
    wrapper.set_gp_budget_exhausted(True)
    assert gp._gp_budget_exhausted is True
```

Thêm vào `tests/unit/test_closed_loop.py` (dùng fake idea_source/refiner sẵn có trong file làm mẫu):

```python
def test_on_gp_budget_exhausted_goi_dung_mot_lan():
    """Khi gp_sims_used chạm max_gp_sims, callback bắn True đúng 1 lần dù nhiều candidate GP sau đó."""
    # Dựng ClosedLoop với max_gp_sims=1, refiner giả trả outcome sims_used=1 origin gp,
    # idea_source giả trả 3 batch mỗi batch 1 candidate origin "gp".
    goi = []
    loop = _lam_closed_loop(  # helper/fixture theo pattern test hiện có trong file
        max_gp_sims=1, on_gp_budget_exhausted=lambda f: goi.append(f),
    )
    loop.run()
    assert goi == [True]
```

(Điều chỉnh helper theo fixture thật trong file test — đọc các test `max_gp_sims` sẵn có (grep `gp_budget` trong tests/) và bắt chước cách dựng.)

- [ ] **Bước 2: Chạy test, xác nhận FAIL** — `python -m pytest tests/unit/test_closed_loop_adapters.py -k gp_budget -q` → AttributeError `set_gp_budget_exhausted`.

- [ ] **Bước 3: Implement**

`src/app/closed_loop_adapters.py` — trong `GPIdeaSource.__init__` thêm `self._gp_budget_exhausted = False`; thêm method (đặt cạnh `set_saturated_families`):

```python
    def set_gp_budget_exhausted(self, flag: bool) -> None:
        """A1: ClosedLoop báo trần sim GP/phiên đã chạm (True) — next_batch bỏ hẳn tiến hoá
        (kết quả trước giờ vẫn bị vứt ở gate gp_budget, bỏ chạy = không mất gì, tiết kiệm
        3–14 phút/batch). Epoch reseed (B1) gọi lại với False để mở lại."""
        self._gp_budget_exhausted = flag
```

Đầu `GPIdeaSource.next_batch()` thêm:

```python
        if self._gp_budget_exhausted:
            return []
```

Mỗi wrapper (CuratedIdeaSource, AltDataIdeaSource, CombinerIdeaSource trong file này; NearMissVariantSource trong `src/generation/near_miss_variants.py`) thêm method ủy quyền — copy đúng pattern `set_saturated_families` liền kề:

```python
    def set_gp_budget_exhausted(self, flag: bool) -> None:
        if hasattr(self._fallback, "set_gp_budget_exhausted"):
            self._fallback.set_gp_budget_exhausted(flag)
```

`src/pipeline/closed_loop.py` — constructor thêm param `on_gp_budget_exhausted=None` + `self.on_gp_budget_exhausted = on_gp_budget_exhausted`; trong `run()` thêm biến cục bộ `gp_budget_da_bao = False`; trong nhánh gp_budget (ngay trước khi dựng `outcome` stop_reason="gp_budget", ~dòng 387):

```python
                    if not gp_budget_da_bao and self.on_gp_budget_exhausted is not None:
                        gp_budget_da_bao = True
                        self.on_gp_budget_exhausted(True)
```

`build_closed_loop` (~1242, cạnh `on_family_closed`):

```python
    def on_gp_budget_exhausted(flag: bool) -> None:
        idea_source.set_gp_budget_exhausted(flag)  # type: ignore[attr-defined]
```

và truyền `on_gp_budget_exhausted=on_gp_budget_exhausted` vào `ClosedLoop(...)` cuối hàm.

- [ ] **Bước 4: Chạy test pass** — `python -m pytest tests/unit/test_closed_loop_adapters.py tests/unit/test_closed_loop.py -q` PASS.
- [ ] **Bước 5: Toàn suite** — `python -m pytest -q` xanh.
- [ ] **Bước 6: Commit** — `git commit -m "feat(engine): tắt tiến hoá GP khi gp_budget cạn (A1)"`

---

### Task A2: Lọc meaningfulness + họ-đã-đóng TRONG GP, trước backtest

**Files:**
- Modify: `src/gp/engine.py` (`GPEngine.__init__` ~85, `_evaluate_population` ~223)
- Modify: `src/app/closed_loop_adapters.py` (`GPIdeaSource._run_one_batch` ~872 truyền saturated)
- Test: `tests/unit/test_gp_engine.py`

**Interfaces:**
- Consumes: `check_meaningful(node, registry) -> tuple[bool, str]` (`src/lang/meaningfulness.py:70`), `classify_family(expr: str) -> str` (`src/reporting/diagnostics.py:59`).
- Produces: `GPEngine.__init__(..., saturated_families: frozenset[str] | set[str] = frozenset())`. Cá thể degenerate/họ-đóng: KHÔNG backtest, `fitness=None`, persist status `"failed_gate"` với fail_reasons `["degenerate: <lý do>"]` hoặc `["họ đã đóng: <họ>"]`.

- [ ] **Bước 1: Viết test fail** — thêm vào `tests/unit/test_gp_engine.py` (bắt chước fixture data/repo/registry giả sẵn có trong file):

```python
def test_evaluate_population_bo_ca_the_vo_nghia_khong_backtest(engine_fixture, monkeypatch):
    """Cá thể volume-only bị chặn TRƯỚC backtest: Backtester.run không được gọi."""
    from src.lang.parser import parse
    from src.gp.individual import Individual
    import src.gp.engine as eng_mod

    goi_backtest = []
    monkeypatch.setattr(
        eng_mod.Backtester, "run",
        lambda self, w, d: goi_backtest.append(1) or (_ for _ in ()).throw(AssertionError),
    )
    engine = engine_fixture  # GPEngine với data giả có field 'volume'
    ind = Individual(expr=parse("rank(ts_zscore(volume, 5))"))
    n_ev, n_pa = engine._evaluate_population([ind], _pool_corr_rong())
    assert ind.fitness is None
    assert goi_backtest == []


def test_evaluate_population_bo_ca_the_ho_da_dong(engine_fixture_voi_saturated):
    """engine dựng với saturated_families={'pv_reversal'} -> cá thể close-open bị bỏ, không backtest."""
    from src.lang.parser import parse
    from src.gp.individual import Individual

    engine = engine_fixture_voi_saturated  # saturated_families={"pv_reversal"}
    ind = Individual(expr=parse("multiply(-1, ts_mean(subtract(close, open), 10))"))
    n_ev, n_pa = engine._evaluate_population([ind], _pool_corr_rong())
    assert ind.fitness is None
```

- [ ] **Bước 2: Chạy fail** — `python -m pytest tests/unit/test_gp_engine.py -k "vo_nghia or ho_da_dong" -q` → FAIL (Backtester được gọi / thiếu param).

- [ ] **Bước 3: Implement**

`GPEngine.__init__`: thêm param `saturated_families: "frozenset[str] | set[str]" = frozenset()` → `self.saturated_families = frozenset(saturated_families)`.

Trong `_evaluate_population`, sau `if ind.fitness is not None: continue`, chèn TRƯỚC `_evaluate_individual`:

```python
            # A2: chặn TRƯỚC backtest — cá thể vô nghĩa/họ-đóng không tốn eval, không chiếm
            # suất NSGA-II (fitness=None -> bị loại khỏi chọn lọc), vẫn persist để avoid-list học.
            ok, ly_do = check_meaningful(ind.expr, self.registry)
            ho: str | None = None
            if ok:
                expr_str = ind.expr.accept(Serializer())
                ho = classify_family(expr_str)
                if ho not in self.saturated_families:
                    ho = None  # không vi phạm
            if not ok or ho is not None:
                reasons = [f"degenerate: {ly_do}"] if not ok else [f"họ đã đóng: {ho}"]
                self._persist(ind, "failed_gate", reasons, None, None)
                ind.fitness = None
                n_evaluated += 1
                continue
```

Import bổ sung đầu file: `from src.lang.meaningfulness import check_meaningful`, `from src.reporting.diagnostics import classify_family` (Serializer đã có import trong engine — kiểm tra, nếu chưa thì thêm từ `src.lang.visitors`).

LƯU Ý: `ind.fitness = None` giữ nguyên None nhưng vòng lặp hiện skip theo `if ind.fitness is not None` — cá thể này sẽ bị đánh giá LẠI ở lần gọi sau. Tránh: thêm slot `Individual.da_loai: bool = False` là quá đà (YAGNI) — thay vào đó dùng chính điều kiện đã persist? Đơn giản nhất: gắn `ind.fitness = None` và chấp nhận check_meaningful chạy lại (rẻ, thuần AST, không backtest) — chi phí không đáng kể so với backtest. GIỮ cách này.

`GPIdeaSource._run_one_batch`: thêm `saturated_families=self._saturated` vào constructor `GPEngine(...)`.

- [ ] **Bước 4: Chạy pass** — test mới PASS; `python -m pytest tests/unit/test_gp_engine.py tests/integration/test_gp_engine_run.py -q` xanh.
- [ ] **Bước 5: Toàn suite** — `python -m pytest -q` xanh.
- [ ] **Bước 6: Commit** — `git commit -m "feat(gp): lọc degenerate + họ-đóng trước backtest trong GP (A2)"`

---

### Task A3: Cache backtest xuyên batch theo canonical_hash

**Files:**
- Modify: `src/gp/engine.py` (`__init__`, `_evaluate_individual` ~111)
- Modify: `src/app/closed_loop_adapters.py` (`GPIdeaSource.__init__` + `_run_one_batch`)
- Test: `tests/unit/test_gp_engine.py`

**Interfaces:**
- Produces: `GPEngine.__init__(..., eval_cache: "dict[str, tuple] | None" = None)`. Entry: `canonical_hash -> ("ok", bt, metrics) | ("error", fail_reasons)`. Phần phụ thuộc pool (gate, pool_rho, fitness vector) LUÔN tính lại tươi.
- Produces: `GPIdeaSource` giữ `self._eval_cache: dict[str, tuple] = {}` truyền vào mọi GPEngine; cap 5 000 entry (vượt → `clear()`).

- [ ] **Bước 1: Test fail**

```python
def test_eval_cache_hit_khong_backtest_lai(engine_fixture_voi_cache, monkeypatch):
    """Cùng canonical_hash lần 2: Backtester.run không được gọi lại, kết quả giống hệt."""
    from src.lang.parser import parse
    from src.gp.individual import Individual
    import src.gp.engine as eng_mod

    engine, cache = engine_fixture_voi_cache  # GPEngine(eval_cache=cache), data giả nhiều field
    i1 = Individual(expr=parse("ts_mean(subtract(close, open), 10)"))
    fv1, st1, rs1, bt1 = engine._evaluate_individual(i1, _pool_corr_rong())
    assert len(cache) == 1

    so_lan = {"n": 0}
    that = eng_mod.Backtester.run
    monkeypatch.setattr(
        eng_mod.Backtester, "run",
        lambda self, w, d: so_lan.__setitem__("n", so_lan["n"] + 1) or that(self, w, d),
    )
    i2 = Individual(expr=parse("ts_mean(subtract(close, open), 10)"))
    fv2, st2, rs2, bt2 = engine._evaluate_individual(i2, _pool_corr_rong())
    assert so_lan["n"] == 0
    assert st2 == st1
    import numpy as np
    np.testing.assert_array_equal(bt2.daily_pnl, bt1.daily_pnl)
```

- [ ] **Bước 2: Chạy fail** — thiếu param `eval_cache` → TypeError.

- [ ] **Bước 3: Implement**

`GPEngine.__init__`: `eval_cache: "dict[str, tuple] | None" = None` → `self.eval_cache = eval_cache`.

`_evaluate_individual` tách phần thuần: đầu hàm tính `ch = ind.expr.accept(CanonicalHasher())`; tra cache:

```python
        ch = ind.expr.accept(CanonicalHasher())
        cached = self.eval_cache.get(ch) if self.eval_cache is not None else None
        if cached is not None and cached[0] == "error":
            return None, "error", list(cached[1]), None
        if cached is not None:
            _tag, bt, metrics = cached
        else:
            # ... (3 khối try/except eval/backtest/metrics HIỆN CÓ, giữ nguyên; các nhánh
            # error TRƯỚC KHI return thì ghi cache: self.eval_cache[ch] = ("error", reasons))
            ...
            if self.eval_cache is not None:
                self.eval_cache[ch] = ("ok", bt, metrics)
```

Phần còn lại của hàm (depth/fields/gate/pool_rho/fitness) giữ NGUYÊN — chạy lại tươi mỗi lần (pool lớn dần). `CanonicalHasher` import từ `src.lang.visitors` (đã dùng trong `_persist` — tái dùng).

`GPIdeaSource.__init__`: `self._eval_cache: dict[str, tuple] = {}`. `_run_one_batch`: trước khi dựng engine, `if len(self._eval_cache) > 5000: self._eval_cache.clear()`; truyền `eval_cache=self._eval_cache`.

- [ ] **Bước 4: Chạy pass.** Chú ý test xác nhận thêm: `MetricsCalculator` trong `_persist` vẫn hoạt động với bt từ cache (không đổi gì).
- [ ] **Bước 5: Toàn suite xanh.**
- [ ] **Bước 6: Commit** — `git commit -m "perf(gp): cache backtest thuần theo canonical_hash xuyên batch (A3)"`

---

### Task A4: Giảm max_empty_retries 8 → 2

**Files:**
- Modify: `src/app/closed_loop_adapters.py:849` (default `max_empty_retries`)
- Test: `tests/unit/test_closed_loop_adapters.py`

**Interfaces:** không đổi chữ ký — chỉ default. PHỤ THUỘC: làm SAU A2 (lọc trong-GP khiến lô rỗng = cạn thật).

- [ ] **Bước 1: Test fail**

```python
def test_max_empty_retries_mac_dinh_la_2():
    from src.app.closed_loop_adapters import GPIdeaSource
    src = GPIdeaSource(data=None, repo=None, config=None, registry=None)
    assert src.max_empty_retries == 2
```

- [ ] **Bước 2: Chạy fail** (đang là 8).
- [ ] **Bước 3: Đổi default `max_empty_retries: int = 8` → `= 2`** + cập nhật docstring/comment quanh `next_batch` (lý do: A2 đã lọc họ-đóng trong tiến hoá, lô rỗng không còn là "xui vì lọc sau-sinh"). Grep test cũ tham chiếu 8 (`grep -rn "max_empty_retries" tests/`) và cập nhật nếu có.
- [ ] **Bước 4: Chạy pass; toàn suite xanh.**
- [ ] **Bước 5: Commit** — `git commit -m "perf(engine): giảm max_empty_retries 8->2 sau khi có lọc trong-GP (A4)"`

---

### Task B1: Reseed epoch tự động khi cạn ý tưởng

**Files:**
- Modify: `src/pipeline/closed_loop.py` (constructor + nhánh batch rỗng ~dòng 351)
- Modify: `src/app/closed_loop_adapters.py` (`GPIdeaSource.reseed_epoch`, wrapper ủy quyền, wiring build_closed_loop)
- Test: `tests/unit/test_closed_loop.py`, `tests/unit/test_closed_loop_adapters.py`

**Interfaces:**
- Produces: `ClosedLoop.__init__(..., on_epoch_reseed=None)` — callable `() -> bool`; True = đã reseed (chạy tiếp), False/None = không reseed được (dừng như cũ).
- Produces: `GPIdeaSource.reseed_epoch() -> None`: `self._epoch += 1`; `self.base_seed += 10_000`; `self._batch = 0`; `self._gp_budget_exhausted = False`; xoay nhóm field ưu tiên (xem dưới). `self._epoch = 0` khởi tạo trong `__init__`. Wrapper ủy quyền `reseed_epoch` xuống fallback (mọi wrapper — pattern set_saturated_families).
- Semantics vòng: batch rỗng → nếu `on_epoch_reseed` trả True VÀ lần next_batch ngay trước đó KHÔNG phải ngay-sau-reseed → log `🔄 Epoch #k` + reset `gp_sims_used=0` + `continue`; batch rỗng NGAY SAU một reseed → dừng `no_more_ideas` (cạn tuyệt đối). GIỮ NGUYÊN: `closed_families`, `seen`, avoid-list.
- Xoay field: `GPIdeaSource.__init__` nhận thêm `field_groups: "tuple[tuple[str, ...], ...] | None" = None` (composition root dựng từ `repo.dataset_of_fields` nếu có — nhóm field theo dataset; None = không xoay). `_run_one_batch` khi `field_groups` và `self._epoch > 0`: truyền `fields_override=field_groups[self._epoch % len(field_groups)]` vào GPEngine; `GPEngine.__init__` thêm `fields_override: "tuple[str, ...] | None" = None`, trong `run()` dòng 296 đổi thành:

```python
        fields = (
            tuple(sorted(self.fields_override))
            if self.fields_override
            else tuple(sorted(self.data.field_names()))
        )
```

(fields_override phải ⊆ `data.field_names()` — composition root lọc trước khi truyền; epoch 0 luôn dùng toàn bộ field như cũ.)

- [ ] **Bước 1: Test fail** — `tests/unit/test_closed_loop.py`:

```python
def test_batch_rong_goi_reseed_roi_chay_tiep():
    """Batch rỗng lần 1 -> on_epoch_reseed()=True -> loop gọi next_batch tiếp; rỗng ngay
    sau reseed -> dừng no_more_ideas. gp_sims_used được reset (candidate gp lại được sim)."""
    goi = []
    # idea_source giả: lần 1 trả [], lần 2 trả [], (sau reseed thứ nhất vẫn rỗng -> dừng)
    report = _chay_loop_voi_batches(batches=[[], []], on_epoch_reseed=lambda: goi.append(1) or True)
    assert goi == [1]
    assert report.stop_reason == "no_more_ideas"
```

`tests/unit/test_closed_loop_adapters.py`:

```python
def test_reseed_epoch_doi_seed_va_mo_lai_gp():
    from src.app.closed_loop_adapters import GPIdeaSource
    src = GPIdeaSource(data=None, repo=None, config=None, registry=None, base_seed=42)
    src.set_gp_budget_exhausted(True)
    src.reseed_epoch()
    assert src.base_seed == 10_042
    assert src._gp_budget_exhausted is False
    assert src._batch == 0


def test_reseed_epoch_xoay_field_groups(monkeypatch):
    from src.app import closed_loop_adapters as m
    nhan = []
    monkeypatch.setattr(m, "GPEngine", lambda **kw: nhan.append(kw.get("fields_override")) or _EngineGiaRong())
    monkeypatch.setattr(m, "generate_many", lambda **kw: [])
    groups = (("close", "open"), ("volume", "vwap"))
    src = m.GPIdeaSource(data=_DataGia(), repo=_RepoGia(), config=None, registry=None,
                         field_groups=groups, max_empty_retries=1)
    src.next_batch()                 # epoch 0: không override
    assert nhan[-1] is None
    src.reseed_epoch()               # epoch 1: nhóm groups[1 % 2]
    src.next_batch()
    assert nhan[-1] == ("volume", "vwap")
```

- [ ] **Bước 2: Chạy fail.**
- [ ] **Bước 3: Implement** theo Interfaces trên. Wiring `build_closed_loop`:

```python
    # B1: nhóm field theo dataset cho xoay epoch (ưu tiên originality — dataset ít dùng lên
    # trước). repo không có dataset_of_fields (test giả) -> None, không xoay.
    field_groups = None
    if callable(_ds_fn) and hasattr(data, "field_names"):
        try:
            _mapping = _ds_fn(sorted(data.field_names()))  # {field: dataset}
            _by_ds: dict[str, list[str]] = {}
            for f, ds in _mapping.items():
                _by_ds.setdefault(ds or "khac", []).append(f)
            if len(_by_ds) >= 2:
                # dataset ÍT field trước (proxy "ít dùng"), pv lớn xuống cuối
                field_groups = tuple(
                    tuple(sorted(fs)) for _, fs in sorted(_by_ds.items(), key=lambda kv: len(kv[1]))
                )
        except Exception:
            field_groups = None
```

(LƯU Ý: kiểm tra chữ ký thật của `repo.dataset_of_fields` — grep định nghĩa trong `src/` trước khi viết; nếu nhận từng field một thì map từng cái. `_ds_fn` đã được lấy ở ~dòng 1188.)

Truyền `field_groups=field_groups` vào `GPIdeaSource(...)`; thêm callback:

```python
    def on_epoch_reseed() -> bool:
        idea_source.reseed_epoch()  # type: ignore[attr-defined]
        return True
```

`ClosedLoop.run` nhánh batch rỗng (~351) đổi thành:

```python
            if not batch:
                if (not vua_reseed) and self.on_epoch_reseed is not None and self.on_epoch_reseed():
                    vua_reseed = True
                    so_epoch += 1
                    gp_sims_used = 0
                    logger.info("🔄 Epoch #{}: reseed (batch rỗng) — seed mới + xoay dataset, giữ họ đóng.", so_epoch)
                    continue
                logger.info("Cạn ý tưởng (batch rỗng) — dừng vòng kín.")
                return _report("no_more_ideas")
            vua_reseed = False
```

(`vua_reseed = False`, `so_epoch = 0` khởi tạo đầu `run()`.)

- [ ] **Bước 4: Chạy pass; toàn suite xanh.**
- [ ] **Bước 5: Commit** — `git commit -m "feat(engine): reseed epoch tự động khi cạn ý tưởng — seed mới + xoay dataset, giữ họ đóng (B1)"`

---

### Task B2: Cân bằng dataset khi sinh cây ngẫu nhiên

**Files:**
- Modify: `src/gp/init.py` (`_random_leaf`, `random_tree`, `ramped_half_and_half`, `init_population`)
- Modify: `src/gp/engine.py` (truyền `field_groups` xuống `init_population`)
- Modify: `src/app/closed_loop_adapters.py` (`GPIdeaSource._run_one_batch` truyền `field_groups`)
- Test: `tests/unit/test_gp_init.py` (tạo mới nếu chưa có — grep `init_population` trong tests/ trước)

**Interfaces:**
- Produces: các hàm trên nhận thêm `field_groups: "tuple[tuple[str, ...], ...] | None" = None`. Khi có: `_random_leaf` chọn NHÓM uniform trước, rồi field uniform trong nhóm (two-stage) — field thuộc dataset nhỏ không bị dataset lớn (pv hàng trăm field) áp đảo xác suất. None: hành vi cũ nguyên vẹn (uniform phẳng).
- `GPEngine.__init__` thêm `field_groups` (default None) và truyền vào `init_population`; `fields` phẳng vẫn dùng cho mutation/crossover như cũ.

- [ ] **Bước 1: Test fail**

```python
def test_random_leaf_two_stage_can_bang_nhom():
    """1000 leaf với nhóm (1 field pv) vs (1 field alt): tỉ lệ mỗi nhóm ~50% (±10 điểm %),
    dù nhóm pv có 99 field và alt chỉ 1 field thì mỗi NHÓM vẫn 50%."""
    import numpy as np
    from src.gp.init import _random_leaf
    from src.lang.ast import Field

    rng = np.random.default_rng(7)
    pv = tuple(f"pv_{i}" for i in range(99))
    groups = (pv, ("alt_duy_nhat",))
    fields = pv + ("alt_duy_nhat",)
    dem_alt = sum(
        1 for _ in range(1000)
        if _random_leaf(rng, fields, field_groups=groups).name == "alt_duy_nhat"
    )
    assert 400 <= dem_alt <= 600  # uniform phẳng chỉ cho ~10/1000


def test_field_groups_none_giu_hanh_vi_cu():
    import numpy as np
    from src.gp.init import _random_leaf
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    f = ("a", "b", "c")
    cu = [_random_leaf(rng1, f).name for _ in range(50)]
    moi = [_random_leaf(rng2, f, field_groups=None).name for _ in range(50)]
    assert cu == moi
```

(Kiểm tra attribute thật của `Field` — `.name` hay khác — đọc `src/lang/ast.py` trước khi viết assert.)

- [ ] **Bước 2: Chạy fail.**
- [ ] **Bước 3: Implement** — `_random_leaf`:

```python
def _random_leaf(
    rng: np.random.Generator, fields: tuple[str, ...], *, kind: ArgKind = ArgKind.PANEL,
    field_groups: "tuple[tuple[str, ...], ...] | None" = None,
) -> Node:
    if kind is ArgKind.SCALAR:
        return Constant(_random_scalar(rng))
    if field_groups:
        # B2: two-stage — chọn nhóm dataset uniform rồi field trong nhóm; dataset ít field
        # không bị nhóm pv đông áp đảo xác suất xuất hiện trong quần thể khởi tạo.
        nhom = field_groups[rng.integers(0, len(field_groups))]
        return Field(nhom[rng.integers(0, len(nhom))])
    return Field(fields[rng.integers(0, len(fields))])
```

Luồn `field_groups` qua `random_tree` → `_bounded_random_tree` → `ramped_half_and_half` → `init_population` (keyword-only, default None ở mọi tầng). `GPEngine.run()` truyền `field_groups=self.field_groups` vào `init_population`; `GPIdeaSource._run_one_batch` truyền `field_groups=self.field_groups if self._epoch == 0 else None` — LƯU Ý: khi epoch > 0 đã có `fields_override` (một nhóm duy nhất) thì two-stage thừa, truyền None.

- [ ] **Bước 4: Chạy pass; toàn suite xanh** (đặc biệt test determinism GP hiện có).
- [ ] **Bước 5: Commit** — `git commit -m "feat(gp): cân bằng dataset two-stage khi sinh leaf ngẫu nhiên (B2)"`

---

### Task C1: Song song hoá backtest thuần trong _evaluate_population

**Files:**
- Create: `src/gp/parallel_eval.py`
- Modify: `src/gp/engine.py` (`_evaluate_population` nhánh n_jobs>1; `__init__` đã có `n_jobs`)
- Modify: `src/app/closed_loop_adapters.py` (`GPIdeaSource` nhận `n_jobs`, giữ executor sống xuyên batch)
- Test: `tests/unit/test_parallel_eval.py` (mới)

**Interfaces:**
- Produces: `src/gp/parallel_eval.py`:

```python
"""C1: worker đánh giá PHẦN THUẦN của cá thể GP (eval AST → danh mục → backtest → metrics)
trong process con — KHÔNG SQLite, KHÔNG pool_corr (main process lo). Windows spawn: mọi hàm
module-level; data/config/registry nạp MỘT LẦN/worker qua initializer (không pickle lại mỗi task)."""
from __future__ import annotations

_CTX: dict = {}  # {"data": MarketData, "config": PortfolioConfig, "registry": OperatorRegistry}


def khoi_tao_worker(data, config, registry) -> None:
    _CTX.update(data=data, config=config, registry=registry)


def eval_thuan(expr_string: str):
    """Trả ("ok", daily_pnl, metrics) | ("error", [lý_do]). Nhận CHUỖI expr (Node giữ
    picklable nhưng chuỗi chắc chắn an toàn + rẻ) — parse lại trong worker."""
    from src.backtest.backtester import Backtester
    from src.backtest.metrics_local import MetricsCalculator
    from src.backtest.portfolio import PortfolioBuilder
    from src.engine.evaluator import EvalContext, Evaluator
    from src.engine.subexpr_cache import SubexprCache
    from src.lang.parser import parse
    try:
        node = parse(expr_string)
        ctx = EvalContext(data=_CTX["data"], registry=_CTX["registry"], cache=SubexprCache())
        signal = Evaluator(ctx).evaluate(node)
        weights = PortfolioBuilder().build(signal, _CTX["config"], _CTX["data"])
        bt = Backtester().run(weights, _CTX["data"])
        metrics = MetricsCalculator().compute(bt, _CTX["data"])
    except Exception as exc:  # noqa: BLE001
        return ("error", [f"{type(exc).__name__}: {exc}"])
    return ("ok", bt, metrics)
```

  (Implementer: nếu `BacktestResult` không picklable thì trả `(bt.daily_pnl, metrics)` và dựng lại phía main — kiểm tra dataclass `BacktestResult` trong `src/backtest/backtester.py` trước.)
- `_evaluate_population` khi `self.n_jobs > 1` và có executor: (1) lọc A2 (meaningfulness/họ-đóng) + tra cache A3 TRƯỚC ở main; (2) submit các cá thể miss theo index; (3) nhận kết quả, ghi cache, rồi vòng TUẦN TỰ THEO INDEX GỐC: gate/pool_corr/fitness/persist — xử lý y hệt nhánh tuần tự. `n_jobs=1` (default): đường cũ nguyên vẹn, không import concurrent.futures.
- `GPIdeaSource.__init__(..., n_jobs: int = 1)`: `n_jobs>1` → tạo `ProcessPoolExecutor(max_workers=n_jobs, initializer=khoi_tao_worker, initargs=(data, config, registry))` MỘT LẦN, truyền vào mọi GPEngine (`executor=` param mới, default None), đóng khi GC (`atexit` hoặc method `close()`). CLI/menu nối `n_jobs` sau (ngoài phạm vi task này — default 1 nghĩa là C1 chưa bật cho user, C2 xanh rồi mới nối).

- [ ] **Bước 1: Test fail** — `tests/unit/test_parallel_eval.py`: test `khoi_tao_worker` + `eval_thuan` chạy IN-PROCESS (gọi thẳng, không cần pool — logic thuần):

```python
def test_eval_thuan_tra_ok_va_loi():
    from src.gp.parallel_eval import khoi_tao_worker, eval_thuan
    khoi_tao_worker(_data_gia(), _config_gia(), _registry_that())
    tag, *phan = eval_thuan("ts_mean(subtract(close, open), 5)")
    assert tag == "ok"
    tag2, ly_do = eval_thuan("op_khong_ton_tai(close)")
    assert tag2 == "error"
```

  và một test integration nhỏ với pool thật 2 worker (mark `@pytest.mark.slow` nếu suite có convention đó — grep `slow` trong pytest.ini/pyproject trước).

- [ ] **Bước 2: Chạy fail.**
- [ ] **Bước 3: Implement theo Interfaces.**
- [ ] **Bước 4: Chạy pass; toàn suite xanh.**
- [ ] **Bước 5: Commit** — `git commit -m "perf(gp): song song hoá backtest thuần qua ProcessPoolExecutor, persist tuần tự (C1)"`

---

### Task C2: Test tái lập — song song ≡ tuần tự

**Files:**
- Test: `tests/integration/test_gp_parallel_parity.py` (mới)
- Modify (nếu C2 lộ lệch): sửa C1 cho tới khi parity đạt — KHÔNG nới assert.

**Interfaces:** Consumes: GPEngine với `n_jobs`/`executor` từ C1, fixture data giả từ `tests/integration/test_gp_engine_run.py` (tái dùng — import hoặc copy fixture).

- [ ] **Bước 1: Viết test (đây là deliverable chính)**

```python
def test_song_song_va_tuan_tu_cho_cung_ket_qua():
    """Chốt chặn chất lượng Pha C: cùng seed + cùng data -> quần thể cuối GIỐNG HỆT
    (tập canonical_hash + sharpe từng cá thể) giữa n_jobs=1 và n_jobs=2."""
    r1 = _chay_engine(n_jobs=1, seed=123)
    r2 = _chay_engine(n_jobs=2, seed=123)

    def dau_van(res):
        return sorted(
            (i.expr.accept(CanonicalHasher()), round(i.fitness.sharpe_deflated, 12))
            for i in res.final_population if i.fitness is not None
        )

    assert dau_van(r1) == dau_van(r2)
```

  (`_chay_engine`: dựng GPEngine với data giả deterministic của fixture integration, pop nhỏ (10) × 2 thế hệ cho nhanh; n_jobs=2 dùng pool thật — Windows spawn nên toàn bộ trong `if __name__` không cần, pytest tự lo, nhưng data giả PHẢI picklable.)

- [ ] **Bước 2: Chạy** — `python -m pytest tests/integration/test_gp_parallel_parity.py -v`. PASS ngay = C1 đúng; FAIL = sửa C1 (thứ tự nhận kết quả, rng dùng chung, cache) tới khi xanh.
- [ ] **Bước 3: Toàn suite xanh.**
- [ ] **Bước 4: Commit** — `git commit -m "test(gp): parity song song ≡ tuần tự — chốt chặn chất lượng Pha C (C2)"`

---

## Nghiệm thu cuối

1. `python -m pytest -q` xanh toàn bộ.
2. Chạy menu 5 (USER làm, phiên thật): so `gen_batch_ms` trong log/funnel CSV trước-sau; kịch bản "họ đóng + gp_budget cạn" phải thấy batch < 30 giây (trước: 3–14 phút); log có dòng `🔄 Epoch #1` khi cạn ý tưởng thay vì dừng.
