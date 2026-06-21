# Ground field thật vào ý tưởng — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diệt lỗi "Field/hằng không tồn tại" bằng cách ràng giả thuyết/ý tưởng vào field có thật rồi ghim tập field đó xuống lõi sinh biểu thức.

**Architecture:** Thêm khái niệm "palette field đã ground" (tập field ID có thật, liên quan) sinh ở nguồn (`HypothesisGenerator`) và chảy xuyên suốt xuống 2 hàm lõi chung `build_symbol_context`/`repair_to_expression` (thêm tham số `pinned`). Hybrid: LLM chọn field từ palette có thật, code validate + bổ sung. Tương thích ngược tuyệt đối: `pinned=None`/`palette=None` → hành vi y như cũ.

**Tech Stack:** Python 3.12, pytest, loguru, dataclasses. Không thêm thư viện mới.

## Global Constraints

- TDD bắt buộc: test viết trước, chạy đỏ, rồi mới implement; mỗi task 1 commit.
- Code/comment/commit bằng **tiếng Việt có dấu đầy đủ**.
- Tương thích ngược: mọi nhánh `pinned=None`/`palette=None` phải cho output y hệt hành vi hiện tại — không sửa test regression hiện có trừ khi task ghi rõ.
- KHÔNG đụng `src/simulation/pre_filter.py` (validation đã đúng).
- KHÔNG xử lý lỗi `Số node`/`Độ sâu` lần này (ngoài phạm vi).
- Lệnh test chạy từ gốc repo `D:\wq\WorldQuant-Brain-Alpha`.

## File Structure

- `src/llm/hypothesis.py` — thêm `Hypothesis.fields`, `ground_fields()`, `HypothesisGenerator.generate(palette=None)`.
- `src/llm/expr_synth.py` — `retrieve_field_palette()` (mới), `suggest_fields(pinned=)`, `build_symbol_context(pinned=)`, `repair_to_expression(pinned=)`.
- `src/llm/translator.py` — `field_palette()`, truyền `pinned` qua `_to_expression`/`translate`.
- `src/llm/loop.py` — gọi `translator.field_palette` + `hypothesis_gen.generate(direction, palette)` ở `run()` và `run_mcts()`.
- `src/llm/generator.py` — `_generate_one` ghim palette truy hồi (path generator), giữ `generate_ideas() -> list[str]`.
- `tests/` — test mới + cập nhật 3 fake trong `test_loop.py`.

**`main.py` KHÔNG đổi:** loop tự gọi `translator.field_palette` (có sau Task 7) và `hypothesis_gen.generate(palette)` (có sau Task 6); constructor không đổi.

---

### Task 1: `ground_fields` + `Hypothesis.fields`

**Files:**
- Modify: `src/llm/hypothesis.py`
- Test: `tests/test_hypothesis.py`

**Interfaces:**
- Produces: `ground_fields(llm_fields, palette_ids, min_k=2) -> tuple[str, ...]` — giữ field LLM nêu có trong `palette_ids`, bổ sung từ `palette_ids` nếu < `min_k`, khử trùng lặp giữ thứ tự. `Hypothesis.fields: tuple[str, ...] = ()`.

- [ ] **Step 1: Viết test đỏ** — thêm vào cuối `tests/test_hypothesis.py`:

```python
from src.llm.hypothesis import ground_fields


def test_ground_fields_bo_field_bia_giu_field_that():
    out = ground_fields(["opt6_real", "bia_field"], ["opt6_real", "pcr_oi_30"])
    assert out == ("opt6_real",) or out == ("opt6_real", "pcr_oi_30")
    assert "bia_field" not in out


def test_ground_fields_augment_khi_thieu_min_k():
    # LLM toàn field bịa -> augment từ palette cho đủ min_k=2.
    out = ground_fields(["bia1", "bia2"], ["a", "b", "c"], min_k=2)
    assert out == ("a", "b")


def test_ground_fields_giu_thu_tu_va_khu_trung_lap():
    out = ground_fields(["a", "a", "b"], ["a", "b", "c"], min_k=2)
    assert out == ("a", "b")


def test_ground_fields_rong_tra_tuple_rong():
    assert ground_fields(None, [], min_k=2) == ()
    assert ground_fields("chuoi_don", [], min_k=2) == ()


def test_hypothesis_co_field_mac_dinh_rong():
    from src.llm.hypothesis import Hypothesis
    assert Hypothesis("a", "b", "c", "d").fields == ()
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_hypothesis.py -k "ground_fields or field_mac_dinh" -v`
Expected: FAIL — `ImportError: cannot import name 'ground_fields'`.

- [ ] **Step 3: Implement** — trong `src/llm/hypothesis.py`, thêm `fields` vào dataclass và hàm `ground_fields`:

```python
@dataclass
class Hypothesis:
    observation: str = ""
    background: str = ""
    economic_rationale: str = ""
    implementation_spec: str = ""
    fields: tuple[str, ...] = ()
```

(Giữ nguyên `to_dict`/`from_dict` — chúng chỉ duyệt `_PARTS` nên vẫn 4 khoá, không lộ `fields`.)

Thêm hàm (sau định nghĩa `Hypothesis`):

```python
def ground_fields(llm_fields, palette_ids, min_k: int = 2) -> tuple[str, ...]:
    """Lọc field LLM nêu về tập có thật (palette_ids); thiếu < min_k thì bổ sung
    từ palette. Khử trùng lặp, giữ thứ tự ưu tiên (LLM-hợp-lệ trước, palette sau)."""
    palette = list(dict.fromkeys(p for p in (palette_ids or []) if isinstance(p, str)))
    allowed = set(palette)
    if isinstance(llm_fields, str):
        llm_fields = [llm_fields]
    elif not isinstance(llm_fields, (list, tuple)):
        llm_fields = []
    grounded: list[str] = []
    for f in llm_fields:
        if isinstance(f, str) and f in allowed and f not in grounded:
            grounded.append(f)
    if len(grounded) < min_k:
        for f in palette:
            if f not in grounded:
                grounded.append(f)
            if len(grounded) >= min_k:
                break
    return tuple(grounded)
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_hypothesis.py -v`
Expected: PASS toàn bộ (gồm `test_hypothesis_to_dict_roundtrip` cũ — vẫn 4 khoá).

- [ ] **Step 5: Commit**

```bash
git add src/llm/hypothesis.py tests/test_hypothesis.py
git commit -m "feat(llm): ground_fields + Hypothesis.fields cho grounding field thật"
```

---

### Task 2: `retrieve_field_palette`

**Files:**
- Modify: `src/llm/expr_synth.py`
- Test: `tests/test_expr_synth.py`

**Interfaces:**
- Consumes: `_load_cached`, `_relevant_fields`, `_tokens`, `MAX_FIELDS_IN_PROMPT` (đã có trong expr_synth).
- Produces: `retrieve_field_palette(field_repo, scope, text, k=20, min_k=8) -> list` — trả **danh sách đối tượng field** (có `.id`, `.description`, `.type`), xếp theo độ liên quan với `text`, đảm bảo không rỗng khi cache không rỗng (độn theme alt-data rồi thứ tự cache).

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_expr_synth.py` (dùng `_Field`/`_FieldRepo` đã có trong file):

```python
def test_retrieve_palette_field_lien_quan_dung_dau():
    repo = _FieldRepo([
        _Field("pcr_oi_30", description="put call ratio open interest"),
        _Field("close"), _Field("volume"),
    ])
    out = expr_synth.retrieve_field_palette(repo, None, "put call open interest", min_k=1)
    assert out[0].id == "pcr_oi_30"


def test_retrieve_palette_khong_khop_van_khong_rong():
    repo = _FieldRepo([_Field(f"f{i}") for i in range(10)])
    out = expr_synth.retrieve_field_palette(repo, None, "asset growth rate", min_k=8)
    assert len(out) >= 8
    assert all(getattr(f, "id", None) for f in out)


def test_retrieve_palette_cache_rong_tra_rong():
    assert expr_synth.retrieve_field_palette(_FieldRepo([]), None, "bất kỳ") == []
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_expr_synth.py -k retrieve_palette -v`
Expected: FAIL — `AttributeError: module 'src.llm.expr_synth' has no attribute 'retrieve_field_palette'`.

- [ ] **Step 3: Implement** — trong `src/llm/expr_synth.py`, sau `_relevant_fields`, thêm:

```python
# Token gợi ý dataset thay thế (option/news/social/analyst/graph) để độn palette
# khi xếp hạng từ-vựng không đủ min_k field liên quan.
ALT_THEME_TOKENS = {
    "option", "implied", "iv", "put", "call", "skew", "pcr", "news", "event",
    "sentiment", "novelty", "social", "buzz", "scl", "analyst", "revision",
    "target", "recommendation", "supply", "graph", "competitor", "customer",
}


def retrieve_field_palette(field_repo, scope, text, k: int = 20, min_k: int = 8) -> list:
    """Palette field THẬT liên quan `text`, trả đối tượng field. Đảm bảo không rỗng
    khi cache không rỗng: thiếu thì độn theo theme alt-data rồi thứ tự cache."""
    cached = _load_cached(field_repo, scope)
    by_id = {getattr(f, "id", None): f for f in cached if getattr(f, "id", None)}
    ranked_ids = _relevant_fields(cached, text)
    out = [by_id[fid] for fid in ranked_ids[:k] if fid in by_id]
    if len(out) >= min_k:
        return out
    chosen = {getattr(f, "id", None) for f in out}

    def _pad(candidates):
        for f in candidates:
            fid = getattr(f, "id", None)
            if fid and fid not in chosen:
                out.append(f)
                chosen.add(fid)
                if len(out) >= min_k:
                    return True
        return False

    themed = [
        f for f in cached
        if _tokens(f"{getattr(f, 'id', '')} {getattr(f, 'description', '') or ''}") & ALT_THEME_TOKENS
    ]
    if not _pad(themed):
        _pad(cached)
    return out
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_expr_synth.py -k retrieve_palette -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/expr_synth.py tests/test_expr_synth.py
git commit -m "feat(llm): retrieve_field_palette — palette field thật không rỗng"
```

---

### Task 3: `suggest_fields` fallback không rỗng

**Files:**
- Modify: `src/llm/expr_synth.py:164-183`
- Test: `tests/test_expr_synth.py`

**Interfaces:**
- Produces: `suggest_fields(field_repo, scope, bad_field, limit=5, pinned=None) -> list[str]` — như cũ, nhưng khi không khớp prefix/token thì fallback về `pinned` (nếu có) hoặc top-`limit` field theo thứ tự cache → KHÔNG bao giờ rỗng nếu còn nguồn.

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_expr_synth.py`:

```python
def test_suggest_fields_fallback_khi_khong_khop():
    repo = _FieldRepo([_Field("pcr_oi_30"), _Field("close"), _Field("volume")])
    out = expr_synth.suggest_fields(repo, None, "asset_growth_rate")
    assert out  # không rỗng dù 'asset_growth_rate' không khớp field nào
    assert all(isinstance(x, str) for x in out)


def test_suggest_fields_fallback_uu_tien_pinned():
    repo = _FieldRepo([_Field("close")])
    out = expr_synth.suggest_fields(repo, None, "zzz_khong_khop", pinned=["pcr_oi_30", "scl12_buzz"])
    assert out[0] == "pcr_oi_30"
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_expr_synth.py -k "suggest_fields_fallback" -v`
Expected: FAIL — `test_suggest_fields_fallback_khi_khong_khop` trả `[]`; `pinned` là kwarg lạ → `TypeError`.

- [ ] **Step 3: Implement** — sửa cuối hàm `suggest_fields`:

```python
def suggest_fields(field_repo, scope, bad_field: str, limit: int = 5, pinned=None) -> list[str]:
    """Field thật gần 'bad_field' nhất: ưu tiên cùng tiền tố dataset, rồi trùng token.
    Không khớp gì -> fallback pinned hoặc top field cache (không bao giờ rỗng)."""
    cached = _load_cached(field_repo, scope)
    bad_low = (bad_field or "").lower()
    bad_prefix = bad_low.split("_", 1)[0]
    bad_tokens = set(re.findall(r"[a-z0-9]+", bad_low))
    scored = []
    for f in cached:
        fid = getattr(f, "id", None)
        if not fid:
            continue
        fl = fid.lower()
        score = 0
        if bad_prefix and fl.startswith(bad_prefix):
            score += 50
        score += len(set(re.findall(r"[a-z0-9]+", fl)) & bad_tokens)
        if score:
            scored.append((score, fid))
    scored.sort(key=lambda t: -t[0])
    result = [fid for _, fid in scored[:limit]]
    if not result:
        if pinned:
            result = [p for p in pinned if isinstance(p, str)][:limit]
        else:
            result = [getattr(f, "id", None) for f in cached if getattr(f, "id", None)][:limit]
    return result
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_expr_synth.py -k suggest_fields -v`
Expected: PASS (gồm 2 test cũ `uu_tien_cung_tien_to_dataset` không đổi hành vi).

- [ ] **Step 5: Commit**

```bash
git add src/llm/expr_synth.py tests/test_expr_synth.py
git commit -m "fix(llm): suggest_fields fallback không rỗng + tham số pinned"
```

---

### Task 4: `build_symbol_context(pinned=...)`

**Files:**
- Modify: `src/llm/expr_synth.py:129-147`
- Test: `tests/test_expr_synth.py`

**Interfaces:**
- Produces: `build_symbol_context(field_repo, operator_repo, prefilter, scope, relevance_text="", pinned=None) -> str` — `pinned` không rỗng → `FIELDS khả dụng` = đúng các id trong `pinned` (có trong cache, cắt `MAX_FIELDS_IN_PROMPT`), thêm câu cấm bịa field; `pinned` rỗng → y hệt cũ.

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_expr_synth.py`:

```python
def test_build_symbol_context_pinned_chi_liet_field_ghim():
    repo = _FieldRepo([_Field("pcr_oi_30"), _Field("close"), _Field("volume")])
    ops = FakeSymbolRepo(["rank"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"pcr_oi_30", "close", "volume"})
    out = expr_synth.build_symbol_context(repo, ops, pf, None, "bất kỳ", pinned=["pcr_oi_30"])
    assert "pcr_oi_30" in out
    # field ngoài palette ghim không được liệt vào dòng FIELDS
    field_line = [ln for ln in out.splitlines() if ln.startswith("FIELDS khả dụng")][0]
    assert "volume" not in field_line
    assert "KHÔNG bịa" in out


def test_build_symbol_context_pinned_none_giu_hanh_vi_cu():
    repo = _FieldRepo([_Field("close", "MATRIX"), _Field("svec", "VECTOR")])
    ops = FakeSymbolRepo(["rank", "ts_zscore", "vec_avg", "vec_sum"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close", "svec"})
    out = expr_synth.build_symbol_context(repo, ops, pf, scope=None, relevance_text="svec")
    assert "KHÔNG bịa" not in out  # không có câu ghim khi pinned=None
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_expr_synth.py -k build_symbol_context -v`
Expected: FAIL — `pinned` là kwarg lạ → `TypeError`.

- [ ] **Step 3: Implement** — sửa `build_symbol_context`:

```python
def build_symbol_context(field_repo, operator_repo, prefilter, scope, relevance_text: str = "", pinned=None) -> str:
    operators = [o.name for o in operator_repo.load_cached() if getattr(o, "name", None)]
    cached_fields = _load_cached(field_repo, scope)
    field_by_id = {getattr(f, "id", None): f for f in cached_fields if getattr(f, "id", None)}
    if pinned:
        fields = [fid for fid in dict.fromkeys(pinned) if fid in field_by_id][:MAX_FIELDS_IN_PROMPT]
    else:
        fields = _relevant_fields(cached_fields, relevance_text)
    selected_fields = [field_by_id[fid] for fid in fields if fid in field_by_id]
    type_context = _field_type_context(selected_fields)
    op_line = ", ".join(operators[:80]) or "rank, ts_delta, ts_mean, group_neutralize, ts_corr"
    field_line = ", ".join(fields) or "close, open, high, low, volume, vwap, returns"
    examples = "\n".join(f"- {e}" for e in FEWSHOT_EXAMPLES)
    context = (
        f"OPERATORS hợp lệ: {op_line}\n"
        f"FIELDS khả dụng: {field_line}\n"
        "GROUPS cho neutralize: market, sector, industry, subindustry\n"
        f"Ví dụ alpha hợp lệ:\n{examples}"
    )
    if type_context:
        context += f"\n{type_context}"
    if pinned:
        context += "\nTUYỆT ĐỐI chỉ dùng field trong danh sách FIELDS trên; KHÔNG bịa tên field mới."
    return context
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_expr_synth.py -k build_symbol_context -v`
Expected: PASS (gồm 2 test cũ `chen_quy_tac_vector`, `khong_vector` — pinned=None không đổi).

- [ ] **Step 5: Commit**

```bash
git add src/llm/expr_synth.py tests/test_expr_synth.py
git commit -m "feat(llm): build_symbol_context nhận pinned — ghim palette field thật"
```

---

### Task 5: `repair_to_expression(pinned=...)` tái-tiêm palette

**Files:**
- Modify: `src/llm/expr_synth.py:186-208`
- Test: `tests/test_expr_synth.py`

**Interfaces:**
- Consumes: `suggest_fields(..., pinned=)` (Task 3), `MAX_FIELDS_IN_PROMPT`.
- Produces: `repair_to_expression(deepseek, prefilter, field_repo, scope, system, user, task, pinned=None) -> str | None` — khi lỗi field bịa, prompt repair gồm gợi ý `suggest_fields` (không rỗng) + nhắc lại trọn palette `pinned` ("CHỈ được dùng các field: …").

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_expr_synth.py`:

```python
def test_repair_pinned_tai_tiem_palette_vao_prompt():
    pf = PreFilter(known_operators={"rank"}, known_fields={"pcr_oi_30"})
    ds = FakeDeepSeek([
        json.dumps({"expression": "rank(asset_growth_rate)"}),  # field bịa -> fail
        json.dumps({"expression": "rank(pcr_oi_30)"}),          # sửa hợp lệ
    ])
    out = expr_synth.repair_to_expression(
        ds, pf, _FieldRepo([_Field("pcr_oi_30")]), None, "sys", "usr",
        task=None, pinned=["pcr_oi_30"],
    )
    assert out == "rank(pcr_oi_30)"
    assert "pcr_oi_30" in ds.calls[1][1]
    assert "CHỈ được dùng" in ds.calls[1][1]
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_expr_synth.py -k repair_pinned -v`
Expected: FAIL — `pinned` là kwarg lạ → `TypeError`.

- [ ] **Step 3: Implement** — sửa `repair_to_expression` (chữ ký + nhánh hint):

```python
def repair_to_expression(deepseek, prefilter, field_repo, scope, system, user, task, pinned=None) -> str | None:
    """Vòng LLM -> auto-wrap -> prefilter.check -> retry kèm hint field thay thế."""
    field_types = getattr(prefilter, "field_types", None)
    matrix_only = getattr(prefilter, "matrix_only_ops", None)
    for attempt in range(MAX_REPAIR_ATTEMPTS):
        data = extract_json(deepseek.complete(system, user, json_mode=True, task=task))
        expr = data.get("expression") if isinstance(data, dict) else None
        if not expr:
            user = 'Trả ĐÚNG JSON {"expression": "..."}.'
            continue
        expr = autowrap_vector_fields(expr, field_types, matrix_only)
        ok, reason = prefilter.check(expr)
        if ok:
            return expr
        logger.info("LLM expr lỗi (lần {}): {} — {}", attempt + 1, expr, reason)
        bad = extract_rejected_field(reason)
        hint = ""
        if bad:
            suggestions = suggest_fields(field_repo, scope, bad, pinned=pinned)
            if suggestions:
                hint = f" Field có thật gần nhất: {', '.join(suggestions)}."
            if pinned:
                allowed = ", ".join([p for p in pinned if isinstance(p, str)][:MAX_FIELDS_IN_PROMPT])
                hint += f" CHỈ được dùng các field: {allowed}."
        user = f'Biểu thức "{expr}" bị lỗi: {reason}.{hint} Sửa lại, trả JSON.'
    return None
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_expr_synth.py -v`
Expected: PASS toàn bộ file (gồm `test_repair_them_hint_field_khi_field_bia`, `test_repair_tra_none_khi_het_retry` cũ).

- [ ] **Step 5: Commit**

```bash
git add src/llm/expr_synth.py tests/test_expr_synth.py
git commit -m "feat(llm): repair_to_expression tái-tiêm palette pinned khi field bịa"
```

---

### Task 6: `HypothesisGenerator.generate(palette=...)`

**Files:**
- Modify: `src/llm/hypothesis.py:50-60`
- Test: `tests/test_hypothesis.py`

**Interfaces:**
- Consumes: `ground_fields` (Task 1).
- Produces: `HypothesisGenerator.generate(research_direction, palette=None) -> Hypothesis` — `palette` (list field obj có `.id`/`.description`) không rỗng → prompt liệt kê palette + yêu cầu khoá `fields`; `Hypothesis.fields = ground_fields(data["fields"], [f.id ...])`. `palette=None` → hành vi cũ, `fields=()`.

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_hypothesis.py`:

```python
class _PField:
    def __init__(self, id, description=""):
        self.id = id
        self.description = description


def test_generate_ground_fields_tu_palette():
    payload = {
        "observation": "o", "background": "b", "economic_rationale": "r",
        "implementation_spec": "dùng pcr_oi_30", "fields": ["pcr_oi_30", "bia_field"],
    }
    ds = FakeDeepSeek([json.dumps(payload)])
    h = HypothesisGenerator(ds).generate("flow quyền chọn", palette=[_PField("pcr_oi_30"), _PField("scl12_buzz")])
    assert h.fields == ("pcr_oi_30",)  # bịa bị loại, đủ min? -> xem augment dưới


def test_generate_palette_liet_ke_vao_prompt():
    ds = FakeDeepSeek([json.dumps({"observation": "o"})])
    HypothesisGenerator(ds).generate("x", palette=[_PField("pcr_oi_30", "put call ratio")])
    system, _ = ds.calls[0]
    assert "pcr_oi_30" in system
    assert "fields" in system


def test_generate_thieu_khoa_fields_augment_tu_palette():
    ds = FakeDeepSeek([json.dumps({"observation": "o"})])  # không có "fields"
    h = HypothesisGenerator(ds).generate("x", palette=[_PField("a"), _PField("b"), _PField("c")])
    assert len(h.fields) >= 2  # augment tới min_k=2


def test_generate_khong_palette_giu_hanh_vi_cu():
    ds = FakeDeepSeek([json.dumps({"observation": "o"})])
    h = HypothesisGenerator(ds).generate("x")
    assert h.fields == ()
    system, _ = ds.calls[0]
    assert "fields" not in system
```

(Lưu ý: `test_generate_ground_fields_tu_palette` — `ground_fields(["pcr_oi_30","bia_field"], ["pcr_oi_30","scl12_buzz"], min_k=2)` cho `("pcr_oi_30","scl12_buzz")` vì augment đủ min_k. Sửa assert thành: `assert h.fields[0] == "pcr_oi_30" and "bia_field" not in h.fields`.)

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_hypothesis.py -k "generate_ground or palette or augment" -v`
Expected: FAIL — `generate()` chưa nhận `palette`.

- [ ] **Step 3: Implement** — sửa `HypothesisGenerator.generate`:

```python
    def generate(self, research_direction: str, palette=None) -> Hypothesis:
        user = (
            f'Hướng nghiên cứu: "{research_direction}". '
            "Đề xuất một giả thuyết alpha mới, cụ thể, có thể kiểm chứng. Trả JSON 4 phần."
        )
        system = SYSTEM_PROMPT
        palette_ids: list[str] = []
        if palette:
            palette_ids = [getattr(f, "id", None) for f in palette if getattr(f, "id", None)]
            listing = "\n".join(
                f"- {getattr(f, 'id', '')}: {(getattr(f, 'description', '') or '')[:60]}"
                for f in palette if getattr(f, "id", None)
            )
            system = (
                SYSTEM_PROMPT
                + "\nFIELD CÓ THẬT (chỉ nêu ID lấy ĐÚNG từ danh sách này):\n" + listing
                + '\nTrả thêm khoá "fields" = danh sách ID field bạn dùng; '
                "implementation_spec phải nêu chính các field ID đó."
            )
        content = self.deepseek.complete(system, user, json_mode=True, task="hypothesis")
        data = extract_json(content)
        if not isinstance(data, dict):
            logger.warning("Hypothesis: không parse được JSON, trả rỗng.")
            return Hypothesis(fields=ground_fields(None, palette_ids)) if palette_ids else Hypothesis()
        h = Hypothesis.from_dict(data)
        if palette_ids:
            h.fields = ground_fields(data.get("fields"), palette_ids)
        return h
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_hypothesis.py -v`
Expected: PASS toàn bộ (gồm 5 test cũ — gọi `generate(direction)` không palette).

- [ ] **Step 5: Commit**

```bash
git add src/llm/hypothesis.py tests/test_hypothesis.py
git commit -m "feat(llm): HypothesisGenerator.generate nhận palette, ground field vào hypothesis"
```

---

### Task 7: Translator — `field_palette` + truyền `pinned`

**Files:**
- Modify: `src/llm/translator.py:69-99`
- Test: `tests/test_translator.py`

**Interfaces:**
- Consumes: `expr_synth.retrieve_field_palette` (Task 2), `build_symbol_context(pinned=)` (Task 4), `repair_to_expression(pinned=)` (Task 5), `Hypothesis.fields` (Task 1).
- Produces: `AlphaTranslator.field_palette(text) -> list` (uỷ thác `retrieve_field_palette` với `self.field_repo`/`self._scope`). `translate` truyền `pinned=hypothesis.fields or None` xuống `_to_expression`.

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_translator.py`:

```python
def test_field_palette_uy_thac_retrieve():
    fields = FakeSymbolRepo(["close", "volume", "pcr_oi_30"])
    tr = AlphaTranslator(FakeDeepSeek([]), fields, FakeSymbolRepo(["rank"]),
                         PreFilter(known_operators={"rank"}, known_fields={"close"}))
    out = tr.field_palette("put call open interest")
    assert any(getattr(f, "id", None) == "pcr_oi_30" for f in out)


def test_translate_pinned_ep_chi_field_ghim_vao_prompt():
    fields = FakeSymbolRepo(["close", "volume", "pcr_oi_30"])
    pf = PreFilter(known_operators={"rank"}, known_fields={"close", "volume", "pcr_oi_30"})
    ds = FakeDeepSeek([json.dumps({"description": "d"}), json.dumps({"expression": "rank(pcr_oi_30)"})])
    tr = AlphaTranslator(ds, fields, FakeSymbolRepo(["rank"]), pf)
    h = Hypothesis("o", "b", "r", "s")
    h.fields = ("pcr_oi_30",)
    tr.translate(h)
    expr_system = ds.calls[1][0]
    assert "pcr_oi_30" in expr_system
    assert "KHÔNG bịa" in expr_system  # câu ghim từ build_symbol_context(pinned)
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_translator.py -k "field_palette or pinned" -v`
Expected: FAIL — `AlphaTranslator` chưa có `field_palette`; prompt chưa có câu ghim.

- [ ] **Step 3: Implement** — sửa `src/llm/translator.py`:

Thêm method (sau `set_scope`):

```python
    def field_palette(self, text: str) -> list:
        """Palette field THẬT liên quan `text`, theo đúng scope đang đặt (T6.4)."""
        return expr_synth.retrieve_field_palette(self.field_repo, self._scope, text)
```

Sửa `_to_expression` (thêm tham số `pinned` và truyền xuống):

```python
    def _to_expression(self, description: str, relevance_text: str = "", pinned=None) -> str | None:
        system = (
            "Bạn là chuyên gia viết biểu thức FASTEXPR trên WorldQuant BRAIN.\n"
            f"{expr_synth.build_symbol_context(self.field_repo, self.operator_repo, self.prefilter, self._scope, relevance_text or description, pinned=pinned)}\n"
            f"{self._avoid_context()}"
            f"{expr_synth.build_syntax_constraints(self.prefilter)}"
            "Dịch MÔ TẢ thành MỘT biểu thức FASTEXPR dùng đúng operators/fields được liệt kê. "
            'Trả JSON {"expression": "..."}.'
        )
        user = f"MÔ TẢ: {description}\nViết biểu thức FASTEXPR."
        return expr_synth.repair_to_expression(
            self.deepseek, self.prefilter, self.field_repo, self._scope, system, user,
            task="translate", pinned=pinned,
        )
```

Sửa `translate` (dòng cuối truyền pinned):

```python
        expression = self._to_expression(description, relevance_text, pinned=hypothesis.fields or None)
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_translator.py -v`
Expected: PASS toàn bộ (test cũ dùng `Hypothesis(...)` với `fields=()` → `pinned=None` → hành vi cũ).

- [ ] **Step 5: Commit**

```bash
git add src/llm/translator.py tests/test_translator.py
git commit -m "feat(llm): translator.field_palette + ghim pinned theo hypothesis.fields"
```

---

### Task 8: Loop — gọi palette + grounding; cập nhật fake

**Files:**
- Modify: `src/llm/loop.py:197-199` (trong `run`) và `:274-276` (trong `run_mcts`)
- Test: `tests/test_loop.py:29-44` (cập nhật 3 fake)

**Interfaces:**
- Consumes: `AlphaTranslator.field_palette` (Task 7), `HypothesisGenerator.generate(palette)` (Task 6).
- Produces: không API mới; loop nội bộ ground giả thuyết qua palette.

- [ ] **Step 1: Cập nhật fake + viết test đỏ** — sửa `tests/test_loop.py`:

`_FakeHyp` nhận `palette`:

```python
class _FakeHyp:
    def generate(self, direction, palette=None):
        return Hypothesis("o", "b", "r", "s")
```

`_FakeTranslator` + `_FakeTranslatorNone` thêm `field_palette`:

```python
class _FakeTranslator:
    def __init__(self, expr):
        self.expr = expr

    def field_palette(self, text):
        return []

    def translate(self, hyp):
        return AlphaCandidate(hyp, "mô tả gốc", self.expr)


class _FakeTranslatorNone:
    def field_palette(self, text):
        return []

    def translate(self, hyp):
        return None
```

Thêm test mới chứng minh loop truyền palette vào hypothesis_gen:

```python
def test_loop_truyen_palette_tu_translator_vao_hypothesis():
    captured = {}

    class _SpyHyp:
        def generate(self, direction, palette=None):
            captured["palette"] = palette
            return Hypothesis("o", "b", "r", "s")

    class _SpyTranslator:
        def field_palette(self, text):
            return ["pcr_oi_30"]

        def translate(self, hyp):
            return AlphaCandidate(hyp, "mô tả", "rank(close)")

    sim = FakeSimulator(default=_result("rank(close)", 1.5))
    loop = RefinementLoop(
        hypothesis_gen=_SpyHyp(), translator=_SpyTranslator(), refiner=_FakeRefiner([]),
        simulator=sim, prefilter=_prefilter(), repo=_repo(),
        region="USA", universe="TOP3000", max_simulations=1,
    )
    loop.run("flow quyền chọn")
    assert captured["palette"] == ["pcr_oi_30"]
```

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_loop.py -k "truyen_palette" -v`
Expected: FAIL — loop chưa gọi `field_palette`/chưa truyền palette (`captured["palette"]` là `None`/KeyError).

- [ ] **Step 3: Implement** — trong `src/llm/loop.py`, sửa CẢ `run` và `run_mcts`. Thay đoạn:

```python
        emit("hypothesis", 0.0, research_direction)
        hypothesis = self.hypothesis_gen.generate(research_direction)
```

bằng:

```python
        emit("hypothesis", 0.0, research_direction)
        palette = self.translator.field_palette(research_direction)
        hypothesis = self.hypothesis_gen.generate(research_direction, palette)
```

(Áp dụng y hệt cho cả hai vị trí — trong `run()` quanh dòng 197-198 và `run_mcts()` quanh dòng 274-275.)

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_loop.py -v`
Expected: PASS toàn bộ (3 fake đã cập nhật + test mới).

- [ ] **Step 5: Commit**

```bash
git add src/llm/loop.py tests/test_loop.py
git commit -m "feat(llm): loop ground giả thuyết qua translator.field_palette (run + mcts)"
```

---

### Task 9: Generator path — ghim palette truy hồi trong `_generate_one`

**Files:**
- Modify: `src/llm/generator.py:223-242`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `expr_synth.retrieve_field_palette` (Task 2), `build_symbol_context(pinned=)` (Task 4), `repair_to_expression(pinned=)` (Task 5).
- Produces: `generate_ideas() -> list[str]` GIỮ NGUYÊN (không phá `hybrid.py`/`main.py`). `build_system_prompt(relevance_text="", pinned=None)`; `_generate_one(idea)` tự truy hồi palette từ `idea` và ghim xuống synthesis.

- [ ] **Step 1: Viết test đỏ** — thêm vào `tests/test_llm.py`:

```python
def test_generate_ghim_palette_cam_bia_field():
    # FieldRepo có pcr_oi_30; idea nói về 'put call' -> palette ghim -> prompt cấm bịa.
    pf = PreFilter(known_operators={"rank"}, known_fields={"pcr_oi_30", "close"})
    field_repo = FakeRepo([_Field("pcr_oi_30", "put call ratio"), _Field("close")])
    op_repo = FakeRepo([_Op("rank")])
    gen = LLMAlphaGenerator(deepseek := FakeDeepSeek([json.dumps({"expression": "rank(pcr_oi_30)"})]),
                            field_repo, op_repo, pf)
    out = gen.generate("put call open interest flow", n=1)
    assert out == ["rank(pcr_oi_30)"]
    system_prompt = deepseek.calls[0][0]
    assert "KHÔNG bịa" in system_prompt
```

(Lưu ý: `_Field` trong `tests/test_llm.py` hiện không có `type`; `build_symbol_context` dùng `getattr(field, "type", "")` nên an toàn. Nếu cần, dùng `_Field(id, description, dataset_id)` đúng chữ ký hiện có ở dòng 14-19.)

- [ ] **Step 2: Chạy test, xác nhận đỏ**

Run: `python -m pytest tests/test_llm.py -k ghim_palette -v`
Expected: FAIL — system prompt chưa có câu "KHÔNG bịa" (chưa ghim pinned).

- [ ] **Step 3: Implement** — sửa `src/llm/generator.py`:

`build_system_prompt` nhận `pinned`:

```python
    def build_system_prompt(self, relevance_text: str = "", pinned=None) -> str:
        context = expr_synth.build_symbol_context(
            self.field_repo, self.operator_repo, self.prefilter, None, relevance_text, pinned=pinned
        )
        constraints = expr_synth.build_syntax_constraints(self.prefilter)
        return (
            "Bạn là chuyên gia thiết kế Alpha trên WorldQuant BRAIN, viết biểu thức FASTEXPR.\n"
            "Cú pháp: hàm(đối_số, ...), toán tử + - * /, rank chuẩn hóa cross-sectional, "
            "tiền tố ts_ là chuỗi thời gian với tham số cửa sổ là số nguyên.\n"
            f"{context}\n{constraints}{self._feedback_context()}"
            'Luôn trả về JSON đúng định dạng: {"expression": "...", "rationale": "..."}. '
            "Chỉ dùng operators và fields được liệt kê."
        )
```

`_generate_one` truy hồi palette + ghim:

```python
    def _generate_one(self, idea: str) -> str | None:
        palette = expr_synth.retrieve_field_palette(self.field_repo, None, idea)
        pinned = [getattr(f, "id", None) for f in palette if getattr(f, "id", None)] or None
        system = self.build_system_prompt(idea, pinned=pinned)
        user = f'Ý tưởng alpha: "{idea}". Sinh MỘT biểu thức FASTEXPR. Trả JSON.'
        return expr_synth.repair_to_expression(
            self.deepseek, self.prefilter, self.field_repo, None, system, user, task=None, pinned=pinned
        )
```

- [ ] **Step 4: Chạy test, xác nhận xanh**

Run: `python -m pytest tests/test_llm.py -v`
Expected: PASS toàn bộ (gồm `test_build_system_prompt_co_operators_va_json`, `test_generate_validation_loop_tu_sua`, `test_generate_ideas_parse_json` — `generate_ideas` không đổi).

- [ ] **Step 5: Commit**

```bash
git add src/llm/generator.py tests/test_llm.py
git commit -m "feat(llm): generator ghim palette truy hồi trong _generate_one (cấm bịa field)"
```

---

### Task 10: Regression toàn bộ + xác minh

**Files:** không sửa code (chỉ chạy).

- [ ] **Step 1: Chạy toàn bộ test suite**

Run: `python -m pytest -q`
Expected: PASS toàn bộ. Đặc biệt kiểm: `test_llm.py`, `test_expr_synth.py`, `test_hypothesis.py`, `test_translator.py`, `test_loop.py`, `test_hybrid.py`, `test_novel_ideas.py`, `test_auto_command.py`, `test_auto_pipeline.py`.

- [ ] **Step 2: Nếu có lỗi** — sửa theo nguyên tắc tương thích ngược (nhánh `pinned=None`/`palette=None` không được đổi hành vi). Commit fix riêng nếu cần.

- [ ] **Step 3: Commit (nếu Step 2 có sửa)**

```bash
git add -A
git commit -m "test: xanh toàn bộ suite sau grounding field thật"
```

---

## Self-Review (đã thực hiện)

**Spec coverage:**
- C1 `retrieve_field_palette` → Task 2. C2 `Hypothesis.fields`/`ground_fields`/`generate(palette)` → Task 1 + 6. C3 `build_symbol_context(pinned)`/`repair_to_expression(pinned)` → Task 4 + 5. C4 `suggest_fields` fallback → Task 3. C5 wiring translator/loop/generator → Task 7/8/9.
- **Lệch spec có chủ ý (đã xác nhận với user):** path generator KHÔNG đổi `generate_ideas` sang trả `[(idea, fields)]` (sẽ phá `hybrid.py` + test); thay vào đó ghim palette truy hồi trong `_generate_one` (Task 9). `Hypothesis.to_dict` giữ 4 khoá (không lộ `fields`) để khỏi phá `test_hypothesis_to_dict_roundtrip`. Validate field LLM nêu so với **palette đã trình** (không phải toàn cache) — vì prompt yêu cầu "chỉ chọn từ list này", và giữ `HypothesisGenerator` không phụ thuộc `field_repo`.

**Type consistency:** `pinned` xuyên suốt là `list[str] | None`; `palette` là `list[field_obj]`; `Hypothesis.fields` là `tuple[str,...]`; `field_palette()` trả `list[field_obj]` → loop truyền thẳng vào `generate(palette)` (đúng kiểu). `retrieve_field_palette` trả objects (cho prompt cần `.id`/`.description`); `pinned` rút từ `[f.id ...]`.

**Placeholder scan:** không có TBD/TODO; mọi step có code/lệnh/expected cụ thể.
