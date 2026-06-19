# Thành phần A — Neo field thật + validation cứng — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Không lượt simulation nào còn gửi field bịa/không tồn tại lên API WQ — bịt mọi lỗ rò bằng một cổng validation tại biên `Simulator`, và đẩy phản hồi field chết vào prompt.

**Architecture:** Thêm một cổng tiền-kiểm bắt buộc ngay trước khi `Simulator.simulate()` gọi API: mọi biểu thức (LLM seed / GA / refiner / translator) phải qua `pre_sim_validator` (chính là `PreFilter.check` đã có `known_fields`). Field ngoài catalog → không gửi API, ghi blacklist, trả lỗi sớm. Bổ sung: nhồi blacklist vào prompt và mở rộng field thật theo dataset được nêu.

**Tech Stack:** Python 3.12, pytest, SQLAlchemy (SQLite `wq_alpha.db`), loguru, typer.

## Global Constraints

- Code, comment, commit message bằng tiếng Việt (giữ đúng dấu). TDD bắt buộc, mỗi task ≥1 commit.
- Không đổi engine GA/MCTS lõi. Chỉ thêm cổng lọc + ngữ cảnh prompt.
- Giữ scope USA/TOP3000/delay=1. Catalog field thật lấy từ DB qua `FieldRepository.load_cached`.
- Ngưỡng submit (tham chiếu, không sửa ở A): `sharpe ≥ 1.25`, `fitness > 1.0`, `turnover ∈ [0.01, 0.70]`.
- `PreFilter.check(expr) -> tuple[bool, str]`; reason field lạ có dạng `"Field/hằng không tồn tại: <id>"`.

---

### Task 1: Cổng tiền-kiểm tại biên Simulator (A1 — fix mấu chốt)

**Files:**
- Modify: `src/simulation/simulator.py` (class `Simulator.__init__`, `Simulator.simulate`)
- Create/Modify: `tests/test_simulator.py`

**Interfaces:**
- Consumes: `PreFilter.check(expr) -> (bool, str)` từ `src/simulation/pre_filter.py`.
- Produces:
  - `Simulator.__init__(..., pre_sim_validator: Callable[[str], tuple[bool, str]] | None = None)`
  - `extract_rejected_field(reason: str) -> str | None` trong `src/simulation/simulator.py` — trích `<id>` từ reason `"Field/hằng không tồn tại: <id>"`.
  - Hành vi: nếu `pre_sim_validator` trả `(False, reason)` thì `simulate()` KHÔNG gọi `client.post`, trả `SimulationResult(status="error", raw={"error": "pre-sim reject: <reason>"})`, và nếu trích được field thì gọi `on_invalid_field(field)`.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_simulator.py
from src.simulation.simulator import Simulator, extract_rejected_field


class _SpyClient:
    def __init__(self):
        self.post_calls = 0
    def post(self, *a, **k):
        self.post_calls += 1
        raise AssertionError("không được gọi API khi tiền-kiểm fail")


def test_extract_rejected_field():
    assert extract_rejected_field("Field/hằng không tồn tại: foo_bar") == "foo_bar"
    assert extract_rejected_field("Độ sâu > 6") is None


def test_pre_sim_validator_chan_truoc_khi_goi_api():
    client = _SpyClient()
    recorded = []
    sim = Simulator(
        client,
        pre_sim_validator=lambda e: (False, "Field/hằng không tồn tại: foo_bar"),
        on_invalid_field=recorded.append,
    )
    res = sim.simulate("rank(foo_bar)")
    assert client.post_calls == 0
    assert res.status == "error"
    assert "pre-sim reject" in res.raw["error"]
    assert recorded == ["foo_bar"]


def test_pre_sim_validator_ok_thi_goi_api_nhu_cu(monkeypatch):
    # validator pass -> vẫn đi tới nhánh post (sẽ lỗi vì client giả) nhưng post được gọi.
    class _Client:
        def __init__(self): self.post_calls = 0
        def post(self, *a, **k):
            self.post_calls += 1
            class R:  # phản hồi lỗi để dừng sớm, đủ để xác nhận post được gọi
                status_code = 500; text = "x"; headers = {}
            return R()
    client = _Client()
    sim = Simulator(client, pre_sim_validator=lambda e: (True, "ok"))
    sim.simulate("rank(close)")
    assert client.post_calls == 1
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_simulator.py -k "pre_sim or rejected_field" -v`
Expected: FAIL (`ImportError: cannot import name 'extract_rejected_field'` và `TypeError: unexpected keyword 'pre_sim_validator'`).

- [ ] **Step 3: Cài đặt tối thiểu**

Trong `src/simulation/simulator.py`, thêm regex + helper gần `_INVALID_FIELD_RE`:

```python
_REJECTED_FIELD_RE = re.compile(r"Field/hằng không tồn tại: (\S+)")


def extract_rejected_field(reason: str) -> str | None:
    """Trích field id từ reason tiền-kiểm 'Field/hằng không tồn tại: X'; None nếu khác."""
    if not reason:
        return None
    m = _REJECTED_FIELD_RE.search(reason)
    return m.group(1) if m else None
```

Sửa `Simulator.__init__` thêm tham số (đặt sau `on_invalid_field`):

```python
    def __init__(
        self,
        client: WQBrainClient,
        rate_limiter: RateLimiter | None = None,
        sleep_func=time.sleep,
        time_func=time.monotonic,
        on_invalid_field=None,
        pre_sim_validator=None,
    ):
        ...
        self.on_invalid_field = on_invalid_field
        # callback(expr)->(ok, reason): chặn biểu thức field-bịa TRƯỚC khi tốn 1 lượt API.
        self.pre_sim_validator = pre_sim_validator
        self._consecutive_auth_failures = 0
```

Đầu `simulate()`, ngay trước `body = self._build_body(...)`:

```python
    def simulate(self, expression: str, settings: dict | None = None) -> SimulationResult:
        if self.pre_sim_validator is not None:
            ok, reason = self.pre_sim_validator(expression)
            if not ok:
                logger.warning("Bỏ sim (tiền-kiểm): {} | expr={}", reason, expression)
                bad = extract_rejected_field(reason)
                if bad and self.on_invalid_field is not None:
                    self.on_invalid_field(bad)
                return SimulationResult(
                    expression=expression, status="error",
                    raw={"error": f"pre-sim reject: {reason}"},
                )
        body = self._build_body(expression, settings)
        ...
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `python -m pytest tests/test_simulator.py -k "pre_sim or rejected_field" -v`
Expected: PASS (3 test).

- [ ] **Step 5: Commit**

```bash
git add src/simulation/simulator.py tests/test_simulator.py
git commit -m "feat(sim): cổng tiền-kiểm field tại biên Simulator (chặn field bịa trước khi gọi API)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Wiring cổng tiền-kiểm vào runtime + blacklist động trong phiên (A1+A2)

**Files:**
- Modify: `main.py` (các site dựng `Simulator(...)`: dòng ~607 và ~997)
- Modify: `src/simulation/simulator.py` (cập nhật blacklist động in-memory)
- Modify: `tests/test_simulator.py`

**Interfaces:**
- Consumes: `pf` (`PreFilter` có `known_fields`) đã dựng sẵn tại 2 site trong `main.py`; `extract_invalid_field` (đã có).
- Produces: tại runtime, `Simulator` nhận `pre_sim_validator=pf.check`. Khi WQ từ chối 1 field giữa phiên (`on_invalid_field`), field đó được loại khỏi `pf.known_fields` ngay để không bị thử lại trong cùng phiên.

- [ ] **Step 1: Viết test thất bại (blacklist động trong phiên)**

```python
# tests/test_simulator.py  (thêm)
from src.simulation.pre_filter import PreFilter


def test_field_chet_giua_phien_bi_loai_ngay(monkeypatch):
    pf = PreFilter(known_fields={"close", "dead_fld"}, known_operators={"rank"})

    # giả lập: lần 1 WQ báo dead_fld 'chết' -> recorder loại khỏi pf.known_fields.
    def recorder(field_id):
        pf.known_fields.discard(field_id)

    sim = Simulator(_DummyPost500(), pre_sim_validator=pf.check, on_invalid_field=recorder)
    recorder("dead_fld")  # mô phỏng phát hiện field chết
    ok, reason = pf.check("rank(dead_fld)")
    assert ok is False  # lần sau cùng phiên: bị tiền-kiểm loại, không tốn sim
```

Thêm helper `_DummyPost500` (client trả 500) ở đầu file test nếu chưa có:

```python
class _DummyPost500:
    def post(self, *a, **k):
        class R: status_code = 500; text = "x"; headers = {}
        return R()
```

- [ ] **Step 2: Chạy test để xác nhận fail/pass logic**

Run: `python -m pytest tests/test_simulator.py -k "field_chet_giua_phien" -v`
Expected: PASS ngay (PreFilter đã hỗ trợ `discard`); nếu fail do `known_fields` là `None`, sửa test dùng set rõ ràng. Test này chốt hợp đồng "recorder loại field khỏi known_fields".

- [ ] **Step 3: Wiring runtime trong `main.py`**

Tại site research-loop (≈ dòng 607) và hybrid (≈ dòng 997), đổi:

```python
sim = Simulator(
    client,
    on_invalid_field=_make_invalid_field_recorder(session_factory, region, universe),
)
```

thành (truyền validator + recorder loại field khỏi pf ngay trong phiên):

```python
_record_invalid = _make_invalid_field_recorder(session_factory, region, universe)

def _on_invalid_field(field_id, _pf=pf, _rec=_record_invalid):
    _pf.known_fields and _pf.known_fields.discard(field_id)
    _rec(field_id)

sim = Simulator(
    client,
    on_invalid_field=_on_invalid_field,
    pre_sim_validator=pf.check,
)
```

(Ở site research-loop, biến prefilter là `pf`; site hybrid cũng là `pf` — xác nhận tên trước khi sửa.)

- [ ] **Step 4: Chạy toàn bộ test simulator**

Run: `python -m pytest tests/test_simulator.py -v`
Expected: PASS toàn bộ.

- [ ] **Step 5: Commit**

```bash
git add main.py src/simulation/simulator.py tests/test_simulator.py
git commit -m "feat(sim): wiring tiền-kiểm pf.check vào runtime + loại field chết khỏi known_fields trong phiên

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Nhồi blacklist field chết vào prompt sinh ý tưởng (A2)

**Files:**
- Modify: `src/llm/generator.py` (`build_ideas_system_prompt` và hàm dựng `_idea_field_context` — thêm danh sách field cấm)
- Modify: `tests/test_generator.py` (tạo nếu chưa có)

**Interfaces:**
- Consumes: tập field cấm dạng `set[str]` truyền vào generator (từ `InvalidFieldRepository.blacklist()`).
- Produces: `LLMGenerator` chấp nhận `blacklist: set[str] | None` và chèn dòng `"TUYỆT ĐỐI KHÔNG dùng field: ..."` vào system prompt sinh ý tưởng khi blacklist không rỗng.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_generator.py
def test_prompt_y_tuong_co_dong_cam_field(make_generator):
    gen = make_generator(blacklist={"opt6_1dorhv", "asset_growth_rate"})
    prompt = gen.build_ideas_system_prompt()
    assert "TUYỆT ĐỐI KHÔNG dùng field" in prompt
    assert "opt6_1dorhv" in prompt
```

(`make_generator` fixture: dựng `LLMGenerator` với deepseek giả + field_repo giả trả vài `DataField`. Nếu chưa có fixture, tạo trong cùng file.)

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_generator.py -k "dong_cam_field" -v`
Expected: FAIL (`TypeError: unexpected keyword 'blacklist'` hoặc thiếu chuỗi).

- [ ] **Step 3: Cài đặt tối thiểu**

Trong `src/llm/generator.py`, thêm tham số `blacklist` vào `__init__` (mặc định `None`, lưu `self.blacklist = set(blacklist or ())`). Trong `build_ideas_system_prompt`, trước dòng `return`, chèn:

```python
        blacklist_line = ""
        if self.blacklist:
            cam = ", ".join(sorted(self.blacklist)[:50])
            blacklist_line = (
                "TUYỆT ĐỐI KHÔNG dùng field sau (WQ đã từ chối/chết): "
                f"{cam}.\n"
            )
```

và nối `blacklist_line` vào chuỗi trả về (ngay sau dòng `field_context`).

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `python -m pytest tests/test_generator.py -k "dong_cam_field" -v`
Expected: PASS.

- [ ] **Step 5: Wiring `_make_llm_generator` truyền blacklist**

Trong `main.py`, tại `_make_llm_generator`, lấy `blacklist = InvalidFieldRepository(session_factory).blacklist()` và truyền vào `LLMGenerator(..., blacklist=blacklist)`. (Xác nhận chữ ký `_make_llm_generator` trước khi sửa.)

- [ ] **Step 6: Commit**

```bash
git add src/llm/generator.py main.py tests/test_generator.py
git commit -m "feat(llm): nhồi blacklist field chết vào prompt sinh ý tưởng

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Repair gợi ý field thật gần nhất khi field lỗi (A4)

**Files:**
- Modify: `src/llm/translator.py` (`_to_expression` — khi reason là field lạ, đính kèm gợi ý field thật)
- Modify: `tests/test_translator.py`

**Interfaces:**
- Consumes: `extract_rejected_field` (Task 1) để lấy field lỗi từ reason; `self.field_repo.load_cached(**scope)` để lấy field thật.
- Produces: `AlphaTranslator._suggest_fields(bad_field: str, limit: int = 5) -> list[str]` — trả tối đa `limit` field thật có prefix dataset/ký tự gần `bad_field` nhất (so khớp tiền tố + độ trùng token); prompt repair chèn `"Field có thật gần nhất: ..."`.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_translator.py  (thêm)
def test_suggest_fields_tra_field_that_gan_nhat(make_translator):
    tr = make_translator(field_ids=["opt6_1dorhv_real", "opt6_close", "news12_sent", "close"])
    out = tr._suggest_fields("opt6_1dorhv")
    assert "opt6_1dorhv_real" in out          # cùng tiền tố opt6_ + trùng token
    assert "close" not in out[:1]             # field không liên quan không đứng đầu
```

(`make_translator` fixture: dựng `AlphaTranslator` với field_repo giả trả `DataField(id=...)`.)

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_translator.py -k "suggest_fields" -v`
Expected: FAIL (`AttributeError: _suggest_fields`).

- [ ] **Step 3: Cài đặt tối thiểu**

Thêm vào `AlphaTranslator`:

```python
    def _suggest_fields(self, bad_field: str, limit: int = 5) -> list[str]:
        """Field thật gần 'bad_field' nhất: ưu tiên cùng tiền tố dataset, rồi trùng token."""
        cached = self.field_repo.load_cached(**self._scope) if self._scope else self.field_repo.load_cached()
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
        return [fid for _, fid in scored[:limit]]
```

Trong `_to_expression`, sau khi có `reason` không OK, nếu trích được field lạ thì bổ sung gợi ý vào prompt sửa:

```python
            logger.info("Translator expr lỗi (lần {}): {} — {}", attempt + 1, expr, reason)
            bad = extract_rejected_field(reason)
            hint = ""
            if bad:
                suggestions = self._suggest_fields(bad)
                if suggestions:
                    hint = f" Field có thật gần nhất: {', '.join(suggestions)}."
            user = f'Biểu thức "{expr}" bị lỗi: {reason}.{hint} Sửa lại, trả JSON.'
```

(Thêm `from src.simulation.simulator import extract_rejected_field` ở đầu `translator.py`.)

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `python -m pytest tests/test_translator.py -k "suggest_fields" -v`
Expected: PASS.

- [ ] **Step 5: Chạy toàn bộ test liên quan**

Run: `python -m pytest tests/test_translator.py tests/test_simulator.py tests/test_generator.py -v`
Expected: PASS toàn bộ.

- [ ] **Step 6: Commit**

```bash
git add src/llm/translator.py tests/test_translator.py
git commit -m "feat(llm): repair gợi ý field thật gần nhất khi field lỗi

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Theo sau (kế hoạch riêng, sau khi A xong + verify)

- **Thành phần B** (vòng phản hồi số thật): bỏ metric LLM bịa khỏi xếp hạng; cấp ngân sách bandit; refine theo `blocking_dimensions`. Cần đọc `src/llm/loop.py`, `src/optimization/hybrid.py`, khâu parse `directions`.
- **Thành phần C** (cổng độc đáo tiền-sim): gate AST `ReferenceZoo` trước sim; steer dataset ít dùng; `CorrelationChecker` xác nhận trước nộp.

## Self-Review

- **Spec coverage:** A1→Task 1+2; A2→Task 2 (blacklist động) + Task 3 (prompt); A3 (dataset-scoped injection) — phần lớn đã có sẵn trong `_relevant_fields` (score +100 field đích danh, +20 dataset); A4→Task 4. *Khoảng trống có ý thức:* A3 mở rộng "bơm nhiều field hơn của dataset được nêu" để dành cho Thành phần B/C hoặc khi đo thấy còn hallucination; chưa làm ở plan này để giữ phạm vi.
- **Placeholder scan:** không có TBD/TODO; mọi step có code/lệnh cụ thể.
- **Type consistency:** `pre_sim_validator: Callable[[str], tuple[bool,str]]` khớp `PreFilter.check`; `extract_rejected_field` dùng chung Task 1↔Task 4; `blacklist: set[str]` khớp `InvalidFieldRepository.blacklist()`.
