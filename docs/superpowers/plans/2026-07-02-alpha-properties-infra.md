# Alpha Properties Infra (Sub-project C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm khả năng gọi `PATCH /alphas/{id}` (set name/color/tags/description) lên WQ
Brain, có test, có audit DB — làm nền cho sub-project A (Power Pool) sau này. KHÔNG nối vào
luồng `submit()` trong đợt này.

**Architecture:** 3 lớp mỏng: (1) `WQBrainClient.patch()` — generic HTTP method tái dùng
`_request()` sẵn có; (2) `SubmissionManager.set_properties()` — dựng payload đúng shape WQ,
gọi patch, ghi/đọc audit DB, có idempotency; (3) 3 cột mới nullable trên `SubmissionModel`.

**Tech Stack:** Python, SQLAlchemy, httpx, pytest (đã có sẵn trong repo).

## Global Constraints

- TDD bắt buộc: viết test FAIL trước, xác nhận fail đúng lý do, rồi mới code tối thiểu.
- Code, comment, docstring, commit message: tiếng Việt có dấu đầy đủ.
- Mỗi task = 1 commit riêng.
- Chạy test bằng `venv/Scripts/python -m pytest` (không dùng `python` hệ thống).
- Endpoint thật: `PATCH /alphas/{alpha_id}` body `{"name", "color", "tags", "selectionDesc",
  "comboDesc", "regular": {"description": ...}}` — field nào không set thì KHÔNG đưa vào
  payload (không gửi `null`).
- KHÔNG nối `set_properties()` vào `submit()` hay CLI nào trong plan này (ngoài phạm vi).

---

### Task 1: `WQBrainClient.patch()`

**Files:**
- Modify: `src/data/client.py:326-330` (ngay sau `get()`/trước `post()`/`close()`)
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: `WQBrainClient._request(method, path, **kwargs) -> httpx.Response` (đã có sẵn,
  dòng 316).
- Produces: `WQBrainClient.patch(path: str, **kwargs) -> httpx.Response` — dùng bởi Task 3.

- [ ] **Step 1: Viết test FAIL**

Thêm vào cuối `tests/test_client.py`:

```python
def test_patch_goi_dung_method_va_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/authentication":
            return httpx.Response(200, json={"user": {"id": "u1"}})
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(200, json={"id": "WQ1"})

    client = _client_with(handler)
    resp = client.patch("/alphas/WQ1", json={"tags": ["a"]})
    assert resp.status_code == 200
    assert seen["method"] == "PATCH"
    assert seen["path"] == "/alphas/WQ1"
    assert b'"tags"' in seen["body"]
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/test_client.py::test_patch_goi_dung_method_va_path -v`
Expected: FAIL với `AttributeError: 'WQBrainClient' object has no attribute 'patch'`

- [ ] **Step 3: Cài tối thiểu**

Trong `src/data/client.py`, ngay trước `def close(self)`:

```python
    def patch(self, path: str, **kwargs) -> httpx.Response:
        return self._request("PATCH", path, **kwargs)
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/test_client.py::test_patch_goi_dung_method_va_path -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/client.py tests/test_client.py
git commit -m "feat(client): them WQBrainClient.patch() cho PATCH /alphas/{id}"
```

---

### Task 2: `FakeClient.patch()` (hạ tầng test dùng chung cho Task 3)

**Files:**
- Modify: `tests/fakes.py:21-44` (class `FakeClient`)
- Test: không có test riêng — được dùng làm fixture cho Task 3, xác nhận gián tiếp qua test
  của Task 3.

**Interfaces:**
- Consumes: không có.
- Produces: `FakeClient.queue_patch(response)`, `FakeClient.patch(path, **kwargs) ->
  FakeResponse` (giống `queue_post`/`post` đã có) — dùng bởi Task 3.

- [ ] **Step 1: Sửa `FakeClient` (không cần test FAIL riêng — đây là test helper thuần, sẽ
  được Task 3 xác nhận hoạt động qua test thật)**

Trong `tests/fakes.py`, sửa `class FakeClient`:

```python
class FakeClient:
    """Trả response theo hàng đợi cho từng (method, path-prefix)."""

    def __init__(self):
        self.calls = []
        self._get_queue = []
        self._post_queue = []
        self._patch_queue = []

    def queue_get(self, response):
        self._get_queue.append(response)

    def queue_post(self, response):
        self._post_queue.append(response)

    def queue_patch(self, response):
        self._patch_queue.append(response)

    def authenticate(self):
        self._authenticated = True

    def get(self, path, **kwargs):
        self.calls.append(("GET", path, kwargs))
        return self._get_queue.pop(0)

    def post(self, path, **kwargs):
        self.calls.append(("POST", path, kwargs))
        return self._post_queue.pop(0)

    def patch(self, path, **kwargs):
        self.calls.append(("PATCH", path, kwargs))
        return self._patch_queue.pop(0)
```

- [ ] **Step 2: Chạy toàn bộ test suite hiện có, xác nhận không có gì vỡ**

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS (không giảm số test PASS so với trước khi sửa `fakes.py`)

- [ ] **Step 3: Commit**

```bash
git add tests/fakes.py
git commit -m "test(fakes): them FakeClient.patch()/queue_patch() cho set_properties"
```

---

### Task 3: cột mới trên `SubmissionModel`

**Files:**
- Modify: `src/storage/models.py:127-135` (class `SubmissionModel`)
- Test: `tests/test_submission.py`

**Interfaces:**
- Consumes: không có.
- Produces: `SubmissionModel.tags: str | None` (JSON-encoded list), `SubmissionModel.
  regular_desc: str | None`, `SubmissionModel.properties_set_at: datetime | None` — dùng bởi
  Task 4.

- [ ] **Step 1: Viết test FAIL**

Thêm vào `tests/test_submission.py`:

```python
def test_submission_model_co_cot_properties():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    try:
        session.add(
            SubmissionModel(
                id="sub1", alpha_id="WQ1", status="properties_set",
                tags='["PowerPoolSelected"]', regular_desc="Idea: ...",
            )
        )
        session.commit()
        row = session.query(SubmissionModel).filter_by(id="sub1").one()
        assert row.tags == '["PowerPoolSelected"]'
        assert row.regular_desc == "Idea: ..."
        assert row.properties_set_at is None
    finally:
        session.close()
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/test_submission.py::test_submission_model_co_cot_properties -v`
Expected: FAIL với `TypeError: 'tags' is an invalid keyword argument for SubmissionModel`

- [ ] **Step 3: Cài tối thiểu**

Trong `src/storage/models.py`, sửa `class SubmissionModel`:

```python
class SubmissionModel(Base):
    __tablename__ = "submissions"

    id = Column(String, primary_key=True)
    alpha_id = Column(String, ForeignKey("alphas.id"))
    status = Column(String)  # submitted/rejected/error/properties_set
    self_correlation = Column(Float)
    detail = Column(Text)
    submitted_at = Column(DateTime, default=_utcnow)
    # Sub-project C: audit lần set properties/tags (PATCH /alphas/{id}) — nullable vì
    # không phải mọi submission đều set properties.
    tags = Column(Text)  # JSON-encoded list[str]
    regular_desc = Column(Text)
    properties_set_at = Column(DateTime)
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/test_submission.py::test_submission_model_co_cot_properties -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage/models.py tests/test_submission.py
git commit -m "feat(storage): them cot tags/regular_desc/properties_set_at vao SubmissionModel"
```

---

### Task 4: `SubmissionManager.set_properties()`

**Files:**
- Modify: `src/submission/manager.py` (thêm dataclass `PropertiesResult` + method
  `set_properties` vào `class SubmissionManager`)
- Test: `tests/test_submission.py`

**Interfaces:**
- Consumes: `WQBrainClient.patch()` (Task 1), `FakeClient.patch()/queue_patch()` (Task 2),
  `SubmissionModel.tags/regular_desc/properties_set_at` (Task 3).
- Produces: `SubmissionManager.set_properties(wq_alpha_id: str, *, name=None, tags=None,
  regular_desc=None, combo_desc=None, selection_desc=None, color=None) -> PropertiesResult`
  — dùng bởi sub-project A sau này (ngoài phạm vi plan này).

- [ ] **Step 1: Viết test FAIL (case cơ bản: không có row cũ -> insert row mới)**

Thêm vào `tests/test_submission.py`:

```python
# --------------------------------------------------------------- set_properties
def test_set_properties_insert_row_moi_khi_chua_tung_submit():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_patch(FakeResponse(200, json_data={"id": "WQ1"}))
    mgr = SubmissionManager(client, sf, FakeCorr())

    result = mgr.set_properties("WQ1", tags=["PowerPoolSelected"], regular_desc="Idea: " + "x" * 100)
    assert result.status == "ok"

    method, path, kwargs = client.calls[-1]
    assert method == "PATCH"
    assert path == "/alphas/WQ1"
    payload = kwargs["json"]
    assert payload["tags"] == ["PowerPoolSelected"]
    assert payload["regular"] == {"description": "Idea: " + "x" * 100}
    assert "name" not in payload  # None -> không đưa vào payload

    session = sf()
    try:
        row = session.query(SubmissionModel).filter_by(alpha_id="WQ1").one()
        assert row.status == "properties_set"
        assert row.tags == '["PowerPoolSelected"]'
        assert row.properties_set_at is not None
    finally:
        session.close()


def test_set_properties_update_row_da_submit():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_post(FakeResponse(201))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))
    mgr.submit("WQ1")  # tạo sẵn 1 row status=submitted

    client.queue_patch(FakeResponse(200, json_data={"id": "WQ1"}))
    mgr.set_properties("WQ1", tags=["t1"])

    session = sf()
    try:
        rows = session.query(SubmissionModel).filter_by(alpha_id="WQ1").all()
        assert len(rows) == 1  # KHÔNG insert thêm row mới
        assert rows[0].status == "submitted"  # giữ nguyên status gốc
        assert rows[0].tags == '["t1"]'
    finally:
        session.close()


def test_set_properties_goi_lai_cung_payload_thi_bo_qua():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_patch(FakeResponse(200, json_data={"id": "WQ1"}))
    mgr = SubmissionManager(client, sf, FakeCorr())

    r1 = mgr.set_properties("WQ1", tags=["a"], regular_desc="mo ta")
    assert r1.status == "ok"
    n_calls_before = len(client.calls)

    r2 = mgr.set_properties("WQ1", tags=["a"], regular_desc="mo ta")
    assert r2.status == "unchanged"
    assert len(client.calls) == n_calls_before  # không gọi PATCH thêm


def test_set_properties_loi_http_khong_crash():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    _seed(sf)

    client = FakeClient()
    client.queue_patch(FakeResponse(500, text="server error"))
    mgr = SubmissionManager(client, sf, FakeCorr())

    result = mgr.set_properties("WQ1", tags=["a"])
    assert result.status == "error"

    session = sf()
    try:
        row = session.query(SubmissionModel).filter_by(alpha_id="WQ1").one()
        assert row.status == "properties_set"
        assert row.properties_set_at is None  # lỗi -> không đánh dấu đã set thành công
    finally:
        session.close()
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/test_submission.py -k set_properties -v`
Expected: FAIL với `AttributeError: 'SubmissionManager' object has no attribute
'set_properties'`

- [ ] **Step 3: Cài tối thiểu**

Trong `src/submission/manager.py`, thêm import `json` ở đầu file (sau `import uuid`) và thêm
dataclass + method:

```python
import json
import uuid
from dataclasses import dataclass
```

Thêm dataclass `PropertiesResult` ngay dưới `SubmissionResult`:

```python
@dataclass
class PropertiesResult:
    wq_alpha_id: str
    status: str  # ok/unchanged/error
    detail: str = ""
```

Thêm method vào `class SubmissionManager`, sau `submit()` (trước `run_daily`):

```python
    # ---------------------------------------------------------- set_properties
    def set_properties(
        self,
        wq_alpha_id: str,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        regular_desc: str | None = None,
        combo_desc: str | None = None,
        selection_desc: str | None = None,
        color: str | None = None,
    ) -> PropertiesResult:
        """Set name/color/tags/mô tả cho alpha qua PATCH /alphas/{id} (T-C.4). Idempotent:
        bỏ qua gọi API nếu tags+regular_desc giống hệt lần set gần nhất đã lưu."""
        payload: dict = {}
        if name:
            payload["name"] = name
        if color:
            payload["color"] = color
        if tags:
            payload["tags"] = tags
        if selection_desc:
            payload["selectionDesc"] = selection_desc
        if combo_desc:
            payload["comboDesc"] = combo_desc
        if regular_desc:
            payload["regular"] = {"description": regular_desc}

        tags_json = json.dumps(tags) if tags else None
        session = self.session_factory()
        try:
            row = (
                session.query(SubmissionModel)
                .filter(SubmissionModel.alpha_id == wq_alpha_id)
                .order_by(SubmissionModel.submitted_at.desc())
                .first()
            )
            if row is not None and row.tags == tags_json and row.regular_desc == regular_desc:
                return PropertiesResult(wq_alpha_id, "unchanged", "giống lần set trước")
        finally:
            session.close()

        try:
            resp = self.client.patch(f"/alphas/{wq_alpha_id}", json=payload)
        except Exception as exc:  # noqa: BLE001 - không để pipeline crash
            self._record_properties(wq_alpha_id, tags_json, regular_desc, ok=False)
            return PropertiesResult(wq_alpha_id, "error", str(exc))

        if resp.status_code not in (200, 201):
            self._record_properties(wq_alpha_id, tags_json, regular_desc, ok=False)
            return PropertiesResult(wq_alpha_id, "error", f"HTTP {resp.status_code}")

        self._record_properties(wq_alpha_id, tags_json, regular_desc, ok=True)
        return PropertiesResult(wq_alpha_id, "ok", "da set properties")

    def _record_properties(
        self, wq_alpha_id: str, tags_json: str | None, regular_desc: str | None, *, ok: bool
    ) -> None:
        session = self.session_factory()
        try:
            row = (
                session.query(SubmissionModel)
                .filter(SubmissionModel.alpha_id == wq_alpha_id)
                .order_by(SubmissionModel.submitted_at.desc())
                .first()
            )
            set_at = _utcnow() if ok else None
            if row is not None:
                row.tags = tags_json
                row.regular_desc = regular_desc
                row.properties_set_at = set_at
            else:
                session.add(
                    SubmissionModel(
                        id=uuid.uuid4().hex,
                        alpha_id=wq_alpha_id,
                        status="properties_set",
                        tags=tags_json,
                        regular_desc=regular_desc,
                        properties_set_at=set_at,
                    )
                )
            session.commit()
        finally:
            session.close()
```

Thêm import `_utcnow` vào đầu file (dùng lại helper đã có trong `src/storage/models.py`):

```python
from src.storage.models import AlphaModel, SimulationModel, SubmissionModel, _utcnow
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/test_submission.py -k set_properties -v`
Expected: PASS (4/4 test case)

- [ ] **Step 5: Chạy toàn bộ suite, xác nhận không vỡ gì**

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS toàn bộ (trừ 1 fail có sẵn không liên quan:
`tests/test_db_postgres.py::test_make_engine_postgres_backend`, thiếu module `psycopg`)

- [ ] **Step 6: Commit**

```bash
git add src/submission/manager.py tests/test_submission.py
git commit -m "feat(submission): them SubmissionManager.set_properties() (PATCH /alphas/id)"
```

---

## Self-Review (đã chạy)

- **Spec coverage**: đối chiếu với `docs/superpowers/specs/2026-07-02-submission-compliance-roadmap-design.md`
  phần "Phần 1 — Sub-project C": Component 1 (client.patch) = Task 1; Component 2
  (set_properties) = Task 4; Component 3 (DB cột mới + quy tắc update/insert) = Task 3 + Task 4;
  Testing case 1-7 trong spec = phủ đủ bởi Task 1 (case 1), Task 4 (case 2-7). Không có gap.
- **Placeholder scan**: không còn "TBD"/"tương tự Task N" — mọi step có code đầy đủ.
- **Type consistency**: `PropertiesResult.status` dùng `"ok"/"unchanged"/"error"` nhất quán
  giữa Task 4 spec và code; `set_properties()` chữ ký khớp với phần "Ngoài phạm vi" ghi trong
  spec gốc (sub-project A sẽ gọi lại đúng chữ ký này).
