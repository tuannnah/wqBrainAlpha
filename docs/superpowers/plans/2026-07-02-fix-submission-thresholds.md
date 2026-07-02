# Fix Submission Thresholds (Sub-project B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bỏ filter `MIN_SHARPE=1.5`/`MIN_FITNESS=1.2` sai/thừa trong `SubmissionManager` (WQ
Brain đã tự chấm đúng qua `is.checks` → `SimulationResult.status`), và lưu lại TÊN check bị
FAIL để audit/debug thay vì chỉ biết "failed" chung chung.

**Architecture:** `Simulator._fetch_metrics()` đã đọc `is.checks` từ WQ nhưng chỉ dùng để suy
ra `status` — thêm field `failed_checks` giữ nguyên danh sách tên check FAIL, truyền xuống DB
qua `AlphaRepository.save_simulation()`. `SubmissionManager.select_candidates()` bỏ lớp filter
Sharpe/Fitness cứng, chỉ còn dựa vào `status == "passed"` (đã đúng).

**Tech Stack:** Python, SQLAlchemy, pytest — không thêm dependency mới.

## Global Constraints

- TDD bắt buộc: test FAIL trước, xác nhận đúng lý do fail, code tối thiểu, xác nhận PASS.
- Code/comment/commit tiếng Việt có dấu.
- Mỗi task = 1 commit.
- Chạy test: `venv/Scripts/python -m pytest`.
- KHÔNG dựng bảng ngưỡng Sharpe/Fitness theo region/delay, KHÔNG viết IS-Ladder xấp xỉ local,
  KHÔNG thêm gate Turnover/Weight riêng — `status` của WQ đã đúng các gate này (xem đính chính
  trong `docs/superpowers/specs/2026-07-02-submission-compliance-roadmap-design.md`).

---

### Task 1: `SimulationResult.failed_checks` + lưu DB

**Files:**
- Modify: `src/simulation/simulator.py:127-143` (dataclass `SimulationResult`), `:293-320`
  (`_fetch_metrics`)
- Modify: `src/storage/models.py` (class `SimulationModel`, thêm cột)
- Modify: `src/storage/repository.py:96-136` (`AlphaRepository.save_simulation`)
- Test: `tests/test_simulator.py`, `tests/test_storage.py`

**Interfaces:**
- Consumes: `is_block.get("checks")` (đã có sẵn trong `_fetch_metrics`, dòng 311).
- Produces: `SimulationResult.failed_checks: list[str]` (dùng bởi
  `AlphaRepository.save_simulation`), `SimulationModel.failed_checks: str | None`
  (JSON-encoded list, dùng bởi audit/debug sau này — KHÔNG dùng để gate lựa chọn, chỉ để đọc).

- [ ] **Step 1: Viết test FAIL cho `Simulator`**

Thêm vào `tests/test_simulator.py`, ngay sau `test_simulate_failed_khi_check_fail`:

```python
def test_simulate_luu_ten_check_bi_fail():
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-3"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-y"}))
    client.queue_get(
        FakeResponse(
            200,
            json_data={
                "is": {
                    "sharpe": 0.2,
                    "checks": [
                        {"name": "LOW_SHARPE", "result": "FAIL"},
                        {"name": "LOW_FITNESS", "result": "FAIL"},
                        {"name": "LOW_TURNOVER", "result": "PASS"},
                    ],
                }
            },
        )
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")
    assert result.failed_checks == ["LOW_SHARPE", "LOW_FITNESS"]


def test_simulate_failed_checks_rong_khi_toan_bo_pass():
    client = FakeClient()
    client.queue_post(FakeResponse(201, headers={"Location": "/simulations/sim-4"}))
    client.queue_get(FakeResponse(200, json_data={"status": "COMPLETE", "alpha": "alpha-z"}))
    client.queue_get(
        FakeResponse(200, json_data={"is": {"sharpe": 1.8, "checks": [{"name": "LOW_SHARPE", "result": "PASS"}]}})
    )

    sim = Simulator(
        client, rate_limiter=_no_sleep_limiter(), sleep_func=lambda *_: None, time_func=lambda: 0.0
    )
    result = sim.simulate("rank(close)")
    assert result.failed_checks == []
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/test_simulator.py -k failed_checks -v`
Expected: FAIL với `AttributeError: 'SimulationResult' object has no attribute 'failed_checks'`

- [ ] **Step 3: Cài tối thiểu trong `src/simulation/simulator.py`**

Thêm import `field` nếu chưa có (đã có `from dataclasses import dataclass, field` — kiểm tra
đầu file, nếu chỉ có `dataclass` thì sửa thành `from dataclasses import dataclass, field`).

Sửa `class SimulationResult`, thêm field cuối cùng (trước `raw`):

```python
@dataclass
class SimulationResult:
    expression: str
    alpha_id: str | None = None
    status: str = "error"  # passed/failed/error
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    returns: float | None = None
    drawdown: float | None = None
    margin: float | None = None
    os_sharpe: float | None = None
    os_fitness: float | None = None
    failed_checks: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
```

Sửa `_fetch_metrics`, ngay sau dòng tính `status` (sau `status = "failed" if failed else
"passed"`), thêm:

```python
        failed_check_names = [
            c.get("name") for c in checks
            if isinstance(c, dict) and c.get("result") == "FAIL" and c.get("name")
        ]
```

Và thêm `failed_checks=failed_check_names` vào lệnh `return SimulationResult(...)` ngay sau
dòng `status=status,`.

- [ ] **Step 4: Chạy test Simulator, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/test_simulator.py -k failed_checks -v`
Expected: PASS (2/2)

- [ ] **Step 5: Viết test FAIL cho lưu DB**

Thêm vào `tests/test_storage.py`, ngay sau `test_save_simulation_persists_alpha_va_metrics`:

```python
def test_save_simulation_luu_failed_checks():
    engine = init_db(_engine())
    session_factory = make_session_factory(engine)

    result = SimulationResult(
        expression="rank(close)", alpha_id="a1", status="failed",
        sharpe=0.2, failed_checks=["LOW_SHARPE", "LOW_FITNESS"], raw={"is": {}},
    )
    repo = AlphaRepository(session_factory)
    sim_id = repo.save_simulation(result, region="USA", universe="TOP3000")

    session = session_factory()
    try:
        sim = session.get(SimulationModel, sim_id)
        import json
        assert json.loads(sim.failed_checks) == ["LOW_SHARPE", "LOW_FITNESS"]
    finally:
        session.close()
```

- [ ] **Step 6: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/test_storage.py::test_save_simulation_luu_failed_checks -v`
Expected: FAIL với `TypeError` hoặc `AttributeError` (cột `failed_checks` chưa tồn tại trên
`SimulationModel`, hoặc `save_simulation` chưa lưu nó → giá trị `None`, `json.loads(None)` sẽ
raise `TypeError`)

- [ ] **Step 7: Cài tối thiểu**

Trong `src/storage/models.py`, sửa `class SimulationModel` — thêm cột sau `raw_result`:

```python
    raw_result = Column(Text)  # full JSON
    failed_checks = Column(Text)  # JSON-encoded list[str] tên check WQ tự FAIL (sub-project B)
    sim_at = Column(DateTime, default=_utcnow)
```

Trong `src/storage/repository.py`, sửa `AlphaRepository.save_simulation` — thêm import `json`
đã có sẵn ở đầu file (kiểm tra `import json` tồn tại), rồi thêm dòng vào `SimulationModel(...)`
ngay sau `raw_result=json.dumps(result.raw, ensure_ascii=False),`:

```python
                    raw_result=json.dumps(result.raw, ensure_ascii=False),
                    failed_checks=json.dumps(result.failed_checks, ensure_ascii=False),
```

- [ ] **Step 8: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/test_storage.py::test_save_simulation_luu_failed_checks -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/simulation/simulator.py src/storage/models.py src/storage/repository.py tests/test_simulator.py tests/test_storage.py
git commit -m "feat(submission): luu ten check WQ tu FAIL (failed_checks) de audit"
```

---

### Task 2: Bỏ filter `MIN_SHARPE`/`MIN_FITNESS` thừa/sai trong `SubmissionManager`

**Files:**
- Modify: `src/submission/manager.py:30-88` (class `SubmissionManager`, `select_candidates`)
- Test: `tests/test_submission.py`

**Interfaces:**
- Consumes: `SimulationModel.status` (đã đúng, không đổi).
- Produces: `SubmissionManager` không còn `MIN_SHARPE`/`MIN_FITNESS`/tham số `min_sharpe`/
  `min_fitness` — bất kỳ code nào gọi các tham số này (không có, đã grep xác nhận) sẽ lỗi rõ
  ràng thay vì âm thầm sai.

- [ ] **Step 1: Sửa fixture `_seed()` trong `tests/test_submission.py` cho đúng ngữ nghĩa mới
  (WQ tự đánh `status` — không phải tool tự lọc Sharpe/Fitness)**

Thay toàn bộ hàm `_seed` bằng:

```python
def _seed(session_factory):
    session = session_factory()
    try:
        def add(alpha_id, wq_id, sharpe, fitness, score, status="passed"):
            session.add(AlphaModel(id=alpha_id, expression=f"rank({alpha_id})", source="ga"))
            session.add(
                SimulationModel(
                    id="s_" + alpha_id,
                    alpha_id=alpha_id,
                    wq_alpha_id=wq_id,
                    region="USA",
                    universe="TOP3000",
                    sharpe=sharpe,
                    fitness=fitness,
                    score=score,
                    status=status,
                )
            )

        add("a1", "WQ1", 2.0, 1.5, 0.9)  # WQ tự PASS toàn bộ check -> đạt
        add("a2", "WQ2", 1.6, 1.3, 0.8)  # WQ tự PASS -> đạt
        add("a3", "WQ3", 1.0, 1.3, 0.5, status="failed")  # WQ tự FAIL (vd LOW_SHARPE) -> loại
        add("a4", "WQ4", 2.0, 1.0, 0.7, status="failed")  # WQ tự FAIL (vd LOW_FITNESS) -> loại
        add("a5", "WQ5", 1.9, 1.4, 0.95, status="failed")  # failed -> loại
        session.commit()
    finally:
        session.close()
```

(Đây LÀ bước "viết test trước" của task này — 4 test hiện có `test_select_candidates_chi_lay_
dat_nguong_sap_theo_score`, `test_submit_reject_khi_correlation_cao`, `test_submit_thanh_cong`,
`test_run_daily_dry_run_khong_ghi_submission` dùng `_seed()` này, kỳ vọng vẫn `ids ==
["WQ1", "WQ2"]` như cũ — chỉ đổi LÝ DO loại a3/a4 từ "sharpe/fitness thấp" sang "status=failed"
đúng thực tế WQ.)

- [ ] **Step 2: Chạy test, xác nhận FAIL đúng lý do (còn code cũ filter theo sharpe/fitness
  nên KHÔNG fail — đây là bước xác nhận test hiện tại VẪN PASS trước khi sửa code, tức
  refactor không lén đổi hành vi mong đợi)**

Run: `venv/Scripts/python -m pytest tests/test_submission.py -v`
Expected: TOÀN BỘ vẫn PASS (vì a3 sharpe=1.0 vẫn bị filter bởi `MIN_SHARPE` cũ dù status giờ
là "failed" — status="failed" tự nó ĐÃ bị loại bởi `.filter(SimulationModel.status ==
"passed")` sẵn có, nên thay đổi fixture ở Step 1 không làm gì fail cả; đây là bước xác nhận an
toàn trước khi refactor Step 3).

- [ ] **Step 3: Xoá filter thừa/sai trong `src/submission/manager.py`**

Sửa `class SubmissionManager`:

```python
class SubmissionManager:
    DAILY_QUOTA = 10

    def __init__(
        self,
        client,
        session_factory,
        correlation_checker,
        daily_quota: int | None = None,
        diversify: bool = False,
        max_struct_similarity: float = 0.9,
    ):
        self.client = client
        self.session_factory = session_factory
        self.correlation = correlation_checker
        self.daily_quota = daily_quota if daily_quota is not None else self.DAILY_QUOTA
        # T7.1: loại alpha trùng cấu trúc (AST) với alpha đã chọn trong cùng tập nộp.
        self.diversify = diversify
        self.max_struct_similarity = max_struct_similarity
```

Sửa `select_candidates`:

```python
    # --------------------------------------------------------------- selection
    def select_candidates(self) -> list[Candidate]:
        """Chọn alpha đã pass sim để nộp. `status == "passed"` ĐÃ phản ánh đúng
        Sharpe/Fitness/Turnover/Weight/IS-Ladder thật theo tier tài khoản — WQ Brain tự chấm
        qua `is.checks` (xem `Simulator._fetch_metrics`), KHÔNG tự đoán lại ngưỡng ở đây
        (sub-project B, xem docs/superpowers/specs/2026-07-02-submission-compliance-roadmap-design.md)."""
        session = self.session_factory()
        try:
            submitted = {
                row[0]
                for row in session.query(SubmissionModel.alpha_id)
                .filter(SubmissionModel.status == "submitted")
                .all()
            }
            rows = (
                session.query(SimulationModel, AlphaModel)
                .join(AlphaModel, SimulationModel.alpha_id == AlphaModel.id)
                .filter(SimulationModel.status == "passed")
                .filter(SimulationModel.wq_alpha_id.isnot(None))
                .order_by(SimulationModel.score.desc())
                .all()
            )
        finally:
            session.close()

        candidates: list[Candidate] = []
        seen: set[str] = set()
        for sim, alpha in rows:
            if sim.wq_alpha_id in submitted or sim.wq_alpha_id in seen:
                continue
            seen.add(sim.wq_alpha_id)
            candidates.append(
                Candidate(sim.wq_alpha_id, alpha.expression, sim.sharpe, sim.fitness, sim.score)
            )
        return candidates
```

(Chỉ xoá 2 dòng `.filter(SimulationModel.sharpe >= self.min_sharpe)` và
`.filter(SimulationModel.fitness >= self.min_fitness)` so với bản gốc — phần còn lại giữ
nguyên y hệt.)

- [ ] **Step 4: Chạy lại toàn bộ test, xác nhận vẫn PASS**

Run: `venv/Scripts/python -m pytest tests/test_submission.py tests/test_submission_diversity.py -v`
Expected: PASS toàn bộ, không giảm test nào so với trước Step 3 (đúng như dự đoán ở Step 2 —
`status=="passed"` một mình đã đủ tái tạo hành vi cũ).

- [ ] **Step 5: Chạy toàn bộ suite, xác nhận không vỡ chỗ khác**

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS hết, trừ 1 fail có sẵn không liên quan
(`tests/test_db_postgres.py::test_make_engine_postgres_backend`, thiếu `psycopg`).

- [ ] **Step 6: Commit**

```bash
git add src/submission/manager.py tests/test_submission.py
git commit -m "fix(submission): bo MIN_SHARPE/MIN_FITNESS thua - status da dung tier that"
```

---

## Self-Review (đã chạy)

- **Spec coverage**: đối chiếu mục "Sub-project B" (đã sửa) trong
  `docs/superpowers/specs/2026-07-02-submission-compliance-roadmap-design.md`: việc 1 (xoá
  MIN_SHARPE/MIN_FITNESS) = Task 2; việc 2 (lưu failed_checks) = Task 1; việc 3 (self-corr
  alternate) và việc 4 (prod-corr) — CỐ Ý không có task, đã ghi rõ lý do (dữ liệu không đủ tin
  cậy) ngay trong spec, không phải thiếu sót.
- **Placeholder scan**: không còn TBD, mọi step có code đầy đủ.
- **Type consistency**: `SimulationResult.failed_checks: list[str]` khớp
  `SimulationModel.failed_checks: Text (JSON-encoded list[str])` khớp cách
  `AlphaRepository.save_simulation` serialize — nhất quán xuyên 3 file.
