# Wire Power Pool + Genius Report vào luồng thật Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nối các hàm đã viết (sub-project A/C: `check_power_pool_eligibility`,
`build_power_pool_description`, `set_properties`; sub-project G: `genius_report`) vào luồng
CHẠY THẬT — hiện chúng chỉ là hàm/module độc lập có test, KHÔNG có nơi nào trong `main.py`
gọi tới (đã xác nhận bằng grep). Đây là bước gói vào pipeline mà trước đó tôi đã cố tình để
lại cho bạn, nhưng bạn nhờ tôi làm luôn.

**Phạm vi CHỈ làm phần AN TOÀN/CHẮC CHẮN đúng theo tài liệu** (KHÔNG động tới phần rủi ro chưa
xác nhận):
- **Tự động gắn tag `PowerPoolSelected` + mô tả** cho alpha đã nộp thành công qua đường REGULAR
  (`SubmissionManager.submit()` gọi `POST /alphas/{id}/submit` bình thường) VÀ đồng thời đạt
  điều kiện Power Pool (Sharpe≥1.0, operator/field trong giới hạn) — đây là trường hợp
  **[Power Pool + Regular]**, theo tài liệu KHÔNG cần Power Pool Theme (Theme chỉ bắt buộc cho
  "pure Power Pool" — alpha KHÔNG pass regular test — trường hợp đó KHÔNG tự động nộp ở đây vì
  cần Theme mà ta chưa xác nhận được cách set).
- **Lệnh CLI `genius-report`** — in báo cáo, không đụng gì tới việc nộp.

**KHÔNG làm** (rủi ro/thiếu xác nhận, xem `docs/superpowers/plans/2026-07-02-power-pool-alphas.md`):
- Tự động nộp "pure Power Pool" alpha (Sharpe≥1.0 nhưng KHÔNG pass regular `status`) — thiếu
  Power Pool Theme, gọi liều có thể tốn 1 lượt submit thật mà WQ từ chối hoặc submit sai.
- Kiểm tra quota trước khi nộp loại "pure Power Pool" — không áp dụng vì không tự nộp loại đó.
- Wiring D (`is_single_dataset_alpha`) vào CLI — để dành đợt sau nếu cần, không phải phần cấp
  bách của luồng nộp.

**Architecture:** Thêm method `SubmissionManager._tag_if_power_pool_eligible(wq_alpha_id)`,
gọi từ `submit()` NGAY SAU khi `_record(result)` ghi row `status="submitted"` (để
`set_properties()`/`_record_properties()` tìm thấy đúng row mà UPDATE thay vì tạo row rác).
Lỗi ở bước tag KHÔNG được làm hỏng kết quả submit chính (đã nộp thành công rồi, không thể/
không nên rollback).

## Global Constraints

- TDD bắt buộc cho Task 1 (business logic thật). Task 2 (CLI report thuần) theo đúng convention
  hiện có của repo — các lệnh report-only khác (`cache-status`, `list-fields`) không có test CLI
  riêng (logic đã test ở tầng hàm), Task 2 cũng vậy — KHÔNG cần viết test CLI mới.
- Code/comment/commit tiếng Việt có dấu.
- Mỗi task = 1 commit.
- Chạy test: `venv/Scripts/python -m pytest`.

---

### Task 1: `SubmissionManager._tag_if_power_pool_eligible()` — tự gắn tag sau khi nộp thành công

**Files:**
- Modify: `src/submission/manager.py` (method `submit`, dòng 94-116; thêm method mới)
- Test: `tests/test_submission.py`

**Interfaces:**
- Consumes: `check_power_pool_eligibility(expr, sharpe)`, `build_power_pool_description(hyp)`,
  `is_valid_power_pool_description(text)` (`src/scoring/power_pool.py`, sub-project A),
  `Hypothesis.from_dict(dict)` (`src/llm/hypothesis.py`), `self.set_properties(...)` (đã có,
  sub-project C).
- Produces: hành vi mới của `submit()` — không thêm API công khai mới.

- [ ] **Step 1: Viết test FAIL**

`tests/test_submission.py` hiện CHƯA có `import json` ở đầu file (kiểm tra bằng
`grep -n "^import json" tests/test_submission.py` — nếu trống thì thêm dòng `import json`
ngay dưới `from __future__ import annotations` ở đầu file, trước `from sqlalchemy import
create_engine`).

Thêm vào `tests/test_submission.py`, sau các test `set_properties` hiện có:

```python
# ------------------------------------------------------- power pool auto-tag
def test_submit_tu_gan_tag_power_pool_khi_du_dieu_kien():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    hyp = {
        "observation": "Gia co phieu dao chieu sau chuoi giam manh trong ngan han lien tuc.",
        "background": "Ly thuyet mean-reversion tren thi truong von ngan han duoc ung ho rong rai.",
        "economic_rationale": "Nha dau tu phan ung thai qua roi dieu chinh lai theo thoi gian giao dich.",
        "implementation_spec": "Dung field close, cua so 5 ngay, chuan hoa bang toan tu rank toan thi truong.",
    }
    session.add(AlphaModel(
        id="a1", expression="rank(add(close, open))", source="ga", hypothesis=json.dumps(hyp),
    ))
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=1.5, status="passed",
    ))
    session.commit()
    session.close()

    client = FakeClient()
    client.queue_post(FakeResponse(201))
    client.queue_patch(FakeResponse(200, json_data={"id": "WQ1"}))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))

    result = mgr.submit("WQ1")
    assert result.status == "submitted"

    patch_calls = [c for c in client.calls if c[0] == "PATCH"]
    assert len(patch_calls) == 1
    payload = patch_calls[0][2]["json"]
    assert payload["tags"] == ["PowerPoolSelected"]
    assert "Idea:" in payload["regular"]["description"]


def test_submit_khong_gan_tag_khi_khong_du_dieu_kien_power_pool():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    session.add(AlphaModel(id="a1", expression="rank(close)", source="ga"))
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=0.5, status="passed",  # Sharpe < 1.0 -> không đạt Power Pool
    ))
    session.commit()
    session.close()

    client = FakeClient()
    client.queue_post(FakeResponse(201))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))

    result = mgr.submit("WQ1")
    assert result.status == "submitted"
    assert not any(c[0] == "PATCH" for c in client.calls)


def test_submit_khong_gan_tag_khi_thieu_hypothesis():
    engine = init_db(_engine())
    sf = make_session_factory(engine)
    session = sf()
    session.add(AlphaModel(id="a1", expression="rank(close)", source="ga"))  # không có hypothesis
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=1.5, status="passed",
    ))
    session.commit()
    session.close()

    client = FakeClient()
    client.queue_post(FakeResponse(201))
    mgr = SubmissionManager(client, sf, FakeCorr(value=0.1))

    result = mgr.submit("WQ1")
    assert result.status == "submitted"
    assert not any(c[0] == "PATCH" for c in client.calls)
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/test_submission.py -k power_pool -v`
Expected: 2 test đầu FAIL (assert sai vì hiện chưa gọi PATCH nào — `len(patch_calls) == 1`
sai vì bằng 0; hoặc `IndexError` nếu code cũ không đụng gì tới patch queue, hàng đợi patch dư
không tiêu tốn nên KHÔNG lỗi — kỳ vọng cụ thể: `test_submit_tu_gan_tag...` FAIL ở
`assert len(patch_calls) == 1` vì thực tế bằng 0). 2 test còn lại (`khong_gan_tag_khi...`) PASS
ngay từ đầu (hành vi hiện tại vốn không gọi PATCH) — đó là bình thường, KHÔNG cần fix.

- [ ] **Step 3: Cài tối thiểu**

Trong `src/submission/manager.py`, sửa `submit()`:

```python
    # ------------------------------------------------------------------ submit
    def submit(self, wq_alpha_id: str) -> SubmissionResult:
        corr = self.correlation.max_self_correlation(wq_alpha_id)
        if corr > self.correlation.max_self_corr:
            result = SubmissionResult(
                wq_alpha_id, "rejected", f"self-corr {corr:.3f} > {self.correlation.max_self_corr}", corr
            )
            self._record(result)
            return result

        try:
            resp = self.client.post(f"/alphas/{wq_alpha_id}/submit")
        except Exception as exc:  # noqa: BLE001 - không để pipeline crash
            result = SubmissionResult(wq_alpha_id, "error", str(exc), corr)
            self._record(result)
            return result

        if resp.status_code in (200, 201):
            result = SubmissionResult(wq_alpha_id, "submitted", "ok", corr)
        else:
            result = SubmissionResult(wq_alpha_id, "error", f"HTTP {resp.status_code}", corr)
        self._record(result)
        if result.status == "submitted":
            self._tag_if_power_pool_eligible(wq_alpha_id)
        return result

    def _tag_if_power_pool_eligible(self, wq_alpha_id: str) -> None:
        """Sau khi nộp REGULAR thành công, nếu alpha cũng đạt điều kiện Power Pool (Sharpe>=1.0,
        operator/field unique trong giới hạn) thì tự gắn tag PowerPoolSelected + mô tả
        Idea/Rationale — đây là [Power Pool + Regular] (đã pass regular nên KHÔNG cần Power Pool
        Theme — Theme chỉ bắt buộc cho "pure Power Pool" alpha không pass regular, loại đó KHÔNG
        tự động nộp ở đây, xem docs/superpowers/plans/2026-07-02-power-pool-alphas.md). Lỗi ở
        bước này KHÔNG được làm hỏng kết quả submit chính (đã nộp thành công rồi)."""
        import json as _json

        from src.llm.hypothesis import Hypothesis
        from src.scoring.power_pool import (
            build_power_pool_description,
            check_power_pool_eligibility,
            is_valid_power_pool_description,
        )

        session = self.session_factory()
        try:
            row = (
                session.query(SimulationModel, AlphaModel)
                .join(AlphaModel, SimulationModel.alpha_id == AlphaModel.id)
                .filter(SimulationModel.wq_alpha_id == wq_alpha_id)
                .order_by(SimulationModel.sim_at.desc())
                .first()
            )
        finally:
            session.close()
        if row is None:
            return
        sim, alpha = row

        try:
            verdict = check_power_pool_eligibility(alpha.expression, sim.sharpe)
        except Exception as exc:  # noqa: BLE001 - biểu thức lạ không được chặn kết quả submit
            logger.warning("Không kiểm được điều kiện Power Pool cho {}: {}", wq_alpha_id, exc)
            return
        if not verdict.eligible:
            return

        description = None
        if alpha.hypothesis:
            try:
                hyp = Hypothesis.from_dict(_json.loads(alpha.hypothesis))
                description = build_power_pool_description(hyp)
            except (ValueError, TypeError) as exc:
                logger.warning("Không đọc được hypothesis của {}: {}", wq_alpha_id, exc)

        if not description or not is_valid_power_pool_description(description):
            logger.info(
                "Alpha {} đạt điều kiện Power Pool nhưng thiếu mô tả >=100 ký tự -> bỏ qua gắn tag",
                wq_alpha_id,
            )
            return

        try:
            self.set_properties(wq_alpha_id, tags=["PowerPoolSelected"], regular_desc=description)
        except Exception as exc:  # noqa: BLE001 - không để hỏng kết quả submit chính
            logger.warning("Không gắn được tag Power Pool cho {}: {}", wq_alpha_id, exc)
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/test_submission.py -k power_pool -v`
Expected: PASS (3/3)

- [ ] **Step 5: Chạy toàn bộ suite (kể cả test cũ liên quan `submit`), xác nhận không vỡ gì**

Run: `venv/Scripts/python -m pytest tests/test_submission.py tests/test_submission_diversity.py -v`
Expected: PASS toàn bộ — các test `submit`/`run_daily` cũ dùng alpha KHÔNG đạt điều kiện Power
Pool (sharpe thấp hoặc không có trong DB kiểu này) nên không kích hoạt nhánh mới, hành vi cũ
giữ nguyên.

- [ ] **Step 6: Commit**

```bash
git add src/submission/manager.py tests/test_submission.py
git commit -m "feat(submission): tu gan tag PowerPoolSelected sau khi nop REGULAR thanh cong"
```

---

### Task 2: Lệnh CLI `genius-report`

**Files:**
- Modify: `main.py:1380-1383` (chèn command mới ngay sau lệnh `submit`, trước comment "Menu
  tương tác")

**Interfaces:**
- Consumes: `average_distinct_operators_per_alpha`, `average_distinct_fields_per_alpha`,
  `total_distinct_operators`, `total_distinct_fields` (`src/scoring/genius_report.py`, đã có
  test đầy đủ ở sub-project G).
- Produces: lệnh CLI mới `genius-report` — không có test riêng (theo đúng convention hiện có
  của các lệnh report-only khác như `cache-status`/`list-fields`, không có test CLI vì logic đã
  test ở tầng hàm).

- [ ] **Step 1: Chèn command mới**

Trong `main.py`, ngay sau dòng `console.print(table)` cuối hàm `submit()` (dòng 1380) và TRƯỚC
comment `# ============================ Menu tương tác (start) ============================`
(dòng 1383), chèn:

```python

@app.command("genius-report")
def genius_report_cmd() -> None:
    """Báo cáo tie-break BRAIN Genius tính được LOCAL (avg/total distinct operators/fields của
    alpha đã nộp) — CHỈ để tham khảo, KHÔNG phải gate (sub-project G)."""
    _setup_logging()
    from src.scoring.genius_report import (
        average_distinct_fields_per_alpha,
        average_distinct_operators_per_alpha,
        total_distinct_fields,
        total_distinct_operators,
    )

    engine = init_db(make_engine())
    session_factory = make_session_factory(engine)

    avg_ops = average_distinct_operators_per_alpha(session_factory)
    avg_fields = average_distinct_fields_per_alpha(session_factory)
    total_ops = total_distinct_operators(session_factory)
    total_fields = total_distinct_fields(session_factory)

    table = Table(title="BRAIN Genius — tie-break metrics (chỉ tham khảo, không phải gate)")
    table.add_column("Chỉ số")
    table.add_column("Giá trị", justify="right")
    table.add_row(
        "Avg distinct Operators/Alpha (thấp hơn tốt hơn)",
        "—" if avg_ops is None else f"{avg_ops:.2f}",
    )
    table.add_row(
        "Avg distinct Fields/Alpha (thấp hơn tốt hơn)",
        "—" if avg_fields is None else f"{avg_fields:.2f}",
    )
    table.add_row("Total distinct Operators (cao hơn tốt hơn)", str(total_ops))
    table.add_row("Total distinct Fields (cao hơn tốt hơn)", str(total_fields))
    console.print(table)
    if avg_ops is None:
        console.print("[dim]Chưa có alpha nào status='submitted' trong DB để tính.[/dim]")

```

- [ ] **Step 2: Kiểm tra cú pháp — import lệnh không lỗi**

Run: `venv/Scripts/python -c "import main"`
Expected: không lỗi (import thành công, không exception cú pháp/tên).

- [ ] **Step 3: Chạy toàn bộ suite, xác nhận không vỡ gì**

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS hết, trừ 1 fail có sẵn không liên quan (`test_make_engine_postgres_backend`).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(cli): them lenh genius-report (bao cao tie-break BRAIN Genius)"
```

---

## Self-Review (đã chạy)

- **Spec coverage**: "gắn tag Power Pool khi nộp thành công [Power Pool+Regular]" = Task 1;
  "hiển thị báo cáo Genius" = Task 2. "Pure Power Pool" submission (cần Theme) và wiring D vào
  CLI — CỐ Ý không có task, lý do ghi rõ đầu plan.
- **Placeholder scan**: sạch.
- **Type consistency**: `_tag_if_power_pool_eligible` dùng đúng `check_power_pool_eligibility`,
  `build_power_pool_description`, `is_valid_power_pool_description`, `Hypothesis.from_dict` —
  đúng chữ ký đã có sẵn từ sub-project A/C (đã đọc lại source thật trước khi viết plan, không
  đoán).
