# Menu mục 6 — Xem & nộp alpha đã tìm được Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm mục 6 vào menu `run.bat`/`main.py start`: xem danh sách alpha đã mô phỏng
`status='passed'` (từ mục 5 Auto SIM hoặc CLI `research`/`marathon`) đủ điều kiện nộp, hỏi xác
nhận, rồi mới nộp THẬT — theo đúng lựa chọn của người dùng ("khi tìm được alpha có thể submit
thì lưu vào DB để tôi tự submit") — KHÔNG tự động nộp trong lúc mục 5 đang chạy, submit vẫn là
bước riêng do người dùng xác nhận, chỉ làm nó DỄ TÌM/DỄ TEST hơn (ngay trong menu quen thuộc,
thay vì phải nhớ gõ lệnh CLI `submit` riêng).

**Architecture:** Hàm mới `_menu_view_submit(state)` tái dùng `SubmissionManager.run_daily()`
đã có (đã tích hợp sẵn: `status=="passed"` đúng chuẩn WQ thật — sub-project B; tự gắn tag
Power Pool sau khi nộp — sub-project A/C). Gọi `run_daily(dry_run=True)` để xem trước, hỏi xác
nhận qua `input()`, gọi lại `run_daily(dry_run=False)` nếu người dùng gõ "yes".

## Global Constraints

- TDD bắt buộc: test FAIL trước, code tối thiểu, xác nhận PASS.
- Code/comment/commit tiếng Việt có dấu.
- 1 task = 1 commit.
- Chạy test: `venv/Scripts/python -m pytest`.
- KHÔNG tự nộp nếu người dùng không gõ đúng "yes" — mặc định Enter/bất kỳ chuỗi khác = bỏ qua
  (an toàn, đúng lựa chọn người dùng đã chốt).

---

### Task 1: `_menu_view_submit()` + gắn vào menu mục 6

**Files:**
- Modify: `main.py:1627-1696` (thêm hàm mới sau `_menu_auto_sim`, sửa `_print_menu`, sửa
  `start()`)
- Test: `tests/test_menu_counts.py`

**Interfaces:**
- Consumes: `SubmissionManager.run_daily(dry_run)` (đã có, sub-project B/C/A đã tích hợp sẵn
  vào `submit()`/`select_candidates()` mà `run_daily` gọi), `CorrelationChecker` (đã có).
- Produces: `main._menu_view_submit(state: _MenuState) -> None` — hàm mới, test trực tiếp qua
  import `main` (giống `_menu_counts` đã có test trong `tests/test_menu_counts.py`).

- [ ] **Step 1: Viết test FAIL**

Thêm vào cuối `tests/test_menu_counts.py`:

```python
# ------------------------------------------------------- mục 6: xem & nộp
def _seed_candidate(sf, *, sharpe=1.5, hypothesis=None):
    from src.storage.models import AlphaModel, SimulationModel

    session = sf()
    session.add(AlphaModel(id="a1", expression="rank(close)", source="ga", hypothesis=hypothesis))
    session.add(SimulationModel(
        id="s1", alpha_id="a1", wq_alpha_id="WQ1", region="USA", universe="TOP3000",
        sharpe=sharpe, fitness=1.2, score=0.9, status="passed",
    ))
    session.commit()
    session.close()


def test_menu_view_submit_khong_co_candidate_thi_khong_goi_submit(monkeypatch):
    from tests.fakes import FakeClient

    sf = make_session_factory(init_db(make_engine("sqlite:///:memory:")))
    state = main._MenuState()
    state.session_factory = sf
    state.client = FakeClient()
    monkeypatch.setattr("builtins.input", lambda _: "")

    main._menu_view_submit(state)

    assert state.client.calls == []


def test_menu_view_submit_nguoi_dung_tu_choi_thi_khong_nop(monkeypatch):
    from tests.fakes import FakeClient, FakeResponse

    sf = make_session_factory(init_db(make_engine("sqlite:///:memory:")))
    _seed_candidate(sf)
    state = main._MenuState()
    state.session_factory = sf
    state.client = FakeClient()
    state.client.queue_get(FakeResponse(200, json_data={"max": 0.1}))  # dry-run preview: check corr
    monkeypatch.setattr("builtins.input", lambda _: "")  # Enter -> từ chối

    main._menu_view_submit(state)

    assert not any(c[0] == "POST" for c in state.client.calls)


def test_menu_view_submit_xac_nhan_yes_thi_nop_that(monkeypatch):
    from tests.fakes import FakeClient, FakeResponse

    sf = make_session_factory(init_db(make_engine("sqlite:///:memory:")))
    _seed_candidate(sf)  # không có hypothesis -> không kích hoạt gắn tag Power Pool (đã test riêng)
    state = main._MenuState()
    state.session_factory = sf
    state.client = FakeClient()
    state.client.queue_get(FakeResponse(200, json_data={"max": 0.1}))  # dry-run preview
    state.client.queue_get(FakeResponse(200, json_data={"max": 0.1}))  # check lại lúc nộp thật
    state.client.queue_post(FakeResponse(201))  # submit thật
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    main._menu_view_submit(state)

    assert sum(1 for c in state.client.calls if c[0] == "POST") == 1
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv/Scripts/python -m pytest tests/test_menu_counts.py -k menu_view_submit -v`
Expected: FAIL với `AttributeError: module 'main' has no attribute '_menu_view_submit'`

- [ ] **Step 3: Cài tối thiểu**

Trong `main.py`, thêm hàm mới ngay sau `_menu_auto_sim` (sau dòng 1642, trước
`def _print_menu`):

```python
def _menu_view_submit(state: _MenuState) -> None:
    """Mục 6: xem alpha đã mô phỏng đạt (status='passed', từ mục 5/CLI research/marathon) và
    tự chọn nộp THẬT hay chỉ xem trước — dry-run mặc định, hỏi xác nhận rõ ràng trước khi tốn
    quota nộp ngày thật. Alpha đạt điều kiện Power Pool sẽ tự được gắn tag (sub-project A/C,
    đã tích hợp sẵn trong SubmissionManager.submit())."""
    from src.submission.correlation import CorrelationChecker
    from src.submission.manager import SubmissionManager

    manager = SubmissionManager(state.client, state.session_factory, CorrelationChecker(state.client))
    preview = manager.run_daily(dry_run=True)
    if not preview:
        console.print(
            "[yellow]Chưa có alpha nào đạt điều kiện nộp (status='passed', chưa nộp, qua được "
            "lọc self-correlation/trùng cấu trúc).[/yellow]"
        )
        return

    table = Table(title=f"Sẽ nộp (dry-run) — {len(preview)} alpha, quota/ngày={manager.daily_quota}")
    table.add_column("#")
    table.add_column("WQ Alpha")
    table.add_column("Expression", overflow="fold")
    table.add_column("Sharpe", justify="right")
    table.add_column("Score", justify="right")
    for i, c in enumerate(preview, 1):
        table.add_row(
            str(i), c.wq_alpha_id, c.expression,
            f"{c.sharpe:.3f}" if c.sharpe is not None else "—",
            f"{c.score:.3f}" if c.score is not None else "—",
        )
    console.print(table)

    answer = input(
        f"\nNộp THẬT {len(preview)} alpha này lên WQ Brain (tốn quota nộp ngày thật)? "
        "Gõ 'yes' để xác nhận, Enter để bỏ qua: "
    ).strip().lower()
    if answer != "yes":
        console.print("[dim]Đã bỏ qua — chưa nộp gì, có thể chọn lại mục này sau.[/dim]")
        return

    submitted = manager.run_daily(dry_run=False)
    console.print(
        f"[green]Đã nộp {len(submitted)} alpha.[/green] Alpha đạt điều kiện Power Pool "
        "(Sharpe≥1.0, operator/field trong giới hạn, có mô tả) sẽ tự được gắn tag PowerPoolSelected."
    )
```

Sửa `_print_menu`, thêm dòng ngay sau `console.print(" 5) Auto SIM ...")`:

```python
    console.print(" 5) Auto SIM (vòng kín AI+MiniBrain, cần đăng nhập)")
    console.print(" 6) Xem & nộp alpha đã tìm được (dry-run trước, hỏi xác nhận)")
    console.print(" 0) Thoát")
```

Sửa `start()`, đổi dòng `elif choice in {"2", "3", "5"} and not state.logged_in:` thành:

```python
            elif choice in {"2", "3", "5", "6"} and not state.logged_in:
                console.print("[yellow]Hãy đăng nhập (1) trước.[/yellow]")
```

và thêm nhánh mới ngay sau `elif choice == "5": _menu_auto_sim(state)`:

```python
            elif choice == "5":
                _menu_auto_sim(state)
            elif choice == "6":
                _menu_view_submit(state)
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv/Scripts/python -m pytest tests/test_menu_counts.py -v`
Expected: PASS toàn bộ (2 test cũ + 3 test mới = 5)

- [ ] **Step 5: Chạy toàn bộ suite + kiểm cú pháp `main.py`, xác nhận không vỡ gì**

Run: `venv/Scripts/python -c "import main"`
Expected: không lỗi

Run: `venv/Scripts/python -m pytest tests/ -q`
Expected: PASS hết, trừ 1 fail có sẵn không liên quan (`test_make_engine_postgres_backend`).

- [ ] **Step 6: Cập nhật `README.md`**

`README.md:29-47` có liệt kê 5 mục menu. Sửa đoạn (dòng 44-47):

```
5. **Auto SIM** — vòng kín AI+MiniBrain thật (GP → refine → SIM Brain →
   feedback), chạy đến khi hết quota Brain hoặc Ctrl+C để dừng tay.

DB tách theo email đăng nhập (mỗi tài khoản một file `wq_alpha_<email>.db`).
```

thành:

```
5. **Auto SIM** — vòng kín AI+MiniBrain thật (GP → refine → SIM Brain →
   feedback), chạy đến khi hết quota Brain hoặc Ctrl+C để dừng tay.
6. **Xem & nộp alpha đã tìm được** — liệt kê alpha `status='passed'` (từ mục 5
   hoặc CLI `research`/`marathon`) đủ điều kiện nộp (dry-run trước), hỏi xác
   nhận rõ ràng trước khi nộp THẬT (tốn quota nộp ngày). Alpha đạt điều kiện
   Power Pool tự được gắn tag `PowerPoolSelected`.

DB tách theo email đăng nhập (mỗi tài khoản một file `wq_alpha_<email>.db`).
```

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_menu_counts.py README.md
git commit -m "feat(cli): them muc 6 vao menu - xem & nop alpha da tim duoc (dry-run + xac nhan)"
```

---

## Self-Review (đã chạy)

- **Spec coverage**: đúng lựa chọn người dùng đã chốt (mục 5 vẫn chỉ sim+lưu DB, không tự nộp;
  nộp là bước riêng có xác nhận) — mục 6 chỉ làm bước nộp DỄ TÌM hơn trong menu quen thuộc.
- **Placeholder scan**: sạch.
- **Type consistency**: `_menu_view_submit(state: _MenuState)` cùng chữ ký với các hàm `_menu_*`
  khác trong file; tái dùng `SubmissionManager.run_daily()` đúng chữ ký đã có, không thêm API
  mới ở tầng `SubmissionManager`.
