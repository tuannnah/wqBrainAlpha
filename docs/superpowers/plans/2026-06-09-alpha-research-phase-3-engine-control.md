# Alpha Research Phase 3: Engine And Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Orchestrate autonomous ideas, three root batches, parent selection, targeted variants, review queue updates, live logs, and cooperative `quit` shutdown.

**Architecture:** The engine is a synchronous state machine so only one DeepSeek request or WorldQuant simulation is active at a time. A background input thread only sets a thread-safe stop event. Every state transition is persisted before the next external request, allowing audit and safe termination.

**Tech Stack:** Python 3.12, `threading`, `logging`, `sqlite3`, `unittest`.

---

### Task 1: Console Quit Control

**Files:**
- Create: `run_control.py`
- Create: `tests/test_run_control.py`

- [ ] **Step 1: Write failing stop-listener tests**

```python
class RunControlTest(unittest.TestCase):
    def test_quit_sets_stop_flag_and_ignores_other_input(self):
        input_func = Mock(side_effect=["status", "quit"])
        control = RunControl(input_func=input_func)

        control.start()
        control.join(timeout=1)

        self.assertTrue(control.stop_requested())
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_run_control -v
```

Expected: import error.

- [ ] **Step 3: Implement cooperative stop control**

```python
class RunControl:
    def __init__(self, input_func=input):
        self.input_func = input_func
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._listen,
            name="research-run-control",
            daemon=True,
        )
        self._thread.start()

    def _listen(self):
        while not self._stop_event.is_set():
            try:
                command = self.input_func().strip().lower()
            except (EOFError, KeyboardInterrupt):
                command = "quit"
            if command == "quit":
                self._stop_event.set()

    def request_stop(self):
        self._stop_event.set()

    def stop_requested(self):
        return self._stop_event.is_set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)
```

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_run_control -v
git add run_control.py tests/test_run_control.py
git commit -m "feat: add cooperative quit control"
```

### Task 2: Console, Rotating File, And Database Event Logging

**Files:**
- Create: `logging_setup.py`
- Create: `tests/test_logging_setup.py`

- [ ] **Step 1: Write failing fan-out and secret-redaction tests**

```python
class ResearchLoggingTest(unittest.TestCase):
    def test_event_reaches_console_file_and_database_without_secrets(self):
        stream = io.StringIO()
        logger = create_run_logger(
            run_id=run_id,
            logs_dir=logs_dir,
            research_store=store,
            stream=stream,
            max_bytes=100000,
            backup_count=2,
        )

        logger.event(
            "DEEPSEEK",
            "Đang tạo Alpha",
            {"api_key": "secret", "batch": 1},
        )

        self.assertIn("[DEEPSEEK] Đang tạo Alpha", stream.getvalue())
        self.assertNotIn("secret", stream.getvalue())
        self.assertNotIn("secret", (logs_dir / f"{run_id}.log").read_text())
        self.assertEqual(store.list_events(run_id)[0]["event_type"], "DEEPSEEK")
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_logging_setup -v
```

Expected: import error.

- [ ] **Step 3: Implement event logger**

Use `logging.handlers.RotatingFileHandler`, a console `StreamHandler`, and:

```python
class RunLogger:
    def event(self, event_type, message, payload=None, level=logging.INFO):
        safe_payload = redact(payload or {})
        rendered = f"[{event_type}] {message}"
        if safe_payload:
            rendered += " | " + json.dumps(
                safe_payload, ensure_ascii=False, sort_keys=True
            )
        self.logger.log(level, rendered)
        self.research_store.record_event(
            self.run_id, event_type, message, safe_payload
        )
```

Add `ResearchStore.list_events(run_id)`.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_logging_setup tests.test_research_store -v
git add logging_setup.py research_store.py tests/test_logging_setup.py tests/test_research_store.py
git commit -m "feat: add live and persistent research logging"
```

### Task 3: Prompt Contracts And Response Mapping

**Files:**
- Create: `alpha_prompts.py`
- Create: `tests/test_alpha_prompts.py`

- [ ] **Step 1: Write failing prompt contract tests**

```python
class AlphaPromptTest(unittest.TestCase):
    def test_root_prompt_requires_distinct_hypotheses_and_allowed_ids(self):
        system, payload = build_root_alpha_prompt(
            idea,
            candidate_context,
            count=5,
            lessons=[],
        )

        self.assertIn("exactly 5", system)
        self.assertIn("distinct hypothesis", system)
        self.assertEqual(
            {item["id"] for item in payload["allowed_fields"]},
            {"close", "returns"},
        )
        self.assertNotIn("api_key", json.dumps(payload))

    def test_variant_prompt_has_one_improvement_direction(self):
        system, payload = build_variant_prompt(
            parent,
            "REDUCE_TURNOVER",
            candidate_context,
        )
        self.assertEqual(payload["improvement_direction"], "REDUCE_TURNOVER")
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_alpha_prompts -v
```

Expected: import error.

- [ ] **Step 3: Implement three strict contracts**

Implement:

- `build_idea_prompt(catalog, lessons)`;
- `build_root_alpha_prompt(idea, context, count, lessons)`;
- `build_variant_prompt(parent, direction, context)`.

Every system prompt states:

```text
Return one JSON object only.
Use only allowed_fields, allowed_operators, and allowed_settings.
Do not invent identifiers.
Each Alpha must contain hypothesis, rationale, expression, dataset_ids,
field_ids, operator_names, and settings.
```

Root prompt requires exactly `count` Alpha objects with distinct hypotheses.
Variant prompt requires exactly one Alpha and forbids changing more than the
specified direction.

Add mapping functions that validate required keys and construct `AlphaDraft`.
Reject duplicate hypotheses in one root response.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_alpha_prompts -v
git add alpha_prompts.py tests/test_alpha_prompts.py
git commit -m "feat: define strict alpha generation prompt contracts"
```

### Task 4: Root Batch Loop And Idea Rotation

**Files:**
- Create: `research_engine.py`
- Create: `tests/test_research_engine_roots.py`

- [ ] **Step 1: Write failing three-batch rotation test**

```python
class ResearchEngineRootTest(unittest.TestCase):
    def test_three_bad_batches_exhaust_idea_and_create_next_idea(self):
        llm = FakeLlm(
            ideas=[idea_one, idea_two],
            root_batches=[
                five_bad_drafts(),
                five_bad_drafts(),
                five_bad_drafts(),
                five_parent_eligible_drafts(),
            ],
        )
        worldquant = FakeWorldQuant(
            results=[bad_result()] * 15 + [parent_eligible_result()] * 5
        )
        engine = build_engine(llm=llm, worldquant=worldquant)

        engine.run_until_iteration_boundary()

        self.assertEqual(store.get_idea(idea_one_id)["status"], "EXHAUSTED")
        self.assertEqual(worldquant.simulation_count, 20)
        self.assertEqual(llm.root_batch_count, 4)
```

`run_until_iteration_boundary()` is a test seam that completes one idea or
reaches parent selection; the production `run()` repeatedly calls it.

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_engine_roots -v
```

Expected: import error.

- [ ] **Step 3: Implement persisted root state machine**

```python
class ResearchEngine:
    def run_until_iteration_boundary(self):
        idea = self._get_or_create_idea()
        context = self.selector.select_context(idea)
        candidates = []
        for batch_number in range(1, self.config.max_batches_per_idea + 1):
            if self.control.stop_requested():
                return EngineOutcome("STOPPED")
            drafts = self._generate_root_batch(
                idea, context, batch_number
            )
            for index, draft in enumerate(drafts, 1):
                if self.control.stop_requested():
                    return EngineOutcome("STOPPED")
                outcome = self._validate_simulate_and_record(
                    idea, draft, batch_number, index
                )
                if outcome.qualification.parent_eligible:
                    candidates.append(outcome)
            parents = self._select_parents(candidates)
            if parents:
                return EngineOutcome("PARENTS_READY", idea.id, parents)
        self.store.mark_idea_exhausted(idea.id, "NO_PARENT_AFTER_MAX_BATCHES")
        return EngineOutcome("IDEA_EXHAUSTED", idea.id)
```

Persist the idea, hypothesis, LLM request, validation result, Alpha, simulation,
and qualification before moving to the next Alpha.

After recording qualification, write one compact `research_lessons` row:

- qualified: expression fingerprint plus the metrics that passed;
- parent eligible: fingerprint plus the metrics closest to threshold;
- rejected: validation/error code plus field/operator IDs.

`build_idea_prompt` and `build_root_alpha_prompt` receive the newest bounded
lesson list, so DeepSeek can avoid repeating failed structures without reading
the full Research DB.

Select parents by:

1. qualified first;
2. descending minimum of Sharpe ratio and Fitness ratio;
3. lower turnover;
4. creation order.

Limit to `config.max_parents`.

- [ ] **Step 4: Add invalid-draft token-saving test**

Prove invalid, exact-duplicate, and near-duplicate drafts are recorded but
never sent to `WorldQuantClient.simulate_alpha`.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_engine_roots -v
git add research_engine.py tests/test_research_engine_roots.py
git commit -m "feat: orchestrate root alpha batches and idea rotation"
```

### Task 5: Targeted Variants And No Second Generation

**Files:**
- Modify: `research_engine.py`
- Create: `tests/test_research_engine_variants.py`

- [ ] **Step 1: Write failing variant-direction and generation tests**

```python
class ResearchEngineVariantTest(unittest.TestCase):
    def test_creates_five_targeted_variants_per_parent_only_once(self):
        parents = [parent_one, parent_two]
        engine = build_engine(parents=parents)

        engine._run_variants(idea, context, parents)

        self.assertEqual(llm.variant_request_count, 10)
        directions = [call.direction for call in llm.variant_calls]
        self.assertEqual(
            directions[:5],
            [
                "REDUCE_TURNOVER",
                "IMPROVE_NEUTRALIZATION",
                "ADJUST_TRADE_WHEN",
                "CHANGE_TIME_WINDOW",
                "HANDLE_OUTLIER_OR_SMOOTHING",
            ],
        )
        self.assertTrue(all(alpha.generation == 1 for alpha in store.list_alphas()))
        self.assertFalse(any(alpha.parent_id in child_ids
                             for alpha in store.list_alphas()))
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_engine_variants -v
```

Expected: missing `_run_variants`.

- [ ] **Step 3: Implement one variant per direction per parent**

```python
IMPROVEMENT_DIRECTIONS = (
    "REDUCE_TURNOVER",
    "IMPROVE_NEUTRALIZATION",
    "ADJUST_TRADE_WHEN",
    "CHANGE_TIME_WINDOW",
    "HANDLE_OUTLIER_OR_SMOOTHING",
)


def _run_variants(self, idea, context, parents):
    directions = IMPROVEMENT_DIRECTIONS[:self.config.variants_per_parent]
    for parent in parents:
        for direction in directions:
            if self.control.stop_requested():
                return
            if self.store.count_qualified_for_run(self.run_id) >= (
                self.config.target_qualified_per_run
            ):
                return
            draft = self._generate_variant(parent, direction, context)
            if draft.parent_id != parent.alpha_id or draft.generation != 1:
                draft = replace(
                    draft,
                    parent_id=parent.alpha_id,
                    generation=1,
                    improvement_direction=direction,
                )
            self._validate_simulate_and_record(
                idea, draft, batch_number=None, alpha_index=None
            )
```

If `variants_per_parent` exceeds five, reject config because only five approved
directions exist.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_engine_variants tests.test_research_config -v
git add research_engine.py research_config.py tests/test_research_engine_variants.py tests/test_research_config.py
git commit -m "feat: generate one targeted variant generation"
```

### Task 6: Review Queue, Target Stop, And Graceful Quit

**Files:**
- Modify: `research_engine.py`
- Create: `tests/test_research_engine_stop.py`

- [ ] **Step 1: Write failing target and quit tests**

```python
class ResearchEngineStopTest(unittest.TestCase):
    def test_stops_after_ten_new_qualified_alphas(self):
        engine = build_engine(
            worldquant=FakeWorldQuant(always_qualified=True),
            target=10,
        )

        outcome = engine.run()

        self.assertEqual(outcome.status, "TARGET_REACHED")
        self.assertEqual(store.count_qualified_for_run(outcome.run_id), 10)
        self.assertEqual(len(store.list_pending_review()), 10)

    def test_quit_finishes_current_simulation_but_starts_no_new_work(self):
        control = TriggerAfterCurrentSimulation()
        engine = build_engine(control=control)

        outcome = engine.run()

        self.assertEqual(outcome.status, "STOPPED_BY_USER")
        self.assertEqual(worldquant.simulation_count, 1)
        self.assertEqual(llm.requests_after_stop, 0)
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_engine_stop -v
```

Expected: target/stop assertions fail.

- [ ] **Step 3: Implement run-level target accounting**

```python
def run(self):
    run_id = self.store.start_run(
        self.snapshot_id,
        asdict(self.config),
    )
    self.run_id = run_id
    try:
        while not self.control.stop_requested():
            if self.store.count_qualified_for_run(run_id) >= (
                self.config.target_qualified_per_run
            ):
                self.store.finish_run(run_id, "TARGET_REACHED")
                return RunOutcome(run_id, "TARGET_REACHED")
            boundary = self.run_until_iteration_boundary()
            if boundary.status == "PARENTS_READY":
                self._run_variants(
                    boundary.idea,
                    boundary.context,
                    boundary.parents,
                )
        self.store.finish_run(run_id, "STOPPED_BY_USER")
        return RunOutcome(run_id, "STOPPED_BY_USER")
    except Exception:
        self.store.finish_run(run_id, "FAILED")
        raise
```

After every qualification, enqueue only newly qualified Alpha and immediately
re-check the target before starting another external request.

- [ ] **Step 4: Run all engine tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_engine_roots tests.test_research_engine_variants tests.test_research_engine_stop -v
git add research_engine.py tests/test_research_engine_stop.py
git commit -m "feat: stop research on quit or qualified target"
```

### Task 7: Phase 3 Integration Test

**Files:**
- Create: `tests/test_research_pipeline_integration.py`

- [ ] **Step 1: Write a fake-service end-to-end test**

Build a temporary metadata snapshot and Research DB, then use fake DeepSeek and
fake WorldQuant adapters. Assert:

- one idea and five root hypotheses are persisted;
- invalid expressions never reach simulation;
- qualified Alpha enters `PENDING_REVIEW`;
- parent variants have generation `1` and valid `parent_id`;
- token usage and event logs exist;
- no submit method is called.

- [ ] **Step 2: Run integration and global verification**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_pipeline_integration -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Expected: all commands exit `0`.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_research_pipeline_integration.py
git commit -m "test: cover autonomous research pipeline end to end"
```
