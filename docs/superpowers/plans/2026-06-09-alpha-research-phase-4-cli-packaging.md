# Alpha Research Phase 4: CLI And Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded dataset menu with snapshot creation/selection and autonomous research, while preserving interactive login and producing a working Windows executable.

**Architecture:** `main.py` becomes composition-only code. It authenticates, resolves account storage/config, creates or selects a metadata snapshot, starts the research engine, and exposes pending review records. Legacy hard-coded strategy modules leave the runtime path only after integration tests pass.

**Tech Stack:** Python 3.12, `unittest`, PyInstaller, SQLite.

---

### Task 1: Snapshot Menu And Review Queue CLI

**Files:**
- Modify: `main.py`
- Create: `tests/test_main_research_flow.py`

- [ ] **Step 1: Write failing create/select/review tests**

```python
class MainResearchFlowTest(unittest.TestCase):
    def test_create_snapshot_then_start_engine(self):
        input_func = Mock(side_effect=["1", "USA tháng 6"])
        dependencies = fake_dependencies()

        run_application(
            "user@example.com",
            "secret",
            input_func=input_func,
            dependencies=dependencies,
        )

        dependencies.synchronizer.create_snapshot.assert_called_once_with(
            "USA tháng 6"
        )
        dependencies.engine_factory.assert_called_once()
        dependencies.engine.run.assert_called_once()

    def test_selects_only_ready_snapshot_for_same_account(self):
        input_func = Mock(side_effect=["2", "1"])
        dependencies = fake_dependencies(
            ready_snapshots=[snapshot("id-1", "Old DB")]
        )

        run_application(
            "user@example.com",
            "secret",
            input_func=input_func,
            dependencies=dependencies,
        )

        self.assertEqual(
            dependencies.engine_factory.call_args.kwargs["snapshot_id"],
            "id-1",
        )
```

Add a test for menu option `3` printing `PENDING_REVIEW` Alpha expressions and
WorldQuant IDs without starting research.

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_main_research_flow -v
```

Expected: imports/functions do not exist.

- [ ] **Step 3: Implement dependency composition and menu**

Create:

```python
def run_application(
    email,
    password,
    input_func=input,
    dependencies=None,
):
    paths = build_account_paths(email)
    config = load_config(paths.config_path)
    client = WorldQuantClient(email, password)
    store = ResearchStore.create(paths.research_db)

    print("\n1: Tạo Metadata DB mới")
    print("2: Chọn Metadata DB cũ")
    print("3: Xem Alpha chờ duyệt")
    choice = input_func("\nChọn chức năng: ").strip()

    if choice == "3":
        print_pending_review(store)
        return
    snapshot = create_or_select_snapshot(
        choice, paths, client, input_func
    )
    control = RunControl(input_func=input_func)
    engine = build_research_engine(
        client, snapshot, store, paths, config, control
    )
    print("Nhập quit rồi Enter để dừng an toàn.")
    control.start()
    return engine.run()
```

Keep `main()` responsible for mode-independent credential prompting and frozen
executable pause behavior.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_main_research_flow tests.test_interactive_login -v
git add main.py tests/test_main_research_flow.py tests/test_interactive_login.py
git commit -m "feat: add metadata snapshot and research CLI"
```

### Task 2: Remove Hard-Coded Dataset And Strategy Runtime Path

**Files:**
- Modify: `brain_batch_alpha.py`
- Delete: `dataset_config.py`
- Delete: `alpha_strategy.py`
- Modify: `tests/test_vietnamese_localization.py`
- Create: `tests/test_no_hardcoded_research.py`

- [ ] **Step 1: Write failing runtime-path test**

```python
class NoHardcodedResearchTest(unittest.TestCase):
    def test_runtime_does_not_import_legacy_dataset_or_strategy_modules(self):
        runtime_files = [
            "main.py",
            "worldquant_client.py",
            "research_engine.py",
        ]
        for relative in runtime_files:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("dataset_config", text)
            self.assertNotIn("AlphaStrategy", text)

    def test_hardcoded_modules_are_removed(self):
        self.assertFalse((ROOT / "dataset_config.py").exists())
        self.assertFalse((ROOT / "alpha_strategy.py").exists())
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_no_hardcoded_research -v
```

Expected: failure because legacy files still exist.

- [ ] **Step 3: Remove legacy generation after checking callers**

Run:

```powershell
rg -n "dataset_config|AlphaStrategy|simulate_alphas|_generate_alpha_list|_get_datafields_if_none" -g "*.py"
```

Expected before deletion: only legacy module/tests/build references. Remove
legacy generation methods from `BrainBatchAlpha`; retain it only as a
compatibility alias:

```python
class BrainBatchAlpha(WorldQuantClient):
    """Backward-compatible class name for external imports."""
```

Delete `dataset_config.py` and `alpha_strategy.py`, then update localization
tests so deleted files are not expected.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_no_hardcoded_research tests.test_interactive_login -v
git add -A brain_batch_alpha.py dataset_config.py alpha_strategy.py tests/test_no_hardcoded_research.py tests/test_vietnamese_localization.py
git commit -m "refactor: remove hard-coded alpha generation runtime"
```

### Task 3: Packaging And Runtime Data Rules

**Files:**
- Modify: `.gitignore`
- Modify: `requirements.txt`
- Modify: `setup.py`
- Modify: `build.py`
- Modify: `build_windows.py`
- Modify: `create_zipapp.py`
- Modify: `tests/test_windows_only_structure.py`

- [ ] **Step 1: Write failing packaging tests**

```python
class WindowsOnlyStructureTest(unittest.TestCase):
    def test_builds_include_new_runtime_modules_and_config(self):
        required = [
            "worldquant_client.py",
            "metadata_store.py",
            "metadata_sync.py",
            "research_store.py",
            "deepseek_client.py",
            "research_engine.py",
            "research_config.json",
        ]
        for script_name in ("build.py", "build_windows.py", "create_zipapp.py"):
            text = (ROOT / script_name).read_text(encoding="utf-8")
            for name in required:
                self.assertIn(name, text, f"{script_name}: {name}")

    def test_builds_exclude_removed_hardcoded_modules(self):
        for script_name in ("build.py", "build_windows.py", "create_zipapp.py"):
            text = (ROOT / script_name).read_text(encoding="utf-8")
            self.assertNotIn("dataset_config.py", text)
            self.assertNotIn("alpha_strategy.py", text)
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_windows_only_structure -v
```

Expected: packaging assertions fail.

- [ ] **Step 3: Update packaging**

For PyInstaller, list all new modules in `--add-data` only when they are data
files; Python modules are discovered from imports. Add:

```python
'--add-data=research_config.json{0}.'.format(os.pathsep),
'--collect-all=lark',
```

For zipapp, add every new `.py` module to `source_files`, include
`research_config.json`, and write `lark>=1.1.9,<2` to generated requirements.

Remove `dataset_config.py` and `alpha_strategy.py` references. Do not copy
runtime databases or logs into `dist`.

Add to `.gitignore`:

```text
data/
*.sqlite
*.sqlite.tmp
*.failed.sqlite
logs/
```

- [ ] **Step 4: Run packaging tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_windows_only_structure -v
git add .gitignore requirements.txt setup.py build.py build_windows.py create_zipapp.py tests/test_windows_only_structure.py
git commit -m "build: package autonomous research runtime"
```

### Task 4: README And Operational Documentation

**Files:**
- Modify: `README.md`
- Create: `tests/test_research_documentation.py`

- [ ] **Step 1: Write failing documentation assertions**

```python
class ResearchDocumentationTest(unittest.TestCase):
    def test_readme_documents_required_operation(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in (
            "DEEPSEEK_API_KEY",
            "Tạo Metadata DB mới",
            "Chọn Metadata DB cũ",
            "quit",
            "PENDING_REVIEW",
            "không tự submit",
        ):
            self.assertIn(phrase, text)
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_documentation -v
```

Expected: README is missing one or more required sections.

- [ ] **Step 3: Update README with exact Windows commands**

Document environment setup:

```powershell
[Environment]::SetEnvironmentVariable(
    "DEEPSEEK_API_KEY",
    "your-key",
    "User"
)
```

Document:

- metadata snapshot creation and selection;
- snapshot labels and per-email isolation;
- automatic idea/field/operator selection;
- three batches of five root Alpha per idea;
- targeted variants only after the quality gate;
- `quit` graceful stop;
- target of ten new qualified Alpha per run;
- log and DB location under `%LOCALAPPDATA%\WorldQuantBrainAlpha`;
- manual inspection of `PENDING_REVIEW`;
- no automatic submit.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_documentation tests.test_vietnamese_localization -v
git add README.md tests/test_research_documentation.py
git commit -m "docs: explain autonomous alpha research workflow"
```

### Task 5: Full Verification And Windows Build Smoke Test

**Files:**
- Verify: entire repository

- [ ] **Step 1: Run full automated suite**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax, dependency, and whitespace checks**

```powershell
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Expected: all commands exit `0`.

- [ ] **Step 3: Build the Windows executable**

```powershell
.\.venv\Scripts\python.exe build_windows.py
```

Expected: `dist\Alpha_Tool.exe` exists and PyInstaller reports a successful
build.

- [ ] **Step 4: Smoke test without network calls**

Run the executable with `DEEPSEEK_API_KEY` temporarily unset and valid login
dependencies replaced only in a dedicated smoke-test harness. Verify:

- console starts with UTF-8 output;
- config is created in `%LOCALAPPDATA%\WorldQuantBrainAlpha`;
- missing DeepSeek key is reported before a research run starts;
- the process closes cleanly without creating a `READY` snapshot.

Record the smoke-test result in the commit message body or release notes; do
not add credentials or generated databases to Git.

- [ ] **Step 5: Final review**

```powershell
git status --short
git log --oneline -15
```

Confirm:

- no credential, API key, SQLite DB, or log file is tracked;
- every production module has direct tests;
- no runtime import references deleted hard-coded modules;
- manual submit is not triggered by the research engine;
- unrelated pre-existing workspace changes were not reverted.
