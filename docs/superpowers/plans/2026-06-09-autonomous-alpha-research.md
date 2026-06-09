# Autonomous Alpha Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hard-coded WorldQuant datasets and Alpha templates with an autonomous, auditable DeepSeek-assisted research pipeline.

**Architecture:** Deliver the feature in four sequential phases. Each phase ends with a runnable, tested system boundary: local persistence and metadata synchronization; generation and validation; autonomous research orchestration; then CLI migration and packaging.

**Tech Stack:** Python 3.12, `requests`, `sqlite3` with FTS5, `lark`, `unittest`, PyInstaller.

---

## Execution Order

1. [Phase 1: Storage and WorldQuant Metadata](2026-06-09-alpha-research-phase-1-storage-metadata.md)
2. [Phase 2: DeepSeek Generation and Expression Validation](2026-06-09-alpha-research-phase-2-generation-validation.md)
3. [Phase 3: Research Engine, Logging, and Stop Control](2026-06-09-alpha-research-phase-3-engine-control.md)
4. [Phase 4: CLI Migration, Packaging, and Documentation](2026-06-09-alpha-research-phase-4-cli-packaging.md)

Each phase must be completed and its full verification suite must pass before
starting the next phase. Do not remove `BrainBatchAlpha`, `AlphaStrategy`, or
`dataset_config.py` until Phase 4 has moved all runtime callers.

## File Map

### New production modules

- `research_models.py`: immutable data transfer objects shared across modules.
- `research_config.py`: JSON configuration loading and validation.
- `account_storage.py`: account hashing and Windows user-data paths.
- `metadata_store.py`: metadata snapshot SQLite schema and queries.
- `metadata_sync.py`: full-account WorldQuant metadata synchronization.
- `worldquant_client.py`: authentication, metadata endpoints, and simulation.
- `research_store.py`: research history, LLM audit, Alpha lineage, and review queue.
- `deepseek_client.py`: structured DeepSeek requests, retries, token usage, and redaction.
- `candidate_selector.py`: local dataset, field, and operator selection.
- `expression_parser.py`: FASTEXPR subset parser and structural fingerprinting.
- `expression_validator.py`: metadata, scope, type, duplicate, and similarity checks.
- `qualification.py`: pass/fail policy and parent quality gate.
- `logging_setup.py`: console, rotating file, and structured DB events.
- `run_control.py`: background `quit` listener and cooperative stop flag.
- `research_engine.py`: idea, root batch, parent, variant, and target state machine.

### New configuration

- `research_config.json`: editable non-secret defaults.

### Existing modules changed at integration

- `main.py`: login, create/select snapshot, start engine, inspect review queue.
- `brain_batch_alpha.py`: compatibility adapter during migration.
- `requirements.txt`, `setup.py`: add `lark`.
- `build.py`, `build_windows.py`, `create_zipapp.py`: package new modules and config.
- `.gitignore`: ignore runtime `data/` and logs.
- `README.md`: document DeepSeek key, snapshots, `quit`, logs, and review queue.

## Global Verification

After every phase:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Expected: all tests report `OK`, compile and dependency checks exit `0`, and
`git diff --check` prints no output.
