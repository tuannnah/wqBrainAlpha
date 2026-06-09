# Alpha Research Phase 2: Generation And Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable research persistence, structured DeepSeek generation, local candidate selection, FASTEXPR parsing, duplicate detection, and qualification policy.

**Architecture:** Keep model calls generic and JSON-only. Select a small context from the metadata snapshot before calling DeepSeek. Parse generated expressions into a tree, validate identifiers/types/settings against the snapshot, and evaluate completed simulations through a standalone policy.

**Tech Stack:** Python 3.12, `requests`, `sqlite3`, FTS5, `lark`, `unittest`.

---

### Task 1: Research Database And Audit Trail

**Files:**
- Create: `research_store.py`
- Create: `tests/test_research_store.py`

- [ ] **Step 1: Write failing lineage, token, and review-queue tests**

```python
class ResearchStoreTest(unittest.TestCase):
    def test_records_alpha_lineage_tokens_and_pending_review(self):
        store = ResearchStore.create(path)
        run_id = store.start_run("snapshot-1", {"target_qualified_per_run": 10})
        idea_id = store.create_idea(run_id, "Price reversal", "DEEPSEEK")
        hypothesis_id = store.create_hypothesis(
            idea_id,
            "Extreme short-term losses revert",
            "Behavioral overreaction",
            ["pv1"],
            ["returns", "close"],
        )
        request_id = store.record_llm_request(
            run_id=run_id,
            request_type="ROOT_ALPHA",
            model="deepseek-v4-pro",
            prompt={"idea": "Price reversal"},
            response={"alphas": []},
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )
        parent_id = store.create_alpha(
            run_id, hypothesis_id, "rank(-returns)", "hash1", "finger1",
            {"region": "USA"}, ["pv1"], None, 0, None,
        )
        child_id = store.create_alpha(
            run_id, hypothesis_id, "rank(ts_mean(-returns, 5))",
            "hash2", "finger2", {"region": "USA"}, ["pv1"],
            parent_id, 1, "CHANGE_TIME_WINDOW",
        )
        store.enqueue_review(child_id, "wq-alpha-1")

        self.assertEqual(store.get_alpha(child_id)["parent_id"], parent_id)
        self.assertEqual(store.count_qualified_for_run(run_id), 1)
        self.assertEqual(store.list_pending_review()[0]["status"], "PENDING_REVIEW")
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_store -v
```

Expected: import error for `research_store`.

- [ ] **Step 3: Implement schema and transactions**

Create the following tables with foreign keys enabled:

```sql
CREATE TABLE research_runs (
    id INTEGER PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    qualified_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE ideas (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES research_runs(id),
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    novelty_key TEXT,
    terminal_reason TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE hypotheses (
    id INTEGER PRIMARY KEY,
    idea_id INTEGER NOT NULL REFERENCES ideas(id),
    hypothesis TEXT NOT NULL,
    rationale TEXT NOT NULL,
    dataset_ids_json TEXT NOT NULL,
    field_keywords_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE llm_requests (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES research_runs(id),
    request_type TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_json TEXT NOT NULL,
    response_json TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE alphas (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES research_runs(id),
    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id),
    expression TEXT NOT NULL,
    expression_hash TEXT NOT NULL,
    structural_fingerprint TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    dataset_ids_json TEXT NOT NULL,
    parent_id INTEGER REFERENCES alphas(id),
    generation INTEGER NOT NULL,
    improvement_direction TEXT,
    validation_status TEXT NOT NULL DEFAULT 'PENDING',
    created_at TEXT NOT NULL,
    CHECK (
        (generation = 0 AND parent_id IS NULL AND improvement_direction IS NULL)
        OR
        (generation = 1 AND parent_id IS NOT NULL AND improvement_direction IS NOT NULL)
    )
);
CREATE TABLE simulations (
    id INTEGER PRIMARY KEY,
    alpha_id INTEGER NOT NULL REFERENCES alphas(id),
    worldquant_alpha_id TEXT,
    status TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    checks_json TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    raw_response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE qualification_results (
    id INTEGER PRIMARY KEY,
    simulation_id INTEGER NOT NULL REFERENCES simulations(id),
    qualified INTEGER NOT NULL,
    parent_eligible INTEGER NOT NULL,
    reasons_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE review_queue (
    id INTEGER PRIMARY KEY,
    alpha_id INTEGER NOT NULL REFERENCES alphas(id),
    worldquant_alpha_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
    created_at TEXT NOT NULL,
    reviewed_at TEXT
);
CREATE TABLE research_lessons (
    id INTEGER PRIMARY KEY,
    lesson_type TEXT NOT NULL,
    content TEXT NOT NULL,
    source_alpha_id INTEGER REFERENCES alphas(id),
    created_at TEXT NOT NULL
);
CREATE TABLE run_events (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES research_runs(id),
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

Store JSON as canonical UTF-8 text. Enforce lineage with:

```sql
CHECK (
    (generation = 0 AND parent_id IS NULL AND improvement_direction IS NULL)
    OR
    (generation = 1 AND parent_id IS NOT NULL AND improvement_direction IS NOT NULL)
)
```

Add unique indexes:

```sql
CREATE UNIQUE INDEX alpha_expression_hash_unique ON alphas(expression_hash);
CREATE UNIQUE INDEX review_alpha_unique ON review_queue(alpha_id);
```

Implement:

- `start_run`, `finish_run`;
- `create_idea`, `mark_idea_exhausted`, `next_pending_idea`;
- `create_hypothesis`;
- `record_llm_request`;
- `create_alpha`, `get_alpha`, `find_expression_hashes`, `find_fingerprints`;
- `record_simulation`, `record_qualification`;
- `enqueue_review`, `count_qualified_for_run`, `list_pending_review`;
- `record_event`, `list_lessons`.

- [ ] **Step 4: Add redaction before raw payload persistence**

```python
SENSITIVE_KEYS = {"authorization", "api_key", "apikey", "password", "secret"}


def redact(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def serialize_for_storage(value, max_chars):
    rendered = json.dumps(
        redact(value),
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars] + "\n[TRUNCATED]"
```

Add a test proving `DEEPSEEK_API_KEY`, passwords, and authorization headers do
not appear in the SQLite file bytes. Use
`config.raw_response_max_chars` for LLM and WorldQuant raw payloads.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_store -v
git add research_store.py tests/test_research_store.py
git commit -m "feat: add research history and review queue store"
```

### Task 2: Structured DeepSeek Client

**Files:**
- Create: `deepseek_client.py`
- Create: `tests/test_deepseek_client.py`

- [ ] **Step 1: Write failing API-key, JSON, retry, and usage tests**

```python
class DeepSeekClientTest(unittest.TestCase):
    def test_reads_environment_key_and_returns_usage(self):
        session = Mock()
        session.post.return_value = fake_response(200, {
            "choices": [{"message": {"content": '{"alphas": []}'}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "prompt_cache_hit_tokens": 80,
            },
        })
        client = DeepSeekClient(
            config,
            session=session,
            environ={"DEEPSEEK_API_KEY": "secret-key"},
            sleep_func=Mock(),
        )

        result = client.generate_json(
            "ROOT_ALPHA",
            "Return JSON with an alphas key.",
            {"idea": "reversal"},
        )

        self.assertEqual(result.data, {"alphas": []})
        self.assertEqual(result.usage.prompt_tokens, 120)
        self.assertEqual(result.usage.cache_hit_tokens, 80)
        headers = session.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer secret-key")
```

Also test:

- missing `DEEPSEEK_API_KEY` raises `DeepSeekConfigurationError`;
- payload includes `response_format={"type": "json_object"}`;
- empty content and invalid JSON retry at most configured attempts;
- HTTP 429 respects `Retry-After`;
- error text never includes the API key.

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deepseek_client -v
```

Expected: import error.

- [ ] **Step 3: Implement generic JSON client using `requests`**

```python
@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0


@dataclass(frozen=True)
class DeepSeekResult:
    request_type: str
    model: str
    data: dict
    usage: TokenUsage
    raw_response: dict


class DeepSeekClient:
    def generate_json(self, request_type, system_prompt, user_payload):
        payload = {
            "model": self.config.deepseek_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.config.deepseek_max_output_tokens,
            "stream": False,
        }
        return self._post_with_retry(request_type, payload)
```

Use:

```python
response = self.session.post(
    f"{self.config.deepseek_base_url.rstrip('/')}/chat/completions",
    json=payload,
    headers={
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
    },
    timeout=self.config.deepseek_timeout_seconds,
)
```

Parse `choices[0].message.content`, reject empty/non-object JSON, and return
usage without estimating cost.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deepseek_client -v
git add deepseek_client.py tests/test_deepseek_client.py
git commit -m "feat: add structured DeepSeek JSON client"
```

### Task 3: Local Candidate Selection

**Files:**
- Create: `candidate_selector.py`
- Create: `tests/test_candidate_selector.py`
- Modify: `metadata_store.py`

- [ ] **Step 1: Write failing catalog and candidate tests**

```python
class CandidateSelectorTest(unittest.TestCase):
    def test_builds_small_catalog_and_selects_fields_by_keywords(self):
        selector = CandidateSelector(metadata_store, research_store, config)

        catalog = selector.build_dataset_catalog(limit=8)
        context = selector.select_context(
            idea={
                "title": "Earnings quality",
                "field_keywords": ["earnings", "cash flow", "accrual"],
                "dataset_keywords": ["fundamental"],
            }
        )

        self.assertLessEqual(len(catalog), 8)
        self.assertGreaterEqual(len(context.fields), config.candidate_fields_min)
        self.assertLessEqual(len(context.fields), config.candidate_fields_max)
        self.assertTrue(all(field["dataset_id"] in context.dataset_ids
                            for field in context.fields))
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_candidate_selector -v
```

Expected: import error.

- [ ] **Step 3: Add metadata query APIs**

Implement:

```python
MetadataStore.dataset_catalog(limit, excluded_ids)
MetadataStore.search_fields(query, limit, dataset_ids=None)
MetadataStore.operators_for_types(field_types)
MetadataStore.scope_for_dataset(dataset_id)
MetadataStore.field_records(field_ids)
```

Rank datasets by low usage count from `ResearchStore`, then by field count and
description availability. Candidate field search combines escaped FTS tokens
with `OR`, merges results by field ID, and fills the minimum from the selected
datasets when keyword hits are sparse.

- [ ] **Step 4: Implement bounded candidate context**

```python
@dataclass(frozen=True)
class CandidateContext:
    dataset_ids: list
    scope: Scope
    fields: list
    operators: list


class CandidateSelector:
    def select_context(self, idea):
        datasets = self._select_datasets(idea)
        fields = self._search_and_fill_fields(
            idea["field_keywords"],
            datasets,
            self.config.candidate_fields_min,
            self.config.candidate_fields_max,
        )
        field_types = {field["field_type"] for field in fields}
        operators = self.metadata_store.operators_for_types(field_types)
        return CandidateContext(
            dataset_ids=[item["id"] for item in datasets],
            scope=self._common_scope(datasets),
            fields=fields,
            operators=operators,
        )
```

Reject dataset combinations without a common scope.

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_candidate_selector tests.test_metadata_store -v
git add candidate_selector.py metadata_store.py tests/test_candidate_selector.py tests/test_metadata_store.py
git commit -m "feat: select bounded metadata context locally"
```

### Task 4: FASTEXPR Parser And Structural Fingerprints

**Files:**
- Create: `expression_parser.py`
- Create: `tests/test_expression_parser.py`
- Modify: `requirements.txt`
- Modify: `setup.py`

- [ ] **Step 1: Add `lark` dependency**

Add:

```text
lark>=1.1.9,<2
```

to `requirements.txt` and `setup.py`.

- [ ] **Step 2: Write failing parser and fingerprint tests**

```python
class ExpressionParserTest(unittest.TestCase):
    def test_parses_calls_keywords_and_arithmetic(self):
        parsed = parse_expression(
            "group_neutralize(rank(ts_delta(close, 20)), subindustry)"
        )
        self.assertEqual(
            parsed.operator_names,
            {"group_neutralize", "rank", "ts_delta"},
        )
        self.assertEqual(parsed.identifiers, {"close", "subindustry"})

    def test_fingerprint_normalizes_numeric_parameters_and_fields(self):
        first = fingerprint("rank(ts_delta(close, 20))", {"close"})
        second = fingerprint("rank(ts_delta(open, 10))", {"open"})
        self.assertEqual(first, second)
```

- [ ] **Step 3: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_expression_parser -v
```

Expected: import error.

- [ ] **Step 4: Implement the constrained grammar**

Use this exact grammar:

```python
GRAMMAR = r"""
?start: expr
?expr: logical_or
?logical_or: logical_and ("||" logical_and)*
?logical_and: comparison ("&&" comparison)*
?comparison: sum (COMP_OP sum)*
?sum: product (ADD_OP product)*
?product: power (MUL_OP power)*
?power: unary ("^" unary)*
?unary: ("+" | "-" | "!") unary | atom
?atom: function_call | NAME | NUMBER | STRING | "(" expr ")"
function_call: NAME "(" [argument ("," argument)*] ")"
?argument: NAME "=" expr -> keyword_argument
         | expr
COMP_OP: "<=" | ">=" | "==" | "!=" | "<" | ">"
ADD_OP: "+" | "-"
MUL_OP: "*" | "/"
NAME: /[A-Za-z_][A-Za-z0-9_]*/
NUMBER: /\d+(\.\d+)?/
STRING: /'([^'\\]|\\.)*'/ | /"([^"\\]|\\.)*"/
%import common.WS
%ignore WS
"""
```

Return a `ParsedExpression` with ordered tokens, operator names, identifiers,
call tree, normalized expression, exact SHA-256 hash, and structural
fingerprint. Fingerprint replaces known field IDs with `$FIELD` and numeric
literals with `$NUMBER`, while preserving operator names.

- [ ] **Step 5: Install dependency, run tests, and commit**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest tests.test_expression_parser -v
git add expression_parser.py tests/test_expression_parser.py requirements.txt setup.py
git commit -m "feat: parse and fingerprint generated alpha expressions"
```

### Task 5: Metadata-Aware Expression Validation

**Files:**
- Create: `expression_validator.py`
- Create: `tests/test_expression_validator.py`

- [ ] **Step 1: Write failing identifier, type, scope, and similarity tests**

```python
class ExpressionValidatorTest(unittest.TestCase):
    def test_rejects_unknown_field_and_vector_without_reducer(self):
        validator = ExpressionValidator(metadata_store, research_store, config)

        unknown = validator.validate(
            draft("rank(missing_field)"),
            candidate_context,
        )
        vector = validator.validate(
            draft("rank(news_vector)"),
            candidate_context,
        )
        reduced = validator.validate(
            draft("rank(vec_avg(news_vector))"),
            candidate_context,
        )

        self.assertIn("UNKNOWN_FIELD", unknown.error_codes)
        self.assertIn("VECTOR_REDUCER_REQUIRED", vector.error_codes)
        self.assertTrue(reduced.is_valid)

    def test_rejects_exact_and_near_duplicates(self):
        research_store.create_alpha(
            run_id=1,
            hypothesis_id=1,
            expression=existing_expression,
            expression_hash=known_hash,
            structural_fingerprint=known_fingerprint,
            settings={"region": "USA"},
            dataset_ids=["pv1"],
            parent_id=None,
            generation=0,
            improvement_direction=None,
        )
        self.assertIn(
            "DUPLICATE_EXPRESSION",
            validator.validate(draft(existing_expression), context).error_codes,
        )
```

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_expression_validator -v
```

Expected: import error.

- [ ] **Step 3: Implement deterministic validation order**

```python
class ExpressionValidator:
    def validate(self, draft, context):
        errors = []
        try:
            parsed = parse_expression(draft.expression)
        except UnexpectedInput as exc:
            return ValidationResult.invalid("SYNTAX_ERROR", str(exc))
        errors.extend(self._validate_operators(parsed, context))
        errors.extend(self._validate_identifiers(parsed, context))
        errors.extend(self._validate_types(parsed, context))
        errors.extend(self._validate_scope_and_settings(draft, context))
        errors.extend(self._validate_duplicates(parsed))
        return ValidationResult(
            is_valid=not errors,
            error_codes=[item.code for item in errors],
            errors=errors,
            expression_hash=parsed.expression_hash,
            fingerprint=parsed.fingerprint,
            normalized_expression=parsed.normalized_expression,
        )
```

Rules:

- every called function must be in candidate operators;
- identifiers must be candidate fields, known group identifiers, or keyword
  argument names;
- a `VECTOR` field must have an ancestor operator whose name starts `vec_`;
- a `GROUP` field can only appear in the group argument of operators whose
  name starts `group_`;
- settings must exactly match the selected common scope and allowed
  neutralization/configuration values;
- exact hash duplicates are always rejected;
- near duplicates use Jaccard similarity over fingerprint token 3-grams and
  the configured threshold.

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_expression_validator tests.test_expression_parser -v
git add expression_validator.py tests/test_expression_validator.py
git commit -m "feat: validate generated alphas against metadata"
```

### Task 6: Qualification And Parent Quality Gate

**Files:**
- Create: `qualification.py`
- Create: `tests/test_qualification.py`

- [ ] **Step 1: Write failing pass, fail, and parent-selection tests**

```python
class QualificationPolicyTest(unittest.TestCase):
    def test_qualifies_brain_checks_and_selects_near_threshold_parent(self):
        policy = QualificationPolicy(
            sharpe_threshold=1.5,
            fitness_threshold=1.0,
            turnover_min=0.01,
            turnover_hard_limit=0.9,
            quality_gate_ratio=0.8,
        )
        completed = SimulationResult(
            worldquant_alpha_id="a1",
            status="COMPLETED",
            metrics={"sharpe": 1.25, "fitness": 0.85, "turnover": 0.4, "margin": 0.03},
            checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"}],
        )

        result = policy.evaluate(completed)

        self.assertFalse(result.qualified)
        self.assertTrue(result.parent_eligible)
```

Add cases for syntax errors, dataset authorization errors, turnover above hard
limit, failed/pending BRAIN checks, and a fully qualified Alpha.

- [ ] **Step 2: Run test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_qualification -v
```

Expected: import error.

- [ ] **Step 3: Implement explicit reasons**

```python
@dataclass(frozen=True)
class QualificationResult:
    qualified: bool
    parent_eligible: bool
    reasons: list
    sharpe_ratio: float
    fitness_ratio: float


class QualificationPolicy:
    def evaluate(self, simulation):
        if simulation.status != "COMPLETED":
            return QualificationResult(False, False, [simulation.status], 0, 0)
        sharpe = float(simulation.metrics.get("sharpe", 0))
        fitness = float(simulation.metrics.get("fitness", 0))
        turnover = float(simulation.metrics.get("turnover", 0))
        failed_checks = [
            item["name"] for item in simulation.checks
            if item.get("result") != "PASS"
        ]
        qualified = (
            sharpe >= self.sharpe_threshold
            and fitness >= self.fitness_threshold
            and self.turnover_min <= turnover <= self.turnover_hard_limit
            and not failed_checks
        )
        parent_eligible = (
            not failed_checks
            and turnover <= self.turnover_hard_limit
            and sharpe / self.sharpe_threshold >= self.quality_gate_ratio
            and fitness / self.fitness_threshold >= self.quality_gate_ratio
        )
        return QualificationResult(
            qualified,
            qualified or parent_eligible,
            self._reasons(sharpe, fitness, turnover, failed_checks),
            sharpe / self.sharpe_threshold,
            fitness / self.fitness_threshold,
        )
```

- [ ] **Step 4: Run tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_qualification -v
git add qualification.py tests/test_qualification.py
git commit -m "feat: evaluate alpha qualification and parent eligibility"
```

### Task 7: Phase 2 Verification

- [ ] **Step 1: Run focused suite**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_research_store tests.test_deepseek_client tests.test_candidate_selector tests.test_expression_parser tests.test_expression_validator tests.test_qualification -v
```

Expected: `OK`.

- [ ] **Step 2: Run global verification**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Expected: all commands exit `0`.
