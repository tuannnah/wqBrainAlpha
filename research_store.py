"""Research DB: lịch sử ý tưởng, audit LLM, lineage Alpha và hàng chờ duyệt."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SENSITIVE_KEYS = {"authorization", "api_key", "apikey", "password", "secret"}
DEFAULT_MAX_CHARS = 200_000


def _now():
    return datetime.now(timezone.utc).isoformat()


def redact(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def serialize_for_storage(value, max_chars=DEFAULT_MAX_CHARS):
    rendered = json.dumps(redact(value), ensure_ascii=False, sort_keys=True)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars] + "\n[TRUNCATED]"


def _prompt_hash(prompt):
    canonical = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResearchStore:
    SCHEMA = """
    PRAGMA foreign_keys = ON;
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
    CREATE UNIQUE INDEX alpha_expression_hash_unique ON alphas(expression_hash);
    CREATE UNIQUE INDEX review_alpha_unique ON review_queue(alpha_id);
    """

    def __init__(self, connection, max_chars=DEFAULT_MAX_CHARS):
        self.connection = connection
        self.max_chars = max_chars

    @classmethod
    def _connect(cls, path):
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @classmethod
    def create(cls, path, max_chars=DEFAULT_MAX_CHARS):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = cls._connect(path)
        connection.executescript(cls.SCHEMA)
        connection.commit()
        return cls(connection, max_chars)

    @classmethod
    def open(cls, path, max_chars=DEFAULT_MAX_CHARS):
        return cls(cls._connect(path), max_chars)

    def close(self):
        self.connection.close()

    # -- Runs --------------------------------------------------------------

    def start_run(self, snapshot_id, config):
        cursor = self.connection.execute(
            "INSERT INTO research_runs(snapshot_id, config_json, status, started_at)"
            " VALUES(?, ?, 'RUNNING', ?)",
            (snapshot_id, json.dumps(config, ensure_ascii=False), _now()),
        )
        self.connection.commit()
        return cursor.lastrowid

    def finish_run(self, run_id, status):
        self.connection.execute(
            "UPDATE research_runs SET status=?, finished_at=? WHERE id=?",
            (status, _now(), run_id),
        )
        self.connection.commit()

    # -- Ideas -------------------------------------------------------------

    def create_idea(self, run_id, content, source, novelty_key=None):
        cursor = self.connection.execute(
            "INSERT INTO ideas(run_id, content, source, novelty_key, created_at)"
            " VALUES(?, ?, ?, ?, ?)",
            (run_id, content, source, novelty_key, _now()),
        )
        self.connection.commit()
        return cursor.lastrowid

    def mark_idea_exhausted(self, idea_id, reason):
        self.connection.execute(
            "UPDATE ideas SET status='EXHAUSTED', terminal_reason=? WHERE id=?",
            (reason, idea_id),
        )
        self.connection.commit()

    def next_pending_idea(self, run_id):
        row = self.connection.execute(
            "SELECT * FROM ideas WHERE run_id=? AND status='ACTIVE'"
            " ORDER BY id LIMIT 1",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None

    def create_hypothesis(self, idea_id, hypothesis, rationale, dataset_ids,
                          field_keywords):
        cursor = self.connection.execute(
            "INSERT INTO hypotheses"
            "(idea_id, hypothesis, rationale, dataset_ids_json, field_keywords_json, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?)",
            (
                idea_id,
                hypothesis,
                rationale,
                json.dumps(dataset_ids, ensure_ascii=False),
                json.dumps(field_keywords, ensure_ascii=False),
                _now(),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    # -- LLM audit ---------------------------------------------------------

    def record_llm_request(self, run_id, request_type, model, prompt, response,
                           usage, status="SUCCESS", error_message=None):
        usage = usage or {}
        cursor = self.connection.execute(
            "INSERT INTO llm_requests"
            "(run_id, request_type, model, prompt_hash, prompt_json, response_json,"
            " prompt_tokens, completion_tokens, cache_hit_tokens, status,"
            " error_message, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                request_type,
                model,
                _prompt_hash(prompt),
                serialize_for_storage(prompt, self.max_chars),
                serialize_for_storage(response, self.max_chars) if response is not None else None,
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                int(usage.get("prompt_cache_hit_tokens", usage.get("cache_hit_tokens", 0))),
                status,
                error_message,
                _now(),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    # -- Alphas ------------------------------------------------------------

    def create_alpha(self, run_id, hypothesis_id, expression, expression_hash,
                     structural_fingerprint, settings, dataset_ids, parent_id,
                     generation, improvement_direction):
        cursor = self.connection.execute(
            "INSERT INTO alphas"
            "(run_id, hypothesis_id, expression, expression_hash,"
            " structural_fingerprint, settings_json, dataset_ids_json,"
            " parent_id, generation, improvement_direction, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                hypothesis_id,
                expression,
                expression_hash,
                structural_fingerprint,
                json.dumps(settings, ensure_ascii=False),
                json.dumps(dataset_ids, ensure_ascii=False),
                parent_id,
                generation,
                improvement_direction,
                _now(),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_alpha(self, alpha_id):
        row = self.connection.execute(
            "SELECT * FROM alphas WHERE id=?", (alpha_id,)
        ).fetchone()
        return dict(row) if row else None

    def set_alpha_validation(self, alpha_id, status):
        self.connection.execute(
            "UPDATE alphas SET validation_status=? WHERE id=?", (status, alpha_id)
        )
        self.connection.commit()

    def find_expression_hashes(self):
        rows = self.connection.execute(
            "SELECT expression_hash FROM alphas"
        ).fetchall()
        return {row["expression_hash"] for row in rows}

    def find_fingerprints(self):
        rows = self.connection.execute(
            "SELECT structural_fingerprint FROM alphas"
        ).fetchall()
        return [row["structural_fingerprint"] for row in rows]

    def dataset_usage_counts(self):
        """Đếm số lần mỗi dataset xuất hiện trong các Alpha đã tạo."""
        counts = {}
        for row in self.connection.execute("SELECT dataset_ids_json FROM alphas"):
            for dataset_id in json.loads(row["dataset_ids_json"]):
                counts[dataset_id] = counts.get(dataset_id, 0) + 1
        return counts

    # -- Simulations & qualification --------------------------------------

    def record_simulation(self, alpha_id, result):
        cursor = self.connection.execute(
            "INSERT INTO simulations"
            "(alpha_id, worldquant_alpha_id, status, metrics_json, checks_json,"
            " error_code, error_message, raw_response_json, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                alpha_id,
                result.worldquant_alpha_id,
                result.status,
                json.dumps(result.metrics, ensure_ascii=False),
                json.dumps(result.checks, ensure_ascii=False),
                result.error_code,
                result.error_message,
                serialize_for_storage(result.raw_response, self.max_chars),
                _now(),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def record_qualification(self, simulation_id, qualified, parent_eligible, reasons):
        cursor = self.connection.execute(
            "INSERT INTO qualification_results"
            "(simulation_id, qualified, parent_eligible, reasons_json, created_at)"
            " VALUES(?, ?, ?, ?, ?)",
            (
                simulation_id,
                1 if qualified else 0,
                1 if parent_eligible else 0,
                json.dumps(reasons, ensure_ascii=False),
                _now(),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    # -- Review queue ------------------------------------------------------

    def enqueue_review(self, alpha_id, worldquant_alpha_id):
        cursor = self.connection.execute(
            "INSERT INTO review_queue(alpha_id, worldquant_alpha_id, created_at)"
            " VALUES(?, ?, ?)",
            (alpha_id, worldquant_alpha_id, _now()),
        )
        self.connection.commit()
        return cursor.lastrowid

    def count_qualified_for_run(self, run_id):
        row = self.connection.execute(
            "SELECT COUNT(*) AS c FROM review_queue rq"
            " JOIN alphas a ON a.id = rq.alpha_id WHERE a.run_id = ?",
            (run_id,),
        ).fetchone()
        return row["c"]

    def list_pending_review(self):
        rows = self.connection.execute(
            "SELECT * FROM review_queue WHERE status='PENDING_REVIEW' ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    # -- Events & lessons --------------------------------------------------

    def record_event(self, run_id, event_type, message, payload=None):
        self.connection.execute(
            "INSERT INTO run_events(run_id, event_type, message, payload_json, created_at)"
            " VALUES(?, ?, ?, ?, ?)",
            (
                run_id,
                event_type,
                message,
                serialize_for_storage(payload or {}, self.max_chars),
                _now(),
            ),
        )
        self.connection.commit()

    def list_events(self, run_id):
        rows = self.connection.execute(
            "SELECT * FROM run_events WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def add_lesson(self, lesson_type, content, source_alpha_id=None):
        self.connection.execute(
            "INSERT INTO research_lessons(lesson_type, content, source_alpha_id, created_at)"
            " VALUES(?, ?, ?, ?)",
            (lesson_type, content, source_alpha_id, _now()),
        )
        self.connection.commit()

    def list_lessons(self):
        rows = self.connection.execute(
            "SELECT * FROM research_lessons ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]
