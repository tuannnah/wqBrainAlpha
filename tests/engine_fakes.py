"""Fakes và factory dùng chung cho các test của research_engine."""

from candidate_selector import CandidateContext
from expression_validator import ValidationResult
from qualification import QualificationPolicy
from research_config import ResearchConfig
from research_engine import ResearchEngine
from research_models import AlphaDraft, Scope, SimulationResult
from research_store import ResearchStore


def context():
    return CandidateContext(
        dataset_ids=["pv1"],
        scope=Scope("EQUITY", "USA", 1, "TOP3000"),
        fields=[
            {"id": "close", "dataset_id": "pv1", "field_type": "MATRIX"},
            {"id": "returns", "dataset_id": "pv1", "field_type": "MATRIX"},
        ],
        operators=[{"name": "rank"}, {"name": "ts_delta"}],
    )


def settings():
    return {
        "instrumentType": "EQUITY",
        "region": "USA",
        "delay": 1,
        "universe": "TOP3000",
        "neutralization": "SUBINDUSTRY",
    }


def make_drafts(prefix, count=5):
    return [
        AlphaDraft(
            hypothesis=f"{prefix}-h{index}",
            rationale="r",
            expression=f"{prefix}_expr_{index}",
            dataset_ids=["pv1"],
            field_ids=["close"],
            operator_names=["rank"],
            settings=settings(),
            generation=0,
        )
        for index in range(count)
    ]


def bad_result():
    return SimulationResult(
        worldquant_alpha_id="wq",
        status="COMPLETED",
        metrics={"sharpe": 0.3, "fitness": 0.2, "turnover": 0.4},
        checks=[{"name": "X", "result": "PASS"}],
    )


def parent_eligible_result():
    return SimulationResult(
        worldquant_alpha_id="wq",
        status="COMPLETED",
        metrics={"sharpe": 1.3, "fitness": 0.9, "turnover": 0.4},
        checks=[{"name": "X", "result": "PASS"}],
    )


def qualified_result():
    return SimulationResult(
        worldquant_alpha_id="wq",
        status="COMPLETED",
        metrics={"sharpe": 1.7, "fitness": 1.2, "turnover": 0.4},
        checks=[{"name": "X", "result": "PASS"}],
    )


class FakeLlm:
    def __init__(self, ideas=None, root_batches=None, variant_draft=None):
        self._ideas = list(ideas or [{"title": "Idea", "field_keywords": ["close"]}])
        self._root_batches = list(root_batches or [])
        self._variant_draft = variant_draft
        self.idea_count = 0
        self.root_batch_count = 0
        self.variant_request_count = 0
        self.variant_calls = []
        self.requests_after_stop = 0
        self.control = None

    def _check_after_stop(self):
        if self.control is not None and self.control.stop_requested():
            self.requests_after_stop += 1

    def create_idea(self, catalog, lessons):
        self._check_after_stop()
        idea = self._ideas[min(self.idea_count, len(self._ideas) - 1)]
        self.idea_count += 1
        return dict(idea), {"prompt_tokens": 1, "completion_tokens": 1}

    def generate_root_alphas(self, idea, context, count, lessons):
        self._check_after_stop()
        batch = self._root_batches[self.root_batch_count]
        self.root_batch_count += 1
        return list(batch), {"prompt_tokens": 1, "completion_tokens": 1}

    def generate_variant(self, parent, direction, context):
        self._check_after_stop()
        self.variant_request_count += 1
        self.variant_calls.append(_VariantCall(parent, direction))
        base = self._variant_draft or AlphaDraft(
            hypothesis="variant",
            rationale="r",
            expression=f"variant_{self.variant_request_count}",
            dataset_ids=["pv1"],
            field_ids=["close"],
            operator_names=["rank"],
            settings=settings(),
            generation=1,
            parent_id=_parent_id(parent),
            improvement_direction=direction,
        )
        return base, {"prompt_tokens": 1, "completion_tokens": 1}


class _VariantCall:
    def __init__(self, parent, direction):
        self.parent = parent
        self.direction = direction


def _parent_id(parent):
    return getattr(parent, "alpha_id", None)


class FakeWorldQuant:
    def __init__(self, results=None, always_qualified=False):
        self._results = list(results or [])
        self.always_qualified = always_qualified
        self.simulation_count = 0
        self.submit_calls = 0

    def simulate_alpha(self, payload):
        self.simulation_count += 1
        if self.always_qualified:
            return qualified_result()
        result = self._results[self.simulation_count - 1]
        return result

    def submit_alpha(self, alpha_id):  # pragma: no cover - phải không bao giờ gọi
        self.submit_calls += 1
        raise AssertionError("Engine không được tự submit Alpha")


class FakeValidator:
    def __init__(self, invalid_expressions=None):
        self.invalid = set(invalid_expressions or [])
        self._counter = 0

    def validate(self, draft, context):
        self._counter += 1
        if draft.expression in self.invalid:
            return ValidationResult(False, ["UNKNOWN_FIELD"], [], None, None, None)
        marker = self._counter
        return ValidationResult(
            True, [], [],
            expression_hash=f"hash-{marker}",
            fingerprint=f"fp-{marker}",
            normalized_expression=draft.expression,
        )


class FakeSelector:
    def __init__(self, ctx):
        self.ctx = ctx

    def build_dataset_catalog(self, limit):
        return [{"id": "pv1", "name": "PV", "field_count": 10}]

    def select_context(self, idea):
        return self.ctx


class FakeControl:
    def __init__(self, stop=False):
        self._stop = stop

    def stop_requested(self):
        return self._stop


def build_engine(temp_path, llm, worldquant, validator=None, control=None,
                 config=None, target=None):
    store = ResearchStore.create(temp_path)
    config = config or ResearchConfig()
    if target is not None:
        config = ResearchConfig(**{**config.__dict__, "target_qualified_per_run": target})
    validator = validator or FakeValidator()
    control = control or FakeControl()
    if isinstance(llm, FakeLlm):
        llm.control = control
    policy = QualificationPolicy(
        config.sharpe_threshold,
        config.fitness_threshold,
        config.turnover_min,
        config.turnover_hard_limit,
        config.quality_gate_ratio,
    )
    engine = ResearchEngine(
        snapshot_id="snap",
        store=store,
        selector=FakeSelector(context()),
        llm=llm,
        validator=validator,
        worldquant=worldquant,
        policy=policy,
        control=control,
        config=config,
    )
    engine.run_id = store.start_run("snap", {})
    return engine, store
