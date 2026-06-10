"""State machine điều phối nghiên cứu Alpha tự động."""

import json
from dataclasses import dataclass, replace
from typing import List, Optional

from alpha_prompts import (
    build_idea_prompt,
    build_root_alpha_prompt,
    build_variant_prompt,
    map_root_response,
    map_variant_response,
)


CATALOG_LIMIT = 20
LESSON_WINDOW = 20
IMPROVEMENT_DIRECTIONS = (
    "REDUCE_TURNOVER",
    "IMPROVE_NEUTRALIZATION",
    "ADJUST_TRADE_WHEN",
    "CHANGE_TIME_WINDOW",
    "HANDLE_OUTLIER_OR_SMOOTHING",
)


@dataclass
class ParentCandidate:
    alpha_id: int
    expression: str
    hypothesis: str
    metrics: dict
    sharpe_ratio: float
    fitness_ratio: float
    turnover: float
    qualified: bool
    order: int


@dataclass
class RecordOutcome:
    alpha_id: Optional[int]
    parent: Optional[ParentCandidate]
    simulated: bool


@dataclass
class EngineOutcome:
    status: str
    idea: Optional[dict] = None
    context: object = None
    parents: List[ParentCandidate] = None


@dataclass
class RunOutcome:
    run_id: int
    status: str


def _usage_dict(usage):
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "cache_hit_tokens": getattr(usage, "cache_hit_tokens", 0),
    }


class DeepSeekAlphaGenerator:
    """Ghép alpha_prompts với DeepSeekClient để sinh idea/root/variant."""

    def __init__(self, client):
        self.client = client

    def create_idea(self, catalog, lessons):
        system, payload = build_idea_prompt(catalog, lessons)
        result = self.client.generate_json("IDEA", system, payload)
        return result.data, _usage_dict(result.usage)

    def generate_root_alphas(self, idea, context, count, lessons):
        system, payload = build_root_alpha_prompt(idea, context, count, lessons)
        result = self.client.generate_json("ROOT_ALPHA", system, payload)
        return map_root_response(result.data, context), _usage_dict(result.usage)

    def generate_variant(self, parent, direction, context):
        system, payload = build_variant_prompt(parent, direction, context)
        result = self.client.generate_json("VARIANT", system, payload)
        parent_id = getattr(parent, "alpha_id", None)
        return (
            map_variant_response(result.data, parent_id, direction),
            _usage_dict(result.usage),
        )


class ResearchEngine:
    def __init__(self, snapshot_id, store, selector, llm, validator, worldquant,
                 policy, control, config, logger=None):
        self.snapshot_id = snapshot_id
        self.store = store
        self.selector = selector
        self.llm = llm
        self.validator = validator
        self.worldquant = worldquant
        self.policy = policy
        self.control = control
        self.config = config
        self.logger = logger
        self.run_id = None

    # -- Logging -----------------------------------------------------------

    def _log(self, event_type, message, payload=None):
        if self.logger is not None:
            self.logger.event(event_type, message, payload or {})

    def _lessons(self):
        lessons = self.store.list_lessons()[-LESSON_WINDOW:]
        return [lesson["content"] for lesson in lessons]

    # -- Idea handling -----------------------------------------------------

    def _get_or_create_idea(self):
        pending = self.store.next_pending_idea(self.run_id)
        if pending:
            idea = json.loads(pending["content"])
            idea["id"] = pending["id"]
            return idea

        catalog = self.selector.build_dataset_catalog(CATALOG_LIMIT)
        idea_dict, usage = self.llm.create_idea(catalog, self._lessons())
        idea_id = self.store.create_idea(
            self.run_id, json.dumps(idea_dict, ensure_ascii=False), "DEEPSEEK"
        )
        self.store.record_llm_request(
            self.run_id, "IDEA", self.config.deepseek_model,
            {"catalog_size": len(catalog)}, {}, usage,
        )
        idea = dict(idea_dict)
        idea["id"] = idea_id
        self._log("IDEA", "Tạo ý tưởng mới", {"idea_id": idea_id})
        return idea

    # -- Root batch loop ---------------------------------------------------

    def run_until_iteration_boundary(self):
        while not self.control.stop_requested():
            idea = self._get_or_create_idea()
            context = self.selector.select_context(idea)
            candidates = []

            for batch_number in range(1, self.config.max_batches_per_idea + 1):
                if self.control.stop_requested():
                    return EngineOutcome("STOPPED")

                drafts, usage = self.llm.generate_root_alphas(
                    idea, context, self.config.root_alphas_per_batch, self._lessons()
                )
                self.store.record_llm_request(
                    self.run_id, "ROOT_ALPHA", self.config.deepseek_model,
                    {"idea": idea.get("title"), "batch": batch_number}, {}, usage,
                )
                self._log("DEEPSEEK", f"Sinh {len(drafts)} Alpha gốc",
                          {"batch": batch_number})

                for index, draft in enumerate(drafts, 1):
                    if self.control.stop_requested():
                        return EngineOutcome("STOPPED")
                    outcome = self._validate_simulate_and_record(
                        idea, context, draft, batch_number, index
                    )
                    if outcome.parent is not None:
                        candidates.append(outcome.parent)

                parents = self._select_parents(candidates)
                if parents:
                    return EngineOutcome("PARENTS_READY", idea, context, parents)

            self.store.mark_idea_exhausted(idea["id"], "NO_PARENT_AFTER_MAX_BATCHES")
            self._log("RUN", "Ý tưởng cạn kiệt", {"idea_id": idea["id"]})

        return EngineOutcome("STOPPED")

    # -- Variants ----------------------------------------------------------

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
                draft, usage = self.llm.generate_variant(parent, direction, context)
                self.store.record_llm_request(
                    self.run_id, "VARIANT", self.config.deepseek_model,
                    {"direction": direction, "parent": parent.alpha_id}, {}, usage,
                )
                if draft.parent_id != parent.alpha_id or draft.generation != 1:
                    draft = replace(
                        draft,
                        parent_id=parent.alpha_id,
                        generation=1,
                        improvement_direction=direction,
                    )
                self._validate_simulate_and_record(idea, context, draft, None, None)

    # -- Record one alpha --------------------------------------------------

    def _validate_simulate_and_record(self, idea, context, draft, batch_number,
                                      alpha_index):
        hypothesis_id = self.store.create_hypothesis(
            idea["id"], draft.hypothesis, draft.rationale,
            draft.dataset_ids, draft.field_ids,
        )

        result = self.validator.validate(draft, context)
        if not result.is_valid:
            self.store.add_lesson("REJECTED", ",".join(result.error_codes))
            self._log("VALIDATE", "Loại Alpha", {"codes": result.error_codes})
            return RecordOutcome(None, None, simulated=False)

        alpha_id = self.store.create_alpha(
            self.run_id, hypothesis_id, draft.expression, result.expression_hash,
            result.fingerprint, draft.settings, draft.dataset_ids,
            draft.parent_id, draft.generation, draft.improvement_direction,
        )
        self.store.set_alpha_validation(alpha_id, "VALID")

        sim_payload = {
            "type": "REGULAR",
            "settings": draft.settings,
            "regular": draft.expression,
        }
        sim_result = self.worldquant.simulate_alpha(sim_payload)
        simulation_id = self.store.record_simulation(alpha_id, sim_result)

        qualification = self.policy.evaluate(sim_result)
        self.store.record_qualification(
            simulation_id, qualification.qualified,
            qualification.parent_eligible, qualification.reasons,
        )
        self._write_lesson(qualification, result, draft)
        self._log("SIMULATION", "Kết quả simulation", {
            "alpha_id": alpha_id,
            "status": sim_result.status,
            "qualified": qualification.qualified,
        })

        if qualification.qualified:
            self.store.enqueue_review(
                alpha_id, sim_result.worldquant_alpha_id or f"local-{alpha_id}"
            )
            self._log("QUALIFIED", "Alpha đạt chuẩn -> PENDING_REVIEW", {
                "alpha_id": alpha_id,
                "qualified_count": self.store.count_qualified_for_run(self.run_id),
            })

        parent = None
        if qualification.parent_eligible:
            parent = ParentCandidate(
                alpha_id=alpha_id,
                expression=draft.expression,
                hypothesis=draft.hypothesis,
                metrics=sim_result.metrics,
                sharpe_ratio=qualification.sharpe_ratio,
                fitness_ratio=qualification.fitness_ratio,
                turnover=float(sim_result.metrics.get("turnover", 0) or 0),
                qualified=qualification.qualified,
                order=alpha_id,
            )
        return RecordOutcome(alpha_id, parent, simulated=True)

    def _write_lesson(self, qualification, result, draft):
        fingerprint = result.fingerprint or draft.expression
        if qualification.qualified:
            self.store.add_lesson("QUALIFIED", fingerprint)
        elif qualification.parent_eligible:
            self.store.add_lesson("PARENT_ELIGIBLE", fingerprint)
        else:
            self.store.add_lesson("REJECTED", "; ".join(qualification.reasons))

    def _select_parents(self, candidates):
        if not candidates:
            return []
        ordered = sorted(
            candidates,
            key=lambda parent: (
                not parent.qualified,
                -min(parent.sharpe_ratio, parent.fitness_ratio),
                parent.turnover,
                parent.order,
            ),
        )
        return ordered[:self.config.max_parents]

    # -- Run loop ----------------------------------------------------------

    def run(self):
        from dataclasses import asdict
        self.run_id = self.store.start_run(self.snapshot_id, asdict(self.config))
        try:
            while not self.control.stop_requested():
                if self.store.count_qualified_for_run(self.run_id) >= (
                    self.config.target_qualified_per_run
                ):
                    self.store.finish_run(self.run_id, "TARGET_REACHED")
                    return RunOutcome(self.run_id, "TARGET_REACHED")

                boundary = self.run_until_iteration_boundary()
                if boundary.status == "PARENTS_READY":
                    self._run_variants(boundary.idea, boundary.context, boundary.parents)
                    self.store.mark_idea_exhausted(
                        boundary.idea["id"], "PARENTS_PROCESSED"
                    )
                elif boundary.status == "STOPPED":
                    break

            self.store.finish_run(self.run_id, "STOPPED_BY_USER")
            return RunOutcome(self.run_id, "STOPPED_BY_USER")
        except Exception:
            self.store.finish_run(self.run_id, "FAILED")
            raise
