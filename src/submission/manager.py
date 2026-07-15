"""Submission Manager: chọn alpha đạt ngưỡng, check correlation, nộp WQ Brain."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

from loguru import logger

from config.thresholds import SUBMIT_MIN_FITNESS, SUBMIT_MIN_SHARPE
from src.storage.models import AlphaModel, SimulationModel, SubmissionModel, _utcnow


def _parse_retry_after(raw: str | None) -> float | None:
    """Parse header Retry-After thành float; None nếu thiếu/không parse được (an toàn,
    không đoán mò khi header lạ) — cùng phong cách `_parse_positive_number` của Simulator."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_failed_checks(payload: dict) -> list[dict]:
    """Trích các check FAIL trong body `GET /alphas/{id}/submit` khi bị từ chối — bằng chứng
    thật 2026-07-14: `{"is": {"checks": [{"name":..., "result": "FAIL"|"PASS", "limit":...,
    "value":...}, ...]}}`."""
    if not isinstance(payload, dict):
        return []
    checks = ((payload.get("is") or {}).get("checks")) or []
    if not isinstance(checks, list):
        return []
    return [c for c in checks if isinstance(c, dict) and c.get("result") == "FAIL"]


def _format_check(check: dict) -> str:
    """Định dạng 1 check FAIL kiểu 'LOW_SHARPE 1.41<1.58' (đúng bằng chứng response 403
    thật) — so sánh trực tiếp value/limit để suy dấu `<`/`>`, không đoán theo tên check."""
    name = check.get("name", "?")
    value = check.get("value")
    limit = check.get("limit")
    if value is None or limit is None:
        return str(name)
    op = "<" if value < limit else (">" if value > limit else "=")
    return f"{name} {value}{op}{limit}"


@dataclass
class Candidate:
    wq_alpha_id: str
    expression: str
    sharpe: float | None
    fitness: float | None
    score: float | None


@dataclass
class SubmissionResult:
    wq_alpha_id: str
    # submitted/rejected/error (như cũ) + pending (Bug 1: hết SUBMIT_POLL_TIMEOUT mà WQ vẫn
    # chưa tính xong) + unknown (Bug 1: poll không có check FAIL nhưng GET /alphas/{id} chưa
    # xác nhận được dateSubmitted/stage OS — không dám đoán là đã nộp).
    status: str
    detail: str = ""
    self_correlation: float | None = None


@dataclass
class PropertiesResult:
    wq_alpha_id: str
    status: str  # ok/unchanged/error
    detail: str = ""


# Pure Power Pool (docs power-pool-alphas.md): alpha KHÔNG đạt Regular vẫn nộp được nếu khớp
# theme. "Không đạt Regular" nghĩa là CHỈ fail các check hiệu năng Regular-only dưới đây —
# fail check khác (HIGH_TURNOVER, LOW_SUB_UNIVERSE_SHARPE, CONCENTRATED_WEIGHT...) thì Power
# Pool cũng đòi PASS ("Turnover tests PASS, Sub-universe test PASS"), loại thẳng.
_PP_REGULAR_ONLY_FAILS = frozenset(
    {"LOW_SHARPE", "LOW_FITNESS", "IS_LADDER_SHARPE", "LOW_2Y_SHARPE"}
)
# Self-corr trong pool Power Pool phải <= 0.5 (chặt hơn 0.7 của Regular).
_PP_SELF_CORR_MAX = 0.5


@dataclass
class PowerPoolCandidate:
    """Ứng viên pure Power Pool + kết quả chấm theme/mô tả (skip_reason rỗng = nộp được)."""

    wq_alpha_id: str
    expression: str
    sharpe: float | None
    fitness: float | None
    theme_ok: bool
    theme_reasons: list[str]
    description: str | None
    skip_reason: str = ""


class SubmissionManager:
    DAILY_QUOTA = 10
    # Bug 1 (fix-submit-async, bằng chứng thật 2026-07-14 KP9nwpEg): `POST /alphas/{id}/submit`
    # trả 200/201 NGAY nhưng đó KHÔNG phải kết quả — WQ tính bất đồng bộ, phải poll GET cùng
    # path (`GET /alphas/{id}/submit`) tới khi có kết quả thật. Tổng thời gian chờ tối đa.
    # Bug 3 (bằng chứng thật 2026-07-15 KP92dQAx, 3 lần đo): job submit phía Brain HẾT HẠN
    # đúng 30 phút (GET 200-rỗng suốt 30' rồi 404-rỗng, không is.checks nào) -> timeout poll
    # phía client phải DÀI HƠN cửa sổ job 30' để luôn bắt được tín hiệu 404 thay vì bỏ cuộc
    # giữa chừng ở 600s như trước (khi đó không phân biệt được 'đang tính' với 'job chết').
    SUBMIT_POLL_TIMEOUT = 2100.0
    # Bug 3: gặp 404-rỗng (job hết hạn) thì POST vòng mới — Brain cache phần check đã tính
    # nên vòng sau tiến xa hơn (forum 2025-03 khuyên "submit bằng code + retry nhiều lần").
    SUBMIT_RETRY_CYCLES = 4
    # Mặc định khi response poll thiếu header Retry-After.
    SUBMIT_POLL_DEFAULT_RETRY = 3.0
    # Cap 1 lần chờ giữa 2 lần poll — chống Retry-After bất thường làm treo quá lâu.
    SUBMIT_POLL_MAX_RETRY = 30.0

    def __init__(
        self,
        client,
        session_factory,
        correlation_checker,
        daily_quota: int | None = None,
        diversify: bool = False,
        max_struct_similarity: float = 0.9,
        sleep_func=time.sleep,
        time_func=time.monotonic,
        submit_retry_cycles: int | None = None,
    ):
        self.client = client
        self.session_factory = session_factory
        self.correlation = correlation_checker
        self.daily_quota = daily_quota if daily_quota is not None else self.DAILY_QUOTA
        # T7.1: loại alpha trùng cấu trúc (AST) với alpha đã chọn trong cùng tập nộp.
        self.diversify = diversify
        self.max_struct_similarity = max_struct_similarity
        # Bug 1: cùng phong cách injectable clock của Simulator (_sleep/_time) để test poll
        # bất đồng bộ không cần chờ thật.
        self._sleep = sleep_func
        self._time = time_func
        self.submit_retry_cycles = (
            submit_retry_cycles if submit_retry_cycles is not None else self.SUBMIT_RETRY_CYCLES
        )

    # --------------------------------------------------------------- selection
    def select_candidates(self) -> list[Candidate]:
        """Chọn alpha đã pass sim để nộp. `status == "passed"` phản ánh `is.checks` LÚC SIM
        (xem `Simulator._fetch_metrics`) nhưng KHÔNG ĐỦ — Bug 2 (bằng chứng thật 2026-07-14,
        alpha KP9nwpEg: Sharpe 1.41, fitness 0.99, `failed_checks=[]` lúc sim vẫn bị Brain
        403 REJECTED lúc `POST /submit` thật với `is.checks` LOW_SHARPE/LOW_FITNESS) chứng
        minh Brain ENFORCE LẠI Sharpe/Fitness lúc NỘP THẬT, khác lúc sim. Lọc thêm
        `SUBMIT_MIN_SHARPE`/`SUBMIT_MIN_FITNESS` (config/thresholds.py, đo trực tiếp từ
        response 403 đó) ở đây để không chọn ứng viên chắc chắn bị từ chối, tốn quota nộp
        oan."""
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
            if sim.sharpe is None or sim.sharpe < SUBMIT_MIN_SHARPE:
                continue
            if sim.fitness is None or sim.fitness < SUBMIT_MIN_FITNESS:
                continue
            seen.add(sim.wq_alpha_id)
            candidates.append(
                Candidate(sim.wq_alpha_id, alpha.expression, sim.sharpe, sim.fitness, sim.score)
            )
        return candidates

    # -------------------------------------------------------- pure Power Pool
    def select_power_pool_candidates(
        self, on_date=None, calendar=None
    ) -> list[PowerPoolCandidate]:
        """Chọn ứng viên PURE Power Pool (docs power-pool-alphas.md + bằng chứng thật
        2026-07-15 KP92dQAx): Sharpe>=1.0, KHÔNG đạt ngưỡng Regular (đường đó là
        `select_candidates`), chỉ fail check Regular-only (`_PP_REGULAR_ONLY_FAILS`),
        đạt cấu trúc PP (<=8 operator, <=3 field), rồi chấm theme compliance bằng
        settings THẬT đã sim (raw_result) + map field->dataset trong DB. `calendar`
        inject được cho test; None -> lịch thủ công trong power_pool_theme."""
        from datetime import date as _date

        from src.lang.parser import parse_expression
        from src.lang.registry import default_registry
        from src.lang.visitors import FieldCollector
        from src.llm.hypothesis import Hypothesis
        from src.scoring.power_pool import (
            _GROUPING_FIELDS,
            build_power_pool_description,
            check_power_pool_eligibility,
            is_valid_power_pool_description,
        )
        from src.scoring.power_pool_theme import check_theme_compliance
        from src.storage.models import DataFieldModel

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
                .filter(SimulationModel.wq_alpha_id.isnot(None))
                .order_by(SimulationModel.sharpe.desc())
                .all()
            )
            candidates: list[PowerPoolCandidate] = []
            seen: set[str] = set()
            for sim, alpha in rows:
                wq_id = sim.wq_alpha_id
                if wq_id in submitted or wq_id in seen:
                    continue
                if sim.sharpe is None or sim.sharpe < 1.0:
                    continue
                if (
                    sim.sharpe >= SUBMIT_MIN_SHARPE
                    and sim.fitness is not None
                    and sim.fitness >= SUBMIT_MIN_FITNESS
                ):
                    continue  # đạt Regular -> nộp đường run_daily, không phải pure PP
                try:
                    fails = set(json.loads(sim.failed_checks) or []) if sim.failed_checks else set()
                except (ValueError, TypeError):
                    continue  # failed_checks hỏng -> không dám đoán, bỏ ứng viên
                if fails - _PP_REGULAR_ONLY_FAILS:
                    continue  # fail check mà Power Pool cũng đòi PASS (turnover/sub-universe...)
                try:
                    verdict = check_power_pool_eligibility(alpha.expression, sim.sharpe)
                except Exception:  # noqa: BLE001 - biểu thức lạ không được làm sập selection
                    continue
                if not verdict.eligible:
                    continue
                seen.add(wq_id)

                settings: dict = {}
                if sim.raw_result:
                    try:
                        settings = (json.loads(sim.raw_result) or {}).get("settings") or {}
                    except (ValueError, TypeError):
                        settings = {}
                region = settings.get("region") or sim.region or "USA"
                universe = settings.get("universe") or sim.universe or ""
                delay = settings.get("delay", 1)
                neut = settings.get("neutralization") or "NONE"
                try:
                    fields = (
                        FieldCollector(default_registry()).visit(parse_expression(alpha.expression))
                        - _GROUPING_FIELDS
                    )
                except Exception:  # noqa: BLE001
                    fields = set()
                datasets: set[str] = set()
                if fields:
                    datasets = {
                        ds
                        for (ds,) in session.query(DataFieldModel.dataset_id)
                        .filter(DataFieldModel.id.in_(fields))
                        .distinct()
                        .all()
                        if ds
                    }
                theme_ok, reasons = check_theme_compliance(
                    region=region, delay=delay, universe=universe, neutralization=neut,
                    datasets_used=datasets, on_date=on_date or _date.today(), calendar=calendar,
                )

                description = None
                if alpha.hypothesis:
                    try:
                        hyp = Hypothesis.from_dict(json.loads(alpha.hypothesis))
                        desc = build_power_pool_description(hyp)
                        if is_valid_power_pool_description(desc):
                            description = desc
                    except (ValueError, TypeError):
                        description = None

                skip = ""
                if not theme_ok:
                    skip = "lệch theme: " + "; ".join(reasons)
                elif not description:
                    skip = (
                        "thiếu mô tả Idea/Rationale >=100 ký tự (docs bắt buộc) — "
                        "alpha không có hypothesis 4 phần"
                    )
                candidates.append(
                    PowerPoolCandidate(
                        wq_id, alpha.expression, sim.sharpe, sim.fitness,
                        theme_ok, reasons, description, skip,
                    )
                )
            return candidates
        finally:
            session.close()

    def submit_power_pool(
        self, dry_run: bool = True, on_date=None, calendar=None, pp_quota: int = 1,
    ) -> list[tuple[PowerPoolCandidate, "SubmissionResult | None"]]:
        """Nộp pure Power Pool: set description (bắt buộc theo docs) rồi dùng lại `submit()`
        (poll + retry job hết hạn + xác nhận dateSubmitted). Quota mặc định 1 pure PP/lần
        (docs: 1 pure PP/ngày sau 3 tháng; trước đó theo hạn 4 alpha/ngày chung — để 1 cho
        an toàn, chỉnh qua `pp_quota`). Chỉ đếm quota khi nộp THÀNH CÔNG (rejected không
        tiêu slot). self-corr pool phải <= 0.5 (chặt hơn Regular 0.7)."""
        cands = self.select_power_pool_candidates(on_date=on_date, calendar=calendar)
        outcomes: list[tuple[PowerPoolCandidate, SubmissionResult | None]] = []
        n_submitted = 0
        for cand in cands:
            if dry_run or cand.skip_reason or n_submitted >= pp_quota:
                outcomes.append((cand, None))
                continue
            corr = self.correlation.max_self_correlation(cand.wq_alpha_id)
            if corr > _PP_SELF_CORR_MAX:
                cand.skip_reason = (
                    f"self-corr {corr:.3f} > {_PP_SELF_CORR_MAX} (ngưỡng Power Pool)"
                )
                outcomes.append((cand, None))
                continue
            self.set_properties(cand.wq_alpha_id, regular_desc=cand.description)
            # submit() khi thành công tự gắn tag PowerPoolSelected qua
            # _tag_if_power_pool_eligible (cùng nguồn hypothesis) — không gắn lại ở đây.
            result = self.submit(cand.wq_alpha_id)
            if result.status == "submitted":
                n_submitted += 1
            outcomes.append((cand, result))
        return outcomes

    # ------------------------------------------------------------------ submit
    def submit(self, wq_alpha_id: str) -> SubmissionResult:
        corr = self.correlation.max_self_correlation(wq_alpha_id)
        if corr > self.correlation.max_self_corr:
            result = SubmissionResult(
                wq_alpha_id, "rejected", f"self-corr {corr:.3f} > {self.correlation.max_self_corr}", corr
            )
            self._record(result)
            return result

        # Bug 3: job submit phía Brain hết hạn sau ~30' (404-rỗng) -> POST vòng mới, tối đa
        # `submit_retry_cycles` vòng (Brain cache phần đã tính nên vòng sau tiến xa hơn).
        status, detail = "pending", ""
        for cycle in range(1, self.submit_retry_cycles + 1):
            try:
                resp = self.client.post(f"/alphas/{wq_alpha_id}/submit")
            except Exception as exc:  # noqa: BLE001 - không để pipeline crash
                result = SubmissionResult(wq_alpha_id, "error", str(exc), corr)
                self._record(result)
                return result

            if resp.status_code not in (200, 201):
                result = SubmissionResult(wq_alpha_id, "error", f"HTTP {resp.status_code}", corr)
                self._record(result)
                return result

            # Bug 1: POST 200/201 KHÔNG có nghĩa đã nộp — WQ tính bất đồng bộ, phải poll GET
            # /alphas/{id}/submit tới khi có kết quả thật (403 = từ chối, xác nhận GET
            # /alphas/{id} = thành công thật).
            status, detail = self._poll_submit_result(wq_alpha_id)
            if status != "job_expired":
                break
            # 404-rỗng: job hết hạn — NHƯNG có thể nộp đã xong ngay trước khi record biến mất,
            # xác nhận qua GET /alphas/{id} trước khi đốt vòng POST mới.
            confirmed, _ = self._confirm_submitted(wq_alpha_id)
            if confirmed:
                status, detail = "submitted", "ok (xác nhận sau khi job record biến mất)"
                break
            logger.info(
                "Job submit Brain hết hạn (404 rỗng) vòng {}/{} cho {} — POST lại.",
                cycle, self.submit_retry_cycles, wq_alpha_id,
            )
        else:
            status, detail = (
                "pending",
                f"job submit Brain hết hạn {self.submit_retry_cycles} vòng liên tiếp "
                "(mỗi vòng ~30') — kết quả chưa biết, thử lại sau",
            )

        result = SubmissionResult(wq_alpha_id, status, detail, corr)
        self._record(result)
        if result.status == "submitted":
            self._tag_if_power_pool_eligible(wq_alpha_id)
        return result

    def _poll_submit_result(self, wq_alpha_id: str) -> tuple[str, str]:
        """Poll `GET /alphas/{id}/submit` tới khi có kết quả bất đồng bộ (Bug 1, bằng chứng
        thật 2026-07-14 KP9nwpEg): body RỖNG + 200 = đang tính (chờ theo Retry-After rồi poll
        tiếp, cùng kiểu `CorrelationChecker.max_self_correlation`); 403 với `is.checks` chứa
        FAIL = bị từ chối THẬT (khác hẳn code cũ coi `status_code in (200,201)` là đã nộp);
        2xx có kết quả không FAIL nào -> CHƯA vội tin đã nộp, xác nhận thêm bằng
        `GET /alphas/{id}` (dateSubmitted/stage) trước khi trả 'submitted' — không đoán mò
        khi chưa xác nhận được thì trả 'unknown'. Vượt `SUBMIT_POLL_TIMEOUT` mà vẫn rỗng ->
        'pending' (đừng ghi submitted khi chưa biết kết quả thật)."""
        deadline = self._time() + self.SUBMIT_POLL_TIMEOUT
        while True:
            try:
                resp = self.client.get(f"/alphas/{wq_alpha_id}/submit")
            except Exception as exc:  # noqa: BLE001 - không để pipeline crash
                return "error", str(exc)

            body = (getattr(resp, "text", "") or "").strip()
            if resp.status_code in (200, 201) and not body:
                if self._time() >= deadline:
                    return "pending", f"chưa có kết quả sau {self.SUBMIT_POLL_TIMEOUT:.0f}s poll"
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                self._sleep(min(retry_after or self.SUBMIT_POLL_DEFAULT_RETRY, self.SUBMIT_POLL_MAX_RETRY))
                continue

            # Bug 3 (bằng chứng thật 2026-07-15 KP92dQAx): 404 BODY RỖNG sau chuỗi 200-rỗng
            # = job submit phía Brain hết hạn (~30'), KHÔNG phải phán quyết — caller (submit)
            # sẽ xác nhận dateSubmitted rồi POST vòng mới. 404 CÓ body thì vẫn là error thật.
            if resp.status_code == 404 and not body:
                return "job_expired", "GET /submit 404 rỗng — job Brain hết hạn"

            if resp.status_code == 403:
                payload = resp.json() if body else {}
                fails = _extract_failed_checks(payload)
                detail = ", ".join(_format_check(c) for c in fails) if fails else "HTTP 403"
                return "rejected", detail

            if resp.status_code in (200, 201):
                payload = resp.json()
                fails = _extract_failed_checks(payload)
                if fails:
                    return "rejected", ", ".join(_format_check(c) for c in fails)
                confirmed, confirm_detail = self._confirm_submitted(wq_alpha_id)
                if confirmed:
                    return "submitted", "ok"
                return "unknown", confirm_detail

            return "error", f"HTTP {resp.status_code}"

    def _confirm_submitted(self, wq_alpha_id: str) -> tuple[bool, str]:
        """Xác nhận nộp thật bằng `GET /alphas/{id}` (dateSubmitted/stage OS) — poll
        /submit không FAIL nào chỉ là tín hiệu KHẢ NĂNG thành công, không phải bằng chứng
        (chưa có bằng chứng thật cho body 2xx hoàn tất của endpoint /submit), nên xác nhận
        chéo bằng endpoint alpha chính trước khi dám báo 'submitted'."""
        try:
            resp = self.client.get(f"/alphas/{wq_alpha_id}")
        except Exception as exc:  # noqa: BLE001 - không để pipeline crash
            return False, f"không xác nhận được qua GET /alphas/{{id}}: {exc}"
        if resp.status_code not in (200, 201):
            return False, f"không xác nhận được qua GET /alphas/{{id}}: HTTP {resp.status_code}"
        payload = resp.json()
        if payload.get("dateSubmitted") or payload.get("stage") == "OS":
            return True, "ok"
        return (
            False,
            "poll /submit không có check FAIL nhưng GET /alphas/{id} chưa thấy "
            "dateSubmitted/stage OS -> không dám khẳng định đã nộp",
        )

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

    # ---------------------------------------------------------- set_properties
    def set_properties(
        self,
        wq_alpha_id: str,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        regular_desc: str | None = None,
        combo_desc: str | None = None,
        selection_desc: str | None = None,
        color: str | None = None,
    ) -> PropertiesResult:
        """Set name/color/tags/mô tả cho alpha qua PATCH /alphas/{id} (T-C.4). Idempotent:
        bỏ qua gọi API nếu tags+regular_desc giống hệt lần set gần nhất đã lưu."""
        payload: dict = {}
        if name:
            payload["name"] = name
        if color:
            payload["color"] = color
        if tags:
            payload["tags"] = tags
        if selection_desc:
            payload["selectionDesc"] = selection_desc
        if combo_desc:
            payload["comboDesc"] = combo_desc
        if regular_desc:
            payload["regular"] = {"description": regular_desc}

        tags_json = json.dumps(tags) if tags else None
        session = self.session_factory()
        try:
            row = (
                session.query(SubmissionModel)
                .filter(SubmissionModel.alpha_id == wq_alpha_id)
                .order_by(SubmissionModel.submitted_at.desc())
                .first()
            )
            if row is not None and row.tags == tags_json and row.regular_desc == regular_desc:
                return PropertiesResult(wq_alpha_id, "unchanged", "giống lần set trước")
        finally:
            session.close()

        try:
            resp = self.client.patch(f"/alphas/{wq_alpha_id}", json=payload)
        except Exception as exc:  # noqa: BLE001 - không để pipeline crash
            self._record_properties(wq_alpha_id, tags_json, regular_desc, ok=False)
            return PropertiesResult(wq_alpha_id, "error", str(exc))

        if resp.status_code not in (200, 201):
            self._record_properties(wq_alpha_id, tags_json, regular_desc, ok=False)
            return PropertiesResult(wq_alpha_id, "error", f"HTTP {resp.status_code}")

        self._record_properties(wq_alpha_id, tags_json, regular_desc, ok=True)
        return PropertiesResult(wq_alpha_id, "ok", "da set properties")

    def _record_properties(
        self, wq_alpha_id: str, tags_json: str | None, regular_desc: str | None, *, ok: bool
    ) -> None:
        session = self.session_factory()
        try:
            row = (
                session.query(SubmissionModel)
                .filter(SubmissionModel.alpha_id == wq_alpha_id)
                .order_by(SubmissionModel.submitted_at.desc())
                .first()
            )
            set_at = _utcnow() if ok else None
            if row is not None:
                row.tags = tags_json
                row.regular_desc = regular_desc
                row.properties_set_at = set_at
            else:
                session.add(
                    SubmissionModel(
                        id=uuid.uuid4().hex,
                        alpha_id=wq_alpha_id,
                        status="properties_set",
                        tags=tags_json,
                        regular_desc=regular_desc,
                        properties_set_at=set_at,
                    )
                )
            session.commit()
        finally:
            session.close()

    def run_daily(self, dry_run: bool = True) -> list[Candidate]:
        """Chọn ≤ quota alpha tốt nhất, không trùng correlation. Nộp nếu không dry-run."""
        candidates = self.select_candidates()
        selected: list[Candidate] = []
        for cand in candidates:
            if len(selected) >= self.daily_quota:
                break
            if not self.correlation.is_acceptable(cand.wq_alpha_id):
                logger.info("Bỏ {} vì self-correlation cao", cand.wq_alpha_id)
                continue
            if self.diversify and self._too_similar(cand, selected):
                logger.info("Bỏ {} vì trùng cấu trúc với alpha đã chọn", cand.wq_alpha_id)
                continue
            selected.append(cand)
            if not dry_run:
                self.submit(cand.wq_alpha_id)
        return selected

    def _too_similar(self, cand: Candidate, selected: list[Candidate]) -> bool:
        """True nếu cand trùng cấu trúc AST quá ngưỡng với một alpha đã chọn (T7.1)."""
        from src.decorrelation.similarity import similarity_ratio

        for chosen in selected:
            try:
                if similarity_ratio(cand.expression, chosen.expression) >= self.max_struct_similarity:
                    return True
            except ValueError:
                continue  # parse lỗi -> không chặn vì lý do cấu trúc
        return False

    # ------------------------------------------------------------------ record
    def _record(self, result: SubmissionResult) -> None:
        session = self.session_factory()
        try:
            session.add(
                SubmissionModel(
                    id=uuid.uuid4().hex,
                    alpha_id=result.wq_alpha_id,
                    status=result.status,
                    self_correlation=result.self_correlation,
                    detail=result.detail,
                )
            )
            session.commit()
        finally:
            session.close()
