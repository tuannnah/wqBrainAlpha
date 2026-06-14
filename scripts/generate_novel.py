"""Kiểm tra 10 alpha mới mẻ qua cửa cứng PreFilter (operator/field thật) rồi log.

Khác generate_alphas.py: KHÔNG sinh tổ hợp đại trà, mà lấy 10 alpha thủ công từ
novel_ideas.NOVEL_ALPHAS (dataset ít người khai thác: option/news/social/analyst/
supply-chain). Chỉ validate cú pháp + field/operator tồn tại thật, rồi ghi report
chi tiết kèm setting ra output/alphas_novel_<ngày>.txt.

Chạy:  venv/Scripts/python -m scripts.generate_novel
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from loguru import logger

from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository
from src.generation.alpha_logger import format_report
from src.generation.local_select import originality_score
from src.generation.novel_ideas import NOVEL_ALPHAS
from src.decorrelation.zoo import ALPHA101_FASTEXPR
from src.scoring.complexity import complexity_penalty
from src.simulation.pre_filter import PreFilter
from src.storage.db import init_db, make_session_factory

REGION, UNIVERSE, DELAY = "USA", "TOP3000", 1


def _load_symbols() -> tuple[set[str], set[str]]:
    engine = init_db()
    session_factory = make_session_factory(engine)
    op_repo = OperatorRepository(client=None, session_factory=session_factory)
    fld_repo = FieldRepository(client=None, session_factory=session_factory)
    operators = {o.name for o in op_repo.load_cached() if o.name}
    fields = {
        f.id
        for f in fld_repo.load_cached(region=REGION, universe=UNIVERSE, delay=DELAY)
        if f.id
    }
    logger.info("Đã load {} operators, {} fields từ cache", len(operators), len(fields))
    return operators, fields


def run(output_path: Path | None = None) -> Path:
    operators, fields = _load_symbols()
    if not operators or not fields:
        raise SystemExit("Cache rỗng. Hãy login + tải data-fields/operators trước.")

    prefilter = PreFilter(
        known_operators=operators,
        known_fields=fields,
        max_depth=6,
        max_nodes=30,
    )

    passed = []
    for c in NOVEL_ALPHAS:
        ok, reason = prefilter.check(c.expression)
        if not ok:
            logger.error("LOẠI [{}]: {} — {}", c.family, c.expression, reason)
            continue
        c.originality = originality_score(c.expression, ALPHA101_FASTEXPR)
        c.complexity = complexity_penalty(c.expression)
        c.score = 0.6 * c.originality + 0.4 * (1.0 - c.complexity)
        c.reasons = [
            f"originality={c.originality:.2f}",
            f"complexity={c.complexity:.2f}",
            "dataset thay thế (option/news/social/analyst/graph)",
            "cú pháp/field/operator hợp lệ",
        ]
        passed.append(c)

    passed.sort(key=lambda x: x.score, reverse=True)
    logger.success("{}/{} alpha mới qua cửa cứng", len(passed), len(NOVEL_ALPHAS))

    report = format_report(passed)
    out = output_path or Path("output") / f"alphas_novel_{date.today().isoformat()}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    logger.success("Đã ghi report: {}", out)
    return out


if __name__ == "__main__":
    run()
