"""Sinh & lọc alpha kinh điển rồi log chi tiết ra file text.

Nối toàn bộ pipeline (không gọi API, không simulate — chỉ dùng cache + lọc local):
  1. Load 67 operators + 8599 fields đã cache trong DB (scope USA/TOP3000/delay=1).
  2. Sinh ứng viên thô theo 7 họ kinh điển (families.generate_candidates).
  3. Lọc local: cửa cứng PreFilter + originality (so zoo Alpha101) + complexity +
     khử trùng nội bộ + quota đa dạng theo họ (local_select.select_alphas).
  4. Ghi report chi tiết (alpha_logger.format_report) ra output/alphas_<ngày>.txt.

Chạy:  venv/Scripts/python -m scripts.generate_alphas
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from loguru import logger

from src.data.fields import FieldRepository
from src.data.operators import OperatorRepository
from src.decorrelation.zoo import ALPHA101_FASTEXPR
from src.generation.alpha_logger import format_report
from src.generation.families import generate_candidates
from src.generation.local_select import select_alphas
from src.storage.db import init_db, make_session_factory

# Scope khớp config/sim_defaults.yaml.
REGION, UNIVERSE, DELAY = "USA", "TOP3000", 1


def _load_symbols() -> tuple[set[str], set[str]]:
    """Đọc operators + fields đã cache từ DB (không gọi API)."""
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


def run(
    per_family_quota: int = 40,
    per_canon_quota: int = 6,
    max_total: int = 120,
    output_path: Path | None = None,
) -> Path:
    operators, fields = _load_symbols()
    if not operators or not fields:
        raise SystemExit(
            "Cache rỗng. Hãy chạy tool login + tải data-fields/operators trước "
            "(scope USA/TOP3000/delay=1)."
        )

    raw = generate_candidates()
    logger.info("Sinh {} ứng viên thô từ 7 họ kinh điển", len(raw))

    selected = select_alphas(
        raw,
        zoo=ALPHA101_FASTEXPR,
        known_operators=operators,
        known_fields=fields,
        per_canon_quota=per_canon_quota,
        per_family_quota=per_family_quota,
        max_total=max_total,
    )
    logger.success("Còn {} alpha đạt chuẩn sau lọc local", len(selected))

    report = format_report(selected)
    out = output_path or Path("output") / f"alphas_{date.today().isoformat()}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    logger.success("Đã ghi report: {}", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh & lọc alpha kinh điển -> log text")
    parser.add_argument("--per-family", type=int, default=40, help="Quota mỗi họ")
    parser.add_argument(
        "--per-canon", type=int, default=6, help="Số biến thể tối đa mỗi khung cấu trúc"
    )
    parser.add_argument("--max-total", type=int, default=120, help="Trần tổng số alpha")
    parser.add_argument("--out", type=str, default="", help="Đường dẫn file output")
    args = parser.parse_args()

    run(
        per_family_quota=args.per_family,
        per_canon_quota=args.per_canon,
        max_total=args.max_total,
        output_path=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
