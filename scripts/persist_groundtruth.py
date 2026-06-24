"""Ghi kết quả simulation (từ MCP create_multi_simulation) vào DB account đích.

Đọc 1 file JSON là list bản ghi:
  [{"expression": str, "family": str, "wq_alpha_id": str|null,
    "status": "passed"|"failed"|"error",
    "sharpe": float|null, "fitness": ..., "turnover": ..., "returns": ...,
    "drawdown": ..., "margin": ..., "raw": {...}}]

Mỗi bản ghi -> AlphaRepository.save_simulation (lưu alpha + simulation), hoặc
record_failure nếu status='error'. KHÔNG viết SQL raw — dùng đúng API repository.

Dùng: python scripts/persist_groundtruth.py <results.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation.simulator import SimulationResult  # noqa: E402
from src.storage.db import make_engine, init_db, make_session_factory  # noqa: E402
from src.storage.repository import AlphaRepository  # noqa: E402

DB_URL = "sqlite:///wq_alpha_phtrang1229_gmail_com.db"
REGION = "USA"
UNIVERSE = "TOP3000"
# config_key cố định để expr_hash ổn định khi tái mô phỏng bộ này (calibration).
CONFIG_KEY = "USA|TOP3000|delay1|groundtruth"

_METRIC_KEYS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin")


def main() -> None:
    results_path = Path(sys.argv[1])
    records = json.loads(results_path.read_text(encoding="utf-8"))

    engine = make_engine(DB_URL)
    init_db(engine)
    repo = AlphaRepository(make_session_factory(engine))

    n_sim, n_fail = 0, 0
    for rec in records:
        expr = rec["expression"]
        family = rec.get("family", "groundtruth")
        status = rec.get("status", "error")
        if status == "error" or rec.get("sharpe") is None:
            repo.record_failure(
                expression=expr,
                category="sim_error",
                reason=str(rec.get("raw", {}).get("error", rec.get("error", "no metrics"))),
                source=family,
            )
            n_fail += 1
            continue
        metrics = {k: rec.get(k) for k in _METRIC_KEYS}
        result = SimulationResult(
            expression=expr,
            alpha_id=rec.get("wq_alpha_id"),
            status=status,
            raw=rec.get("raw", {}),
            **metrics,
        )
        repo.save_simulation(
            result=result,
            region=REGION,
            universe=UNIVERSE,
            source=family,
            config_key=CONFIG_KEY,
        )
        n_sim += 1

    print(f"PERSISTED sims={n_sim} failures={n_fail}")
    # Tổng kết hiện trạng DB.
    from src.storage.models import SimulationModel
    sf = make_session_factory(engine)
    s = sf()
    try:
        total = s.query(SimulationModel).filter(SimulationModel.sharpe.isnot(None)).count()
        print(f"DB total sims with sharpe NOT NULL = {total}")
    finally:
        s.close()


if __name__ == "__main__":
    main()
