# -*- coding: utf-8 -*-
"""Đối chiếu 100% field của FRONTIER_CORES với catalog DB thật (bảng data_fields).

Chạy TRƯỚC khi merge / sau khi load-fields:  venv\\Scripts\\python.exe tools\\verify_frontier_fields.py
Exit 0 = đủ hết; exit 1 = có field thiếu (in danh sách — cấm merge cho tới khi xử lý).
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.frontier_seeds import FRONTIER_CORES, FRONTIER_FIELDS  # noqa: E402
from src.storage.db import make_engine  # noqa: E402


def main() -> int:
    engine = make_engine()
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, type, dataset_id FROM data_fields")).fetchall()
    catalog = {r[0]: {"type": r[1], "dataset": r[2]} for r in rows}
    thieu = sorted(f for f in FRONTIER_FIELDS if f not in catalog)
    co = {f: catalog[f] for f in sorted(FRONTIER_FIELDS & set(catalog))}
    out = {
        "ngay": date.today().isoformat(), "n_cores": len(FRONTIER_CORES),
        "n_fields": len(FRONTIER_FIELDS), "thieu": thieu, "co": co,
    }
    # Đặt tên riêng "verified_frontier_fields_..." (khác "verified_fields_...")
    # để không đè lên bằng chứng của tools/verify_datasets.py cùng ngày.
    dest = Path("logs") / f"verified_frontier_fields_{date.today().strftime('%Y%m%d')}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Field frontier: {len(FRONTIER_FIELDS)} | thiếu trong catalog: {len(thieu)}")
    for f in thieu:
        print("  THIẾU:", f)
    print("Bằng chứng:", dest)
    return 1 if thieu else 0


if __name__ == "__main__":
    raise SystemExit(main())
