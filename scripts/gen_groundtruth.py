"""Sinh bộ alpha ground-truth cho hiệu chỉnh (calibration) Phase 4.5.

Ràng buộc CỨNG:
- Chỉ 8 field: close, open, high, low, volume, vwap, returns, sector.
- Chỉ operator có TÊN TRÙNG ở CẢ registry local (src/operators_local) LẪN nền tảng
  Brain — để cùng một alpha chấm được ở cả hai phía (điều kiện để Spearman ρ có
  nghĩa). Vì vậy TRÁNH: ts_std (local) / ts_std_dev (Brain) lệch tên;
  regression_neut/vector_neut (chỉ local). group_neutralize có ở cả hai phía.
- Độ sâu cây trần <= MAX_DEPTH (7), kiểm bằng DepthVisitor.
- Đa dạng: 6 family cấu trúc, mỗi family 8-12 alpha.

Determinism: seed RNG cố định để tái tạo bộ; ghi seed ra scratch.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.thresholds import MAX_DEPTH  # noqa: E402
from src.lang.parser import parse_expression  # noqa: E402
from src.lang.visitors import DepthVisitor, FieldCollector  # noqa: E402

SEED = 20260624
ALLOWED_FIELDS = {"close", "open", "high", "low", "volume", "vwap", "returns", "sector"}
# Operator có tên trùng cả local lẫn Brain (giao tập an toàn cho calibration).
ALLOWED_OPS = {
    "add", "subtract", "multiply", "divide", "log", "abs", "sign", "power",
    "max", "min", "rank", "winsorize", "zscore", "scale", "group_neutralize",
    "ts_mean", "ts_delay", "ts_delta", "ts_rank", "ts_zscore", "ts_corr",
    "ts_decay_linear", "ts_backfill", "trade_when", "hump",
}


def _collect_ops(node) -> set[str]:
    from src.lang.ast import Call

    out: set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Call):
            out.add(n.op)
        stack.extend(n.children())
    return out


def build_expressions(rng: random.Random) -> list[tuple[str, str]]:
    """Trả [(family, expression)] — 6 family, mỗi family 8-12 alpha."""
    out: list[tuple[str, str]] = []

    # ---- Family A: Momentum (ts_delta / ts_rank) ----
    mom_windows = [5, 10, 20, 40, 60, 120]
    for d in mom_windows:
        out.append(("momentum", f"rank(ts_delta(close, {d}))"))
    out.append(("momentum", "rank(ts_rank(close, 20))"))
    out.append(("momentum", "ts_rank(ts_delta(vwap, 10), 60)"))
    out.append(("momentum", "rank(subtract(close, ts_delay(close, 20)))"))
    out.append(("momentum", "rank(ts_delta(ts_mean(close, 5), 20))"))

    # ---- Family B: Mean-reversion (đảo dấu tín hiệu ngắn hạn) ----
    for d in [3, 5, 10]:
        out.append(("mean_reversion", f"rank(multiply(-1, ts_delta(close, {d})))"))
    out.append(("mean_reversion", "rank(subtract(ts_mean(close, 20), close))"))
    out.append(("mean_reversion", "multiply(-1, ts_zscore(returns, 10))"))
    out.append(("mean_reversion", "rank(subtract(vwap, close))"))
    out.append(("mean_reversion", "multiply(-1, rank(returns))"))
    out.append(("mean_reversion", "rank(divide(subtract(ts_mean(close, 10), close), close))"))
    out.append(("mean_reversion", "ts_rank(multiply(-1, returns), 20)"))
    out.append(("mean_reversion", "multiply(-1, ts_delta(rank(close), 5))"))

    # ---- Family C: Volume-price interaction ----
    out.append(("volume_price", "rank(multiply(ts_delta(close, 5), volume))"))
    out.append(("volume_price", "rank(ts_corr(close, volume, 20))"))
    out.append(("volume_price", "rank(divide(volume, ts_mean(volume, 20)))"))
    out.append(("volume_price", "multiply(rank(returns), rank(volume))"))
    out.append(("volume_price", "rank(ts_corr(returns, ts_delta(volume, 1), 10))"))
    out.append(("volume_price", "rank(multiply(returns, log(volume)))"))
    out.append(("volume_price", "trade_when(ts_delta(volume, 1), rank(ts_delta(close, 5)), -1)"))
    out.append(("volume_price", "rank(divide(multiply(close, volume), ts_mean(multiply(close, volume), 20)))"))
    out.append(("volume_price", "rank(ts_corr(vwap, volume, 15))"))
    out.append(("volume_price", "multiply(sign(ts_delta(close, 5)), rank(volume))"))

    # ---- Family D: Volatility / dispersion (tránh ts_std; dùng ts_zscore/ts_corr/range) ----
    out.append(("volatility", "rank(ts_zscore(returns, 20))"))
    out.append(("volatility", "rank(divide(subtract(high, low), close))"))
    out.append(("volatility", "rank(ts_mean(divide(subtract(high, low), close), 10))"))
    out.append(("volatility", "multiply(-1, rank(ts_zscore(close, 60)))"))
    out.append(("volatility", "rank(divide(subtract(high, low), ts_mean(subtract(high, low), 20)))"))
    out.append(("volatility", "rank(ts_delta(divide(subtract(high, low), close), 10))"))
    out.append(("volatility", "ts_rank(divide(subtract(high, low), close), 40)"))
    out.append(("volatility", "rank(multiply(ts_zscore(returns, 10), -1))"))

    # ---- Family E: Cross-sectional ranking thuần (intraday range/gap, vwap dist) ----
    out.append(("cross_sectional", "rank(divide(subtract(close, open), open))"))
    out.append(("cross_sectional", "rank(divide(subtract(close, low), subtract(high, low)))"))
    out.append(("cross_sectional", "rank(divide(subtract(high, close), subtract(high, low)))"))
    out.append(("cross_sectional", "zscore(divide(subtract(close, vwap), vwap))"))
    out.append(("cross_sectional", "rank(divide(subtract(open, ts_delay(close, 1)), ts_delay(close, 1)))"))
    out.append(("cross_sectional", "rank(divide(close, vwap))"))
    out.append(("cross_sectional", "winsorize(rank(divide(subtract(close, open), open)), 4)"))
    out.append(("cross_sectional", "rank(subtract(rank(close), rank(vwap)))"))

    # ---- Family F: Sector-relative (group_neutralize theo sector) ----
    out.append(("sector_relative", "group_neutralize(rank(ts_delta(close, 20)), sector)"))
    out.append(("sector_relative", "group_neutralize(returns, sector)"))
    out.append(("sector_relative", "group_neutralize(rank(returns), sector)"))
    out.append(("sector_relative", "rank(group_neutralize(ts_delta(close, 10), sector))"))
    out.append(("sector_relative", "group_neutralize(ts_zscore(returns, 20), sector)"))
    out.append(("sector_relative", "group_neutralize(rank(divide(subtract(high, low), close)), sector)"))
    out.append(("sector_relative", "group_neutralize(multiply(-1, ts_delta(close, 5)), sector)"))
    out.append(("sector_relative", "group_neutralize(rank(multiply(returns, volume)), sector)"))
    out.append(("sector_relative", "rank(group_neutralize(divide(close, vwap), sector))"))
    out.append(("sector_relative", "group_neutralize(ts_rank(returns, 20), sector)"))

    # ---- Bonus: wrapper-stack sâu hơn (decay/hump) để trải Sharpe + test depth ----
    out.append(("wrapper", "scale(ts_decay_linear(group_neutralize(rank(ts_delta(close, 20)), sector), 5))"))
    out.append(("wrapper", "ts_decay_linear(rank(multiply(-1, ts_delta(close, 5))), 10)"))
    out.append(("wrapper", "hump(rank(ts_delta(vwap, 10)), 0.01)"))
    out.append(("wrapper", "scale(group_neutralize(ts_zscore(returns, 20), sector))"))

    rng.shuffle(out)  # trộn thứ tự để lô sim không gom cùng family (seed cố định)
    return out


def main() -> None:
    rng = random.Random(SEED)
    raw = build_expressions(rng)

    valid: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    depth_visitor = DepthVisitor()
    field_collector = FieldCollector()

    for family, expr in raw:
        if expr in seen:
            rejected.append({"expr": expr, "family": family, "reason": "duplicate"})
            continue
        seen.add(expr)
        try:
            node = parse_expression(expr)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"expr": expr, "family": family, "reason": f"parse: {exc}"})
            continue
        depth = depth_visitor.visit(node)
        fields = field_collector.visit(node)
        ops = _collect_ops(node)
        bad_fields = fields - ALLOWED_FIELDS
        bad_ops = ops - ALLOWED_OPS
        if depth > MAX_DEPTH:
            rejected.append({"expr": expr, "family": family, "reason": f"depth {depth}>{MAX_DEPTH}"})
            continue
        if bad_fields:
            rejected.append({"expr": expr, "family": family, "reason": f"field lạ: {sorted(bad_fields)}"})
            continue
        if bad_ops:
            rejected.append({"expr": expr, "family": family, "reason": f"op ngoài giao-tập: {sorted(bad_ops)}"})
            continue
        valid.append({"expr": expr, "family": family, "depth": depth, "fields": sorted(fields)})

    manifest = {
        "seed": SEED,
        "allowed_fields": sorted(ALLOWED_FIELDS),
        "allowed_ops": sorted(ALLOWED_OPS),
        "max_depth": MAX_DEPTH,
        "n_valid": len(valid),
        "n_rejected": len(rejected),
        "valid": valid,
        "rejected": rejected,
    }
    out_path = ROOT / ".superpowers" / "sdd" / "groundtruth-manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    seed_path = ROOT / ".superpowers" / "sdd" / "groundtruth-seed.txt"
    seed_path.write_text(str(SEED), encoding="utf-8")

    from collections import Counter

    fam_counts = Counter(v["family"] for v in valid)
    print(f"VALID={len(valid)}  REJECTED={len(rejected)}  SEED={SEED}")
    print("families:", dict(fam_counts))
    print("depth range:", min(v["depth"] for v in valid), "-", max(v["depth"] for v in valid))
    for r in rejected:
        print("  REJECT:", r["reason"], "|", r["expr"])
    print("manifest ->", out_path)


if __name__ == "__main__":
    main()
