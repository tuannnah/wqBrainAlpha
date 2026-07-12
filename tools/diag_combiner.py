"""tools/diag_combiner.py — Chẩn đoán OFFLINE CombinerIdeaSource, đường THẬT sau Task 2.

Task 1 (`logs/diag_combiner_20260712.md`, re-run `logs/diag_combiner_20260713.md`) đo được
2 nguyên nhân THẬT khiến combiner luôn ra 0 combo: (1) nguồn tín hiệu con xếp theo fitness
LOCAL (ρ=0.308 với Brain — chọn toàn GP junk), (2) gate chấm self-corr với
`repo.load_pool()` 1321+ eval LOCAL đã bão hòa (giết oan combo self-corr 0.70-0.86 trong khi
Brain thật đo 0.40-0.46), cộng thêm depth-fail và tự-so là nguyên nhân phụ. Task 2
(`.superpowers/sdd/task-2-brief.md`) sửa cả 4 tầng.

Script này gọi THẲNG các hàm production thật — KHÔNG tự chế lại logic lần thứ hai (nguồn
sai lệch giữa diag và thật là chính lý do Task 1 phải viết lại kịch bản chẩn đoán):

    repo.brain_proven_signals(COMBINER_MIN_BRAIN_SHARPE)   (Fix 1: nguồn Brain-proven)
      -> backtest local từng expr (_score_one_full, không pool) để lấy PnL
      -> combine_stage(signals, score_fn_factory=..., drop_stats=...)  (Fix 2+3+4: gate pool
         = tín hiệu Brain-proven NGOÀI combo, depth pre-filter nội bộ, điểm-nộp, đếm rớt)

`drop_stats` đọc trực tiếp từ combine_stage — nếu vẫn 0 combo, biết NGAY tầng nào giết
(depth/gate/not_better/greedy_empty), không cần đoán.

KHÔNG login, KHÔNG sim Brain, KHÔNG sửa file nào trong `src/`. Config (region/universe/
delay/neutralization/decay/truncation) dựng Y HỆT `main.py::_run_closed_loop_session`.

Chạy: ./venv/Scripts/python.exe tools/diag_combiner.py
Ghi báo cáo: APPEND vào cuối `logs/diag_combiner_<YYYYMMDD>.md` (không đụng nội dung đã có
từ các lần chạy trước trong ngày/phiên khác — override qua env `DIAG_COMBINER_REPORT_PATH`
để chuyển hướng khi cần thử nghiệm).
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.operators_local  # noqa: E402,F401  nạp 27 operator local vào registry TRƯỚC parse/eval

from config.settings import settings  # noqa: E402
from config.thresholds import COMBINER_MAX_COMPONENT_DEPTH, COMBINER_MIN_BRAIN_SHARPE  # noqa: E402
from main import (  # noqa: E402  tái dùng đúng cách main.py dựng config, không tự chế lại
    _find_market_data_dir,
    _local_neutralization,
    _portfolio_config_from_opts,
)
from src.backtest.gate import local_usable  # noqa: E402
from src.data.adapters.parquet_source import ParquetSource  # noqa: E402
from src.generation.combiner import SubSignal  # noqa: E402
from src.lang.registry import default_registry  # noqa: E402
from src.pipeline.combine_stage import combine_stage  # noqa: E402
from src.pipeline.runner import _score_one_full  # noqa: E402
from src.storage.db import active_database_url, make_engine, make_session_factory, read_active_account  # noqa: E402
from src.storage.repository import MiniBrainRepository  # noqa: E402

# --- Config: khớp NGUYÊN VĂN mặc định của closed-loop thật (menu mục 5 / CLI closed-loop) ---
REGION = "USA"
UNIVERSE = "TOP3000"
DELAY = 1
NEUTRALIZATION = "MARKET"  # main.py::_run_closed_loop_session mặc định
DECAY = 4
TRUNCATION = 0.08
# --- Config combiner: khớp default trong src/generation/combiner.py + build_closed_loop ---
TAU = 0.30
N_MIN = 2
N_MAX = 4
MAX_COMBOS = 5

REPORT_PATH = Path(
    os.environ.get(
        "DIAG_COMBINER_REPORT_PATH", str(ROOT / "logs" / f"diag_combiner_{date.today():%Y%m%d}.md")
    )
)

_lines: list[str] = []


def emit(msg: str = "") -> None:
    """In ra stdout ĐỒNG THỜI gom vào buffer để ghi báo cáo markdown cuối script."""
    print(msg)
    _lines.append(msg)


def _fmt_stats(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"n={arr.size} min={arr.min():.4f} p25={np.percentile(arr, 25):.4f} "
        f"median={np.median(arr):.4f} p75={np.percentile(arr, 75):.4f} max={arr.max():.4f} "
        f"mean={arr.mean():.4f}"
    )


def _local_backtest(expr: str, cfg, data):
    """Backtest local NẾU local-usable — cùng logic `CombinerIdeaSource._local_backtest`
    (Task 2 Fix 1) vì `repo.brain_proven_signals` chỉ trả (expr, sharpe), không có PnL sẵn."""
    try:
        usable = local_usable(expr, data)
    except Exception:
        usable = True
    if not usable:
        return None
    try:
        res = _score_one_full(expr, cfg, data)
    except Exception:
        return None
    if res.pnl.size == 0:
        return None
    return res


def _score_fn_factory_for(cfg, data):
    """Đúng `CombinerIdeaSource._score_fn_factory` (Task 2 Fix 2): pool tự đánh số từ PnL
    local của các tín hiệu NGOÀI combo, KHÔNG phải `repo.load_pool()`."""

    def factory(others: list[SubSignal]):
        pool = {i: (s.dates, s.pnl) for i, s in enumerate(others)} or None

        def score(expr: str):
            return _score_one_full(expr, cfg, data, pool)

        return score

    return factory


def main() -> int:
    emit("\n---\n")
    emit(f"# Chẩn đoán CombinerIdeaSource — re-run sau Task 2 (4 fix)\n")
    emit(f"Chạy lúc: {date.today():%Y-%m-%d}\n")
    emit(
        "> Gọi THẲNG `combine_stage` thật với `score_fn_factory` + `drop_stats` (không tự chế "
        "lại logic tầng 2/3/4 lần thứ hai như bản diag Task 1) — nguồn tầng 0 đổi sang "
        "`repo.brain_proven_signals` (Fix 1). Script này CHỈ tái hiện nhánh tín hiệu \"db\" "
        "(Brain-proven); production `CombinerIdeaSource.next_batch()` còn trộn thêm tín hiệu "
        "\"run\" phát sinh ngay trong batch hiện tại — không phải replay chính xác 1-1.\n"
    )

    # ------------------------------------------------------------------
    # Bước 0: dựng DB/data/config Y HỆT closed-loop thật
    # ------------------------------------------------------------------
    emit("## Bước 0 — Dựng môi trường (DB / panel / config)\n")
    email = read_active_account()
    db_url = active_database_url()
    emit(f"- Tài khoản active (`.wq_account`): `{email or '(rỗng)'}`")
    emit(f"- DB URL: `{db_url}`")
    engine = make_engine(db_url)
    session_factory = make_session_factory(engine)
    repo = MiniBrainRepository(session_factory)

    market_data_dir = _find_market_data_dir()
    if market_data_dir is None:
        emit(
            f"- **BLOCKED**: không tìm thấy thư mục MarketData nào (đã thử "
            f"`{settings.market_data_dir}` và quét `data/*/returns.parquet`)."
        )
        _write_report()
        return 1
    emit(f"- MarketData dir: `{market_data_dir}`")

    try:
        data = ParquetSource(market_data_dir).load("1900-01-01", "2999-12-31", UNIVERSE)
    except (FileNotFoundError, AssertionError, OSError) as exc:
        emit(f"- **BLOCKED**: không load được MarketData: {exc}")
        _write_report()
        return 1

    local_neut = _local_neutralization(NEUTRALIZATION, set(data.groups.keys()))
    cfg = _portfolio_config_from_opts(local_neut, DECAY, TRUNCATION, DELAY)
    registry = default_registry()
    emit(
        f"- Config local gate: neutralization={local_neut} (yêu cầu={NEUTRALIZATION}), "
        f"decay={DECAY}, truncation={TRUNCATION}, delay={DELAY}, region={REGION}, "
        f"universe={UNIVERSE}"
    )
    emit(f"- Panel: {data.dates.size} ngày, groups có sẵn = {sorted(data.groups.keys())}")

    # ------------------------------------------------------------------
    # Tầng 0 (Fix 1): repo.brain_proven_signals — thay good_signals_for_combine (fitness local)
    # ------------------------------------------------------------------
    emit(f"\n## Tầng 0 — `repo.brain_proven_signals(min_sharpe={COMBINER_MIN_BRAIN_SHARPE})`\n")
    raw = repo.brain_proven_signals(COMBINER_MIN_BRAIN_SHARPE)
    emit(f"- Số expr Brain-proven (sharpe >= {COMBINER_MIN_BRAIN_SHARPE}, mọi status): **{len(raw)}**")
    if not raw:
        emit(
            "\n### Kết luận\n\n"
            f"**KẾT LUẬN SỚM**: chưa có `BrainSimLinkModel` nào sharpe >= {COMBINER_MIN_BRAIN_SHARPE} "
            "trong DB -> combiner 0 combo vì KHÔNG CÓ ĐẦU VÀO Brain-proven (cần chạy closed-loop "
            "thật để tích luỹ Brain sim trước khi combiner có gì để ghép)."
        )
        _write_report()
        return 0

    emit("\n10 expr Brain-proven đầu (sort sharpe giảm dần):\n")
    emit("| # | sharpe Brain | expr |")
    emit("|---|---|---|")
    for i, (expr, sharpe) in enumerate(raw[:10], start=1):
        emit(f"| {i} | {sharpe:.4f} | `{expr}` |")

    signals: list[SubSignal] = []
    n_not_usable = 0
    for expr, sharpe in raw:
        res = _local_backtest(expr, cfg, data)
        if res is None:
            n_not_usable += 1
            continue
        signals.append(SubSignal(expr, res.pnl, res.dates, sharpe, source="db"))
    emit(
        f"\n- Backtest local thành công (local-usable + có PnL): **{len(signals)}**/{len(raw)} "
        f"({n_not_usable} bị loại vì không local-usable hoặc backtest lỗi/0 PnL)."
    )
    if signals:
        emit(f"- Phân bố sharpe Brain (= score xếp seed cho greedy): {_fmt_stats([s.score for s in signals])}")

    if len(signals) < N_MIN:
        emit(
            "\n### Kết luận\n\n"
            f"**KẾT LUẬN**: chỉ {len(signals)} tín hiệu local-usable (< N_MIN={N_MIN}) -> "
            "combiner 0 combo vì THIẾU ĐẦU VÀO local-usable, không phải do gate/dominance."
        )
        _write_report()
        return 0

    # ------------------------------------------------------------------
    # Tầng 1-4 (Fix 2+3+4): gọi THẲNG combine_stage thật, đọc drop_stats
    # ------------------------------------------------------------------
    emit(
        f"\n## Tầng 1-4 — `combine_stage` thật (tau={TAU}, n_min={N_MIN}, n_max={N_MAX}, "
        f"max_combos={MAX_COMBOS}, COMBINER_MAX_COMPONENT_DEPTH={COMBINER_MAX_COMPONENT_DEPTH})\n"
    )
    factory = _score_fn_factory_for(cfg, data)
    fallback_score_fn = factory([])  # score_fn cũ — KHÔNG dùng thật (factory ưu tiên), chỉ để khớp chữ ký
    drop_stats: dict[str, int] = {}
    out = combine_stage(
        signals, fallback_score_fn, tau=TAU, n_min=N_MIN, n_max=N_MAX, max_combos=MAX_COMBOS,
        registry=registry, score_fn_factory=factory, drop_stats=drop_stats,
    )

    emit(f"- `drop_stats`: `{drop_stats}`")
    emit(f"- **Số combo QUA HẾT 4 tầng: {len(out)}**")
    for i, c in enumerate(out, start=1):
        emit(
            f"  - combo #{i}: fitness={c.metrics.fitness:.4f} sharpe={c.metrics.sharpe:.4f} "
            f"expr=`{c.expr}`"
        )

    emit("\n## Kết luận (re-run sau Task 2)\n")
    if out:
        emit(
            f"**{len(out)} combo QUA HẾT 4 tầng** — combiner đã sinh combo thành công trên đường "
            "thật (`brain_proven_signals` + `score_fn_factory` + depth pre-filter + điểm-nộp). "
            "Xem chi tiết combo ở trên."
        )
    else:
        emit(
            "**VẪN 0 combo.** `drop_stats` chỉ rõ CHÍNH XÁC tầng nào giết hết combo thô: "
            f"depth={drop_stats.get('depth', 0)}, gate={drop_stats.get('gate', 0)}, "
            f"not_better={drop_stats.get('not_better', 0)}, "
            f"greedy_empty={drop_stats.get('greedy_empty', 0)}. "
            + (
                "`greedy_empty` > 0 nghĩa là ngay bước khử tương quan (tau={:.2f}) giữa "
                "{} tín hiệu Brain-proven local-usable đã không ghép nổi >= {} tín hiệu nào "
                "(khả năng: tập Brain-proven hiện tại quá ít hoặc quá tương quan lẫn nhau — "
                "cần tích luỹ thêm Brain sim đa dạng họ hơn)."
                .format(TAU, len(signals), N_MIN)
                if drop_stats.get("greedy_empty", 0) > 0
                else ""
            )
        )

    _write_report()
    return 0


def _write_report() -> None:
    """APPEND vào cuối báo cáo (khác bản Task 1: ghi đè) — không xoá bằng chứng chẩn đoán
    gốc đã có trong file cùng ngày khi re-run kiểm chứng fix."""
    REPORT_PATH.parent.mkdir(exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n[đã APPEND báo cáo] {REPORT_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
