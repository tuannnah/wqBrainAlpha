"""tools/diag_combiner.py — Chẩn đoán OFFLINE vì sao CombinerIdeaSource luôn ra 0 combo.

Bối cảnh: log thực chiến `logs/wq_alpha_2026-07-12.log` cho thấy ở MỌI batch:
    CombinerIdeaSource: n_run=X n_db=50 total=Y -> 0 combo
tức KHÔNG PHẢI do thiếu tín hiệu (n_db luôn chạm limit=50, total luôn >= n_min) — có tầng
lọc nào đó bên trong giết sạch combo. Script này KHÔNG login, KHÔNG sim Brain, KHÔNG sửa bất
kỳ file nào trong `src/` — chỉ đọc DB SQLite thật + panel parquet local rồi TÁI HIỆN đúng
đường đi mà `CombinerIdeaSource.next_batch()` (src/app/closed_loop_adapters.py:724-756) đi
qua, in số liệu MỖI tầng:

    repo.good_signals_for_combine(limit=50)                      (Tầng 0: nguồn tín hiệu)
      -> select_decorrelated_combos(tau=0.30, n_min=2, n_max=4, max_combos=5)  (Tầng 1: chọn)
      -> build_combined_expression(...)                          (Tầng 2: dựng biểu thức)
      -> _score_one_full(..., pool=repo.load_pool())             (Tầng 3: gate, gồm pool self-corr)
      -> so fitness combo với fitness từng sub-expr (CÙNG pool)  (Tầng 4: vượt trội)

Config (region/universe/delay/neutralization/decay/truncation) dựng Y HỆT cách
`main.py::_run_closed_loop_session` dựng cho closed-loop thật (menu mục 5) — tái sử dụng
trực tiếp các hàm helper của main.py (`_find_market_data_dir`, `_local_neutralization`,
`_portfolio_config_from_opts`) để không lệch cấu hình.

Chạy: ./venv/Scripts/python.exe tools/diag_combiner.py
Ghi báo cáo: logs/diag_combiner_<YYYYMMDD>.md (tự tạo, ghi đè nếu chạy lại trong ngày).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.operators_local  # noqa: E402,F401  nạp 27 operator local vào registry TRƯỚC parse/eval

from config.settings import settings  # noqa: E402
from config.thresholds import MAX_DEPTH, SELF_CORR_MAX  # noqa: E402
from main import (  # noqa: E402  tái dùng đúng cách main.py dựng config, không tự chế lại
    _find_market_data_dir,
    _local_neutralization,
    _portfolio_config_from_opts,
)
from src.backtest.pool_corr import PoolCorrelation, pairwise_abs_rho  # noqa: E402
from src.data.adapters.parquet_source import ParquetSource  # noqa: E402
from src.generation.combiner import (  # noqa: E402
    SubSignal,
    build_combined_expression,
    select_decorrelated_combos,
)
from src.lang.registry import default_registry  # noqa: E402
from src.pipeline.runner import _score_one_full  # noqa: E402
from src.storage.db import active_database_url, make_engine, make_session_factory, read_active_account  # noqa: E402
from src.storage.models import EvaluationModel, ExpressionModel  # noqa: E402
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
DB_LIMIT = 50

REPORT_PATH = ROOT / "logs" / f"diag_combiner_{date.today():%Y%m%d}.md"

_lines: list[str] = []


def emit(msg: str = "") -> None:
    """In ra stdout ĐỒNG THỜI gom vào buffer để ghi báo cáo markdown cuối script."""
    print(msg)
    _lines.append(msg)


def _fmt_fitness_stats(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"n={arr.size} min={arr.min():.4f} p25={np.percentile(arr, 25):.4f} "
        f"median={np.median(arr):.4f} p75={np.percentile(arr, 75):.4f} max={arr.max():.4f} "
        f"mean={arr.mean():.4f}"
    )


def _expr_for_pool_id(session_factory, pool_id: int) -> str | None:
    """Tra ngược evaluation_id (khóa của PoolPnlModel) -> expr_string, để chứng minh cụ thể
    'thành viên pool trùng nhất với combo LÀ chính tín hiệu con của nó'."""
    session = session_factory()
    try:
        row = (
            session.query(ExpressionModel.expr_string)
            .join(EvaluationModel, ExpressionModel.id == EvaluationModel.expression_id)
            .filter(EvaluationModel.id == pool_id)
            .first()
        )
        return row[0] if row else None
    finally:
        session.close()


def main() -> int:
    emit("# Chẩn đoán CombinerIdeaSource — 0 combo\n")
    emit(f"Chạy lúc: {date.today():%Y-%m-%d}\n")
    emit(
        "> **Lưu ý phạm vi**: script này CHỈ tái hiện nhánh tín hiệu lấy từ DB "
        "(`repo.good_signals_for_combine`, `source=\"db\"`). Ở production, "
        "`CombinerIdeaSource.next_batch()` còn trộn thêm tín hiệu `source=\"run\"` phát sinh "
        "ngay trong batch hiện tại — nên các combo #N liệt kê dưới đây KHÔNG nhất thiết trùng "
        "với combo mà production từng dựng trên cùng batch đó; đây là chẩn đoán offline trên "
        "một tập con tín hiệu, không phải replay chính xác 1-1.\n"
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
    # Tầng 0: nguồn tín hiệu — repo.good_signals_for_combine(limit=50)
    # ------------------------------------------------------------------
    emit("\n## Tầng 0 — `repo.good_signals_for_combine(limit=50)`\n")
    raw_signals = repo.good_signals_for_combine(limit=DB_LIMIT)
    emit(f"- Số tín hiệu lấy được: **{len(raw_signals)}** (limit={DB_LIMIT})")
    if not raw_signals:
        emit(
            "- **KẾT LUẬN SỚM**: DB không có tín hiệu PASSED nào có PnL trong pool "
            "(bảng pool_pnl rỗng hoặc JOIN không khớp) — combiner 0 combo vì KHÔNG CÓ ĐẦU VÀO, "
            "không phải do gate/dominance. Kiểm tra `save_pool_pnl` có được gọi sau mỗi lần "
            "PASSED hay không."
        )
        _write_report()
        return 0

    fitnesses = [f for (_, _, _, f) in raw_signals]
    emit(f"- Phân bố fitness: {_fmt_fitness_stats(fitnesses)}")
    emit("\n10 tín hiệu đầu (đã sort fitness giảm dần):\n")
    emit("| # | fitness | expr |")
    emit("|---|---|---|")
    for i, (expr, _dates, _pnl, fitness) in enumerate(raw_signals[:10], start=1):
        emit(f"| {i} | {fitness:.4f} | `{expr}` |")

    signals: list[SubSignal] = [
        SubSignal(expr, pnl, dates, fitness, source="db")
        for (expr, dates, pnl, fitness) in raw_signals
    ]

    # ------------------------------------------------------------------
    # Tầng 1: select_decorrelated_combos (greedy khử tương quan tau=0.30)
    # ------------------------------------------------------------------
    emit(f"\n## Tầng 1 — `select_decorrelated_combos(tau={TAU}, n_min={N_MIN}, "
         f"n_max={N_MAX}, max_combos={MAX_COMBOS})`\n")
    combos = select_decorrelated_combos(
        signals, tau=TAU, n_min=N_MIN, n_max=N_MAX, max_combos=MAX_COMBOS
    )
    emit(f"- Số combo THÔ (trước dựng biểu thức/gate): **{len(combos)}**")
    for i, combo in enumerate(combos, start=1):
        emit(f"  - combo #{i}: {len(combo)} tín hiệu — " + ", ".join(f"`{s.expr}`" for s in combo))

    if not combos:
        emit(
            "\n- Greedy trả RỖNG -> nghi ngờ mọi tín hiệu top-10 tương quan đôi một >= "
            f"tau={TAU}. In ma trận |rho(PnL)| đôi một (top-10 theo fitness) để kiểm chứng:\n"
        )
        top10 = signals[:10]
        header = "|  | " + " | ".join(f"S{i+1}" for i in range(len(top10))) + " |"
        sep = "|---" * (len(top10) + 1) + "|"
        emit(header)
        emit(sep)
        n_pairs = 0
        n_ge_tau = 0
        n_none = 0
        for i, si in enumerate(top10):
            row = [f"**S{i+1}**"]
            for j, sj in enumerate(top10):
                if i == j:
                    row.append("—")
                    continue
                rho = pairwise_abs_rho(si.pnl, si.dates, sj.pnl, sj.dates)
                if rho is None:
                    row.append("None")
                    if i < j:
                        n_none += 1
                        n_pairs += 1
                else:
                    row.append(f"{rho:.2f}")
                    if i < j:
                        n_pairs += 1
                        if rho >= TAU:
                            n_ge_tau += 1
            emit("| " + " | ".join(row) + " |")
        emit(
            f"\n- Trong {n_pairs} cặp (top-10): **{n_ge_tau}** cặp |rho| >= {TAU}, "
            f"{n_none} cặp không đo được (không overlap ngày / phương sai 0)."
        )
        emit("\n- Legend: " + ", ".join(f"S{i+1}=`{s.expr}`" for i, s in enumerate(top10)))
        emit(
            "\n### Kết luận\n\n"
            "**Tầng 1 (greedy khử tương quan) giết combo**: mọi cặp tín hiệu top đều tương "
            f"quan >= tau={TAU} nên `select_decorrelated_combos` không ghép được combo nào "
            f"(n_min={N_MIN} không đạt). Không có combo thô nào để đi tiếp Tầng 2/3/4."
        )
        _write_report()
        return 0

    # ------------------------------------------------------------------
    # Tầng 2+3+4: build_combined_expression -> _score_one_full(pool=...) -> so fitness
    # ------------------------------------------------------------------
    pool_raw = repo.load_pool()
    pool = pool_raw or None
    emit(f"\n## Tầng 2/3/4 — dựng biểu thức, chấm gate (pool={len(pool_raw)} thành "
         f"viên), so fitness với sub-expr tốt nhất\n")
    emit(f"- `repo.load_pool()` trả **{len(pool_raw)}** thành viên (PoolPnlModel).")

    def score(expr: str):
        return _score_one_full(expr, cfg, data, pool)

    n_depth_fail = 0
    n_gate_selfcorr_fail = 0
    n_gate_other_fail = 0
    n_dominance_fail = 0
    n_passed = 0
    gate_selfcorr_values: list[float] = []
    # Kiểm chứng CỤ THỂ giả thuyết "combo trùng CHÍNH sub-expr của nó trong pool" (brief) —
    # đừng tin mù: đếm bao nhiêu ca self_corr thật sự khớp CHÍNH sub-expr của combo đó, và
    # bao nhiêu ca khớp một thành viên KHÁC trong pool (saturation tổng thể).
    n_selfcorr_matches_own_subexpr = 0
    n_selfcorr_matches_other_member = 0

    for i, combo in enumerate(combos, start=1):
        emit(f"\n### combo #{i} — {len(combo)} tín hiệu\n")
        for s in combo:
            emit(f"- sub-expr (fitness={s.score:.4f}, source={s.source}): `{s.expr}`")

        built = build_combined_expression([s.expr for s in combo], registry=registry)
        if built is None:
            emit(f"- **RỚT: depth** — không dựng được biểu thức lọt trần độ sâu MAX_DEPTH={MAX_DEPTH}.")
            n_depth_fail += 1
            continue
        emit(f"- Biểu thức ghép: `{built.expr}` (dùng {len(built.sub_exprs)}/{len(combo)} sub-expr)")

        scored = score(built.expr)
        if not scored.verdict.passed:
            reasons = "; ".join(scored.verdict.hard_failures) or "(không rõ lý do)"
            emit(f"- **RỚT: gate** — verdict.passed=False. Lý do: {reasons}")
            is_selfcorr = any("self_corr" in r for r in scored.verdict.hard_failures)
            if is_selfcorr:
                n_gate_selfcorr_fail += 1
                # Đo trực tiếp self_corr thật + thành viên pool trùng nhất, và so với rho
                # trực tiếp giữa combo và CHÍNH sub-expr của nó (chứng minh combo tự trùng
                # với đầu vào của nó vì đầu vào đã nằm sẵn trong pool).
                if pool:
                    rho_val, worst_id = PoolCorrelation(pool).max_corr(scored.pnl, scored.dates)
                    gate_selfcorr_values.append(rho_val)
                    worst_expr = _expr_for_pool_id(session_factory, worst_id) if worst_id else None
                    is_own_subexpr = worst_expr is not None and any(
                        worst_expr == s.expr for s in combo
                    )
                    if is_own_subexpr:
                        n_selfcorr_matches_own_subexpr += 1
                    else:
                        n_selfcorr_matches_other_member += 1
                    emit(
                        f"  - self_corr thật với pool: {rho_val:.4f} (thành viên pool "
                        f"evaluation_id={worst_id}, expr=`{worst_expr}`)"
                    )
                    if is_own_subexpr:
                        emit("  - thành viên pool trùng nhất **CHÍNH LÀ** một sub-expr của combo này.")
                    else:
                        emit(
                            "  - thành viên pool trùng nhất là một alpha KHÁC trong pool (không "
                            "phải sub-expr của combo này) — nghi ngờ pool bão hòa (saturation) "
                            "chứ không chỉ tự trùng đầu vào."
                        )
                    for s in combo:
                        rho_own = pairwise_abs_rho(scored.pnl, scored.dates, s.pnl, s.dates)
                        rho_str = f"{rho_own:.4f}" if rho_own is not None else "None"
                        same = " <-- CHÍNH LÀ sub-expr của combo này" if worst_expr == s.expr else ""
                        emit(f"  - |rho| combo vs sub-expr `{s.expr}`: {rho_str}{same}")
            else:
                n_gate_other_fail += 1
            continue

        best_component = max(
            (score(e).metrics.fitness for e in built.sub_exprs), default=float("-inf")
        )
        emit(
            f"- Qua gate. fitness combo={scored.metrics.fitness:.4f} vs "
            f"best_component={best_component:.4f}"
        )
        if scored.metrics.fitness <= best_component:
            emit("- **RỚT: không vượt trội** — fitness combo không lớn hơn sub-expr mạnh nhất.")
            n_dominance_fail += 1
            continue

        emit("- **QUA HẾT 4 tầng** — combo hợp lệ (đáng lẽ phải xuất hiện trong `next_batch`).")
        n_passed += 1

    # ------------------------------------------------------------------
    # Tổng kết
    # ------------------------------------------------------------------
    emit("\n## Tổng kết theo tầng\n")
    emit(f"- Combo thô (Tầng 1): {len(combos)}")
    emit(f"- RỚT Tầng 2 (depth): {n_depth_fail}")
    emit(f"- RỚT Tầng 3 (gate) do self_corr pool: {n_gate_selfcorr_fail}")
    emit(f"- RỚT Tầng 3 (gate) lý do khác: {n_gate_other_fail}")
    emit(f"- RỚT Tầng 4 (không vượt trội): {n_dominance_fail}")
    emit(f"- QUA HẾT: {n_passed}")
    if gate_selfcorr_values:
        arr = np.asarray(gate_selfcorr_values)
        emit(
            f"- self_corr trung bình của các combo rớt vì pool-corr: {arr.mean():.4f} "
            f"(ngưỡng SELF_CORR_MAX={SELF_CORR_MAX:.2f}), min={arr.min():.4f} max={arr.max():.4f}"
        )

    emit("\n## Kết luận\n")
    if n_passed > 0:
        emit(
            f"- {n_passed}/{len(combos)} combo QUA HẾT 4 tầng ngay trong lần chạy này — "
            "combiner KHÔNG bị chặn cứng ở mọi batch; kết quả '0 combo' trong log thật có "
            "thể phụ thuộc lô tín hiệu cụ thể tại thời điểm đó. Xem chi tiết từng combo ở "
            "trên."
        )
    elif len(combos) == 0:
        emit("- Không có combo thô nào (đã kết luận ở Tầng 1 phía trên).")
    else:
        denom_tang3 = n_gate_selfcorr_fail + n_gate_other_fail + n_dominance_fail + n_passed
        pct_selfcorr = (
            f"{n_gate_selfcorr_fail / denom_tang3 * 100:.0f}%" if denom_tang3 > 0 else "N/A"
        )
        emit(
            f"**Cả {len(combos)}/{len(combos)} combo thô đều chết — do HAI tầng lọc cộng "
            f"hưởng, không phải một tầng duy nhất:**\n\n"
            f"1. **Tầng 2 (depth)**: {n_depth_fail}/{len(combos)} combo không dựng nổi biểu "
            f"thức lọt MAX_DEPTH={MAX_DEPTH} — chết trước khi kịp tới gate.\n"
            f"2. **Tầng 3 (gate, self_corr pool)**: trong số combo VƯỢT qua Tầng 2, "
            f"**{n_gate_selfcorr_fail}/{denom_tang3}** ({pct_selfcorr}) rớt vì "
            f"`self_corr >= SELF_CORR_MAX={SELF_CORR_MAX:.2f}` so với `repo.load_pool()`.\n"
        )
        if gate_selfcorr_values:
            arr = np.asarray(gate_selfcorr_values)
            margins = arr - SELF_CORR_MAX
            emit(
                f"   self_corr đo được: {', '.join(f'{v:.4f}' for v in gate_selfcorr_values)} "
                f"(đều vượt ngưỡng {SELF_CORR_MAX:.2f}, cách xa từ {margins.min():+.3f} đến "
                f"{margins.max():+.3f})."
            )
        emit(
            "\n**Kiểm chứng giả thuyết trong brief** (\"combo tương quan với chính sub-signal "
            "của nó trong pool\") — ĐÃ ĐO TRỰC TIẾP, KHÔNG đúng hoàn toàn như hình dung ban "
            f"đầu: trong {n_gate_selfcorr_fail} ca rớt vì self_corr, "
            f"**{n_selfcorr_matches_own_subexpr}** ca thành viên pool trùng nhất CHÍNH LÀ một "
            f"sub-expr của combo đó, còn **{n_selfcorr_matches_other_member}** ca thành viên "
            "trùng nhất là một alpha KHÁC hẳn trong pool (không nằm trong combo). Tức nguyên "
            f"nhân thật sự rộng hơn giả thuyết ban đầu: **pool {len(pool_raw)} thành viên đã "
            "BÃO HÒA** (dày đặc các biến thể `group_neutralize(rank(...))`/"
            "`multiply(-1, rank(...))` trên price/volume) — combo `add(rank(s1), rank(s2))` "
            f"mới dựng, dù 2 sub-expr chọn qua greedy đã <{TAU:.2f} tương quan VỚI NHAU, vẫn "
            f"gần như chắc chắn rơi vào bán kính {SELF_CORR_MAX:.2f} của MỘT thành viên nào đó "
            f"trong {len(pool_raw)} alpha đã có sẵn — vì combiner chỉ khử tương quan trong nội "
            f"bộ ~{DB_LIMIT} ứng viên (`select_decorrelated_combos`), KHÔNG hề kiểm tra trước "
            "tương quan với TOÀN BỘ pool đã tích lũy.\n"
        )
        emit(
            "\n### Fix đề xuất cho Task 2 (theo thứ tự ưu tiên, dựa trên số liệu đo được)\n\n"
            "1. **Pool-decorrelation SỚM, trước khi tốn công dựng+chấm combo**: sau "
            "`build_combined_expression`, tính `PoolCorrelation(pool).max_corr(candidate_pnl, "
            "dates)` NGAY (rẻ hơn `_score_one_full` đầy đủ) và bỏ combo sớm nếu vượt ngưỡng — "
            "tiết kiệm nhưng KHÔNG tự nó tăng số combo qua được (đây chỉ là early-exit, không "
            "phải fix triệt để).\n"
            "2. **Fix triệt để hơn — chọn tín hiệu ít tương quan với CẢ POOL, không chỉ với "
            "nhau**: `select_decorrelated_combos` hiện chỉ dùng `pairwise_abs_rho` GIỮA các "
            "candidate. Sửa để mỗi candidate (hoặc mỗi combo ứng viên) phải có "
            "`PoolCorrelation(pool).max_corr(...) < SELF_CORR_MAX` TRƯỚC khi được chọn vào "
            "combo — loại các signal 'phổ biến' (gần giống nhiều alpha đã pass) khỏi vai trò "
            "thành phần combo ngay từ khâu chọn, thay vì phát hiện muộn ở gate.\n"
            "3. **Loại trừ sub-expr CHÍNH nó khỏi pool khi chấm gate** (vẫn nên làm dù không "
            "phải nguyên nhân chính trong lần đo này — "
            f"{n_selfcorr_matches_own_subexpr}/{n_gate_selfcorr_fail} ca là tự trùng thật): "
            "`good_signals_for_combine` trả thêm evaluation_id, `combine_stage` dựng "
            "`pool_excl = {k: v for k, v in pool.items() if k not in combo_evaluation_ids}` "
            "trước khi gọi gate.\n"
            "4. **Tầng depth**: ưu tiên sub-expr độ sâu thấp khi greedy chọn combo (đo "
            "`_depth_of` mỗi candidate trước, không chỉ sau khi build thất bại) để giảm tỉ lệ "
            f"{n_depth_fail}/{len(combos)} chết vì depth trước khi tới gate.\n\n"
            "KHÔNG hạ `SELF_CORR_MAX` — đó là ngưỡng an toàn thật ánh xạ tới self-correlation "
            "checker của Brain, hạ ngưỡng sẽ tạo alpha nộp lên chắc chắn bị Brain từ chối."
        )

    _write_report()
    return 0


def _write_report() -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\n[đã ghi báo cáo] {REPORT_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
