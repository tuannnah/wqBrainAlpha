"""ClosedLoop — orchestrator vòng kín AI + MiniBrain (thuần logic, network-agnostic).

Lặp: lấy ý tưởng (GP→short-list, qua `idea_source`) → với mỗi ý tưởng gọi `refiner.
refine_and_sim` (bọc RefinementLoop thật ở phase wiring) → ghi kết quả SIM qua
`repo.record_brain_sim` (cầu DB Phase 1) → tránh trùng → dừng gọn khi Brain hết quota
(`QuotaExhausted`) hoặc cạn ý tưởng.

Dependency rule B1: KHÔNG import `src.llm`/`src.gp`/`src.simulation` — mọi dependency
injected qua Protocol structural; việc dựng cụ thể nằm ở `main.py`/adapter."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from loguru import logger

from config.thresholds import FRONTIER_MIN_FRACTION, FRONTIER_SATURATION_K, SELF_CORR_MAX
from src.calibration.stats import spearman
from src.pipeline.shortlist import ShortlistCandidate
from src.storage.repository import MiniBrainRepository

# WS3 T3.1/T3.2 (.superpowers/sdd/20260719/task-3-brief.md): 2 nhãn family THUẦN price/volume
# theo đúng từ vựng `classify_family` (src/reporting/diagnostics.py) — lặp lại CHUỖI HẰNG (không
# import module đó) để giữ B1: module này KHÔNG phụ thuộc cụ thể vào tầng phân loại family, chỉ
# nhận family đã suy sẵn qua `family_fn` injected (như mọi nơi khác trong file này). "pv_reversal"
# và "momentum" là 2 family DUY NHẤT mà `classify_family` suy được CHỈ TỪ field close/open/high/
# low/volume/vwap trong panel local 6-field — đây chính xác là tập family "bão hoà vì panel local
# nghèo field" mà T3.1 muốn đảm bảo không chiếm trọn batch.
_PV_FAMILY_LABELS: frozenset[str] = frozenset({"pv_reversal", "momentum"})


def compute_frontier_topup(total: int, non_pv: int, min_fraction: float, available: int) -> int:
    """T3.1: số candidate CẦN rút từ `frontier_reserve` để batch (kích thước `total`, đã có
    `non_pv` candidate non-PV) đạt tỉ lệ non-PV tối thiểu `min_fraction` SAU KHI bổ sung, giới
    hạn bởi `available` (KHÔNG chặn batch khi reserve không đủ — thiếu thì dùng hết những gì
    có, trả về đúng `available`).

    Suy từ (non_pv + k) / (total + k) >= min_fraction (min_fraction < 1)
        => k >= (min_fraction*total - non_pv) / (1 - min_fraction).
    Batch rỗng/min_fraction<=0/đã đạt sàn sẵn -> 0 (không cần bổ sung gì)."""
    if total <= 0 or min_fraction <= 0.0 or non_pv >= min_fraction * total:
        return 0
    needed = total if min_fraction >= 1.0 else math.ceil(
        (min_fraction * total - non_pv) / (1.0 - min_fraction)
    )
    return max(0, min(needed, available))


def compute_family_pool_share(families: list[str]) -> dict[str, float]:
    """T3.2: tỉ lệ mỗi family trong `families` (mỗi phần tử = family đã classify sẵn của 1
    alpha trong pool hiện tại). Rỗng -> {} (không family nào bão hoà theo nghĩa nào cả)."""
    if not families:
        return {}
    counts: dict[str, int] = {}
    for f in families:
        counts[f] = counts.get(f, 0) + 1
    total = len(families)
    return {f: n / total for f, n in counts.items()}


def rotate_reserve_by_saturation(
    reserve: "deque[ShortlistCandidate]", family_fn, family_share: dict[str, float],
    threshold: float,
) -> None:
    """T3.2: đẩy candidate trong `reserve` có family chiếm > `threshold` trong pool (theo
    `family_share`, từ `compute_family_pool_share`) xuống CUỐI hàng đợi — giữ nguyên thứ tự
    tương đối trong từng nhóm (giữ/đẩy), KHÔNG xoá candidate nào (chỉ hạ ưu tiên, tránh tiếp
    tục đào family đã bão hoà; family còn 1 cơ hội khi các family khác cạn trước)."""
    if not reserve:
        return
    keep: list[ShortlistCandidate] = []
    demote: list[ShortlistCandidate] = []
    for cand in reserve:
        share = family_share.get(family_fn(cand.expr), 0.0)
        (demote if share > threshold else keep).append(cand)
    reserve.clear()
    reserve.extend(keep)
    reserve.extend(demote)


def _read_pool_exprs(repo: object) -> list[str]:
    """T3.2: đọc expr các alpha ĐÃ PASS trong pool hiện tại — "pool" theo nghĩa self-corr/
    submission của dự án (alpha đã pass mới thật sự cạnh tranh vị trí trong pool). Guard
    getattr+callable: repo fake/cũ thiếu `load_brain_sims` vẫn chạy được, coi như pool rỗng
    (tương thích ngược, cùng pattern `submit_ready_alphas` ở `_report` trong `run()`)."""
    fn = getattr(repo, "load_brain_sims", None)
    if not callable(fn):
        return []
    try:
        rows = fn()
    except Exception:  # noqa: BLE001 - đọc pool là phụ, không được phá vòng kín chính
        return []
    return [r.expr_string for r in rows if getattr(r, "status", None) == "passed"]


def _short(expr: str, n: int = 70) -> str:
    """Rút gọn biểu thức cho log 1 dòng (bỏ khoảng trắng thừa, cắt đuôi)."""
    s = " ".join(str(expr).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt(x: float | None) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def compute_avoided_hashes(repo, dedup_key_fn) -> set[str]:
    """UNION 3 tập avoid-list bền: `avoided_hashes()` (BrainSimLinkModel status='failed',
    hash SAU tune) ∪ `avoided_hashes_original()` (TriedHashModel, hash GỐC pre-tune) ∪
    dedup(`avoided_exprs()`) (chuỗi thô, qua `dedup_key_fn` để cùng không gian key). Đây CHÍNH
    LÀ tập `seen` mà `ClosedLoop.run` dựng ở đầu phiên — tách thành hàm dùng chung để
    `build_closed_loop` lọc core tại NGUỒN (Curated/AltDataIdeaSource) đúng KHỚP với tập run()
    dùng để chặn refine trùng (Finding #2 review: trước đây build_closed_loop chỉ nạp
    avoided_hashes_original(), bỏ lọt 2 tập kia -> core đã Brain-sim-fail phiên trước (chỉ có
    trong avoided_hashes() hoặc avoided_exprs()) lọt qua lọc nguồn, đốt quota thật rồi mới bị
    `seen` ở run() chặn — không còn dấu vết vì đã sim thật). Guard getattr+callable: repo fake/
    cũ thiếu method vẫn chạy được (tương thích ngược)."""
    seen: set[str] = set()
    avoided_hashes_fn = getattr(repo, "avoided_hashes", None)
    if callable(avoided_hashes_fn):
        seen |= set(avoided_hashes_fn())
    avoided_hashes_original_fn = getattr(repo, "avoided_hashes_original", None)
    if callable(avoided_hashes_original_fn):
        seen |= set(avoided_hashes_original_fn())
    avoided_exprs_fn = getattr(repo, "avoided_exprs", None)
    if callable(avoided_exprs_fn):
        seen |= {dedup_key_fn(e) for e in avoided_exprs_fn()}
    return seen


class QuotaExhausted(Exception):
    """Refiner ném khi Brain hết quota SIM — ClosedLoop dừng vòng gọn, persist mọi thứ."""


@dataclass(frozen=True, slots=True)
class IdeaOutcome:
    """Kết quả refine+sim một ý tưởng (refiner-protocol trả về)."""

    expr: str
    canonical_hash: str
    passed: bool
    wq_alpha_id: str | None
    sharpe: float | None
    fitness: float | None
    turnover: float | None
    self_corr: float | None
    sims_used: int
    stop_reason: str
    # Cờ Power Pool (docs Brain): Sharpe>=1.0, <=8 operator, <=3 field (trừ grouping),
    # self_corr<=0.5. Default False để tương thích ngược với nơi tạo IdeaOutcome cũ.
    power_pool_eligible: bool = False
    # Settings Brain thật đã dùng khi sim (dict từ SimConfig.to_settings) — None nếu ý tưởng bị
    # gate local chặn (0 sim). Nguồn ý tưởng (alt_data/gp_local_tuner...) để soi độ lặp công thức.
    sim_settings: dict | None = None
    source: str | None = None
    # Finding #4 (review): `source` ở trên là NHÃN ĐƯỜNG XỬ LÝ (vd mọi thứ qua LocalTunerRefiner
    # đều ra "gp_local_tuner" bất kể candidate đến từ đâu) — KHÔNG đo được tiêu chí nghiệm thu
    # "≥60% sim thuộc seed/hypothesis/combiner". `origin` là NHÃN NGUỒN Ý TƯỞNG GỐC của candidate
    # (curated/gp/alt_data/combiner — xem ShortlistCandidate.origin), do `ClosedLoop.run` stamp
    # vào outcome (composition root; refiner không biết origin). Default None để tương thích
    # ngược với outcome dựng trước khi field này tồn tại.
    origin: str | None = None
    # --- Instrumentation Pha 0 (IMPROVEMENT_SPEC §3): trả lời "chết ở đâu, vì sao, tốn bao lâu".
    # stage_reached: chặng cuối cùng ứng viên đi tới (idea|syntax|depth|dedup|local_floor|
    #   sub_universe|simmed|corr_checked|passed). fail_check: mã check thất bại (LOW_SHARPE/
    #   SELF_CORR/DEPTH/DUP/...). family: họ nhân tố. expr_depth: độ sâu cây. *_ms: thời gian
    #   mỗi mốc. dedup_key: canonical hash (đã fold). local_sharpe: Sharpe backtest local (TÁCH
    #   khỏi `sharpe` vốn là Brain sharpe — spec: đừng gộp local/brain vào 1 cột).
    stage_reached: str = ""
    fail_check: str = ""
    family: str = ""
    expr_depth: int | None = None
    gen_ms: float | None = None
    backtest_ms: float | None = None
    sim_ms: float | None = None
    dedup_key: str | None = None
    local_sharpe: float | None = None
    # --- Task 3 (spec C2): phân biệt "chưa chạm Brain" (pre-sim reject) khỏi "sim thật rớt".
    # presim_reason: reason gốc từ SimulationResult.presim_reason (None nếu đã sim thật, kể cả
    #   khi sim đó lỗi). is_brain_sim: True <=> outcome này ĐÃ gọi Brain thật (mặc định True để
    #   tương thích ngược — mọi call site cũ dựng IdeaOutcome đều từ 1 sim thật hoặc chưa sim gì
    #   cả nhưng không set presim_reason, nên default True không làm sai lệch outcome cũ).
    presim_reason: str | None = None
    is_brain_sim: bool = True


@dataclass(frozen=True, slots=True)
class ClosedLoopReport:
    """Thống kê một lần chạy ClosedLoop + lý do dừng."""

    ideas_tried: int
    sims_used: int
    n_passed: int
    n_abandoned: int
    stop_reason: str
    rho_sharpe: float | None = None


class CalibrationTracker:
    """Theo dõi độ tin ranking local: sau mỗi `every` sim, tính lại Spearman ρ giữa local
    sharpe và Brain sharpe (trên các expression đã có cả hai). ρ < `rho_bar` -> cảnh báo
    (ranking local có thể không còn đáng tin -> nên điều tra data/operator fidelity)."""

    def __init__(
        self, repo: MiniBrainRepository, *, every: int = 10, rho_bar: float = 0.5,
    ) -> None:
        self.repo = repo
        self.every = every
        self.rho_bar = rho_bar
        self.last_rho: float | None = None
        self._last_mark = 0

    def maybe_calibrate(self, sims_total: int) -> float | None:
        """Tính ρ nếu `sims_total` đã qua mốc bội số `every` kể từ lần trước; ngược lại None.
        ρ tính qua `spearman` trên `brain_local_sharpe_pairs()` (NaN nếu < 2 cặp)."""
        if sims_total < self._last_mark + self.every:
            return None
        self._last_mark = sims_total - (sims_total % self.every)
        pairs = self.repo.brain_local_sharpe_pairs()
        if len(pairs) < 2:
            self.last_rho = None
            return None
        local = np.array([p[0] for p in pairs], dtype=np.float64)
        brain = np.array([p[1] for p in pairs], dtype=np.float64)
        rho = spearman(local, brain)
        self.last_rho = rho
        if not np.isnan(rho) and rho < self.rho_bar:
            logger.warning(
                "Calibration ρ={:.3f} < bar {:.2f} — ranking local kém tin", rho, self.rho_bar,
            )
        return rho


class _GeneratesIdeas(Protocol):
    def next_batch(self) -> list[ShortlistCandidate]: ...


class _RefinesIdea(Protocol):
    def refine_and_sim(self, candidate: ShortlistCandidate) -> IdeaOutcome: ...


class ClosedLoop:
    """Vòng kín: lấy ý tưởng → refine+sim mỗi cái → persist kết quả SIM → dừng khi hết
    quota / cạn ý tưởng. Thuần điều phối; mọi dependency injected."""

    def __init__(
        self,
        idea_source: _GeneratesIdeas,
        refiner: _RefinesIdea,
        repo: MiniBrainRepository,
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        max_ideas: int | None = None,
        calibration_tracker: CalibrationTracker | None = None,
        alpha_logger=None,
        session_summary=None,
        dedup_key_fn=None,
        family_fn=None,
        max_per_family: int | None = None,
        on_family_closed=None,
        max_gp_sims: int | None = 3,
        pp_ready_fn=None,
        on_gp_budget_exhausted=None,
        on_epoch_reseed=None,
        frontier_reserve: "list[ShortlistCandidate] | None" = None,
        frontier_presim_fn=None,
    ) -> None:
        self.idea_source = idea_source
        self.refiner = refiner
        self.repo = repo
        self.region = region
        self.universe = universe
        self.max_ideas = max_ideas
        self.calibration_tracker = calibration_tracker
        # Logger CSV mọi ý tưởng (Task 2 RunAlphaLogger); None -> bỏ qua, tương thích ngược.
        self.alpha_logger = alpha_logger
        # Thu funnel cuối phiên (Pha 0 SessionSummary); None -> bỏ qua, tương thích ngược.
        self.session_summary = session_summary
        # Hàm chuẩn hoá expr -> dedup key (canonical fold, Pha 1.2). None -> identity (dùng
        # chuỗi thô như cũ). Giữ B1: ClosedLoop KHÔNG import src.lang; composition root
        # (main/adapter) tiêm CanonicalHasher thật vào đây.
        self.dedup_key_fn = dedup_key_fn or (lambda expr: expr)
        # Family-aware budget + exhaustion guard (Pha 2.2). family_fn: expr -> nhãn họ (None =
        # tắt, không giới hạn theo họ). max_per_family: trần ứng viên/họ/phiên; khi một họ đã
        # refine >= trần mà 0 pass -> ĐÓNG họ trong phiên (chuyển ngân sách sang họ khác). Họ có
        # >=1 pass thì không đóng (còn tiềm năng). Giữ B1: family_fn tiêm từ composition root.
        self.family_fn = family_fn
        self.max_per_family = max_per_family
        # Callback khi một họ bị đóng (Pha 2.3): nhận set họ bão hoà -> composition root nối
        # tới idea_generator.set_saturated_families để LLM tránh tái sinh. None -> bỏ qua.
        self.on_family_closed = on_family_closed
        # Cap ngân sách sim Brain riêng cho candidate origin "gp" (Task 3): 2 phiên chạy thật
        # gần nhất GP đốt ~10 sim (≈50% quota) ra toàn Sharpe ≤0.31 — calibration ρ=0.308 nên
        # floor local không lọc nổi rác GP TRƯỚC khi chạm Brain. Cap cứng rẻ hơn cải thiện
        # calibration: candidate origin "gp" vượt trần KHÔNG refine+sim (0 sim thật), ưu tiên
        # quota còn lại cho seed đã kiểm chứng (curated/alt_data) và alpha ghép (combiner) —
        # các nguồn này KHÔNG bị cap (YAGNI: chỉ GP là nguồn nhiễu đã có bằng chứng thật).
        # None = không cap (tương thích ngược cho ai đang set max_gp_sims=None tường minh).
        self.max_gp_sims = max_gp_sims
        # Yêu cầu 2026-07-18: callable () -> list[PowerPoolCandidate] (submission.manager.
        # select_power_pool_candidates bọc session_factory) — _report chấm PP-ready (cấu trúc
        # + theme + description) cuối phiên và in khối "⭐ PP SẴN SÀNG NỘP". Inject từ
        # composition root (build_closed_loop) để pipeline KHÔNG import tầng submission (B1).
        # None -> bỏ qua khối (tương thích ngược).
        self.pp_ready_fn = pp_ready_fn
        # A1: callback báo TRẦN NGÂN SÁCH SIM GP đã chạm (gp_sims_used >= max_gp_sims) —
        # composition root (build_closed_loop) nối xuống GPIdeaSource.set_gp_budget_exhausted
        # để nguồn GP BỎ HẲN tiến hoá (3–14 phút/batch) thay vì chạy xong rồi mới bị vứt ở gate
        # gp_budget bên dưới. Gọi ĐÚNG MỘT LẦN với True trong run() (xem `gp_budget_da_bao`);
        # None -> bỏ qua (tương thích ngược).
        self.on_gp_budget_exhausted = on_gp_budget_exhausted
        # B1: callable () -> bool — gọi khi batch RỖNG (KHÔNG chắc đã cạn tuyệt đối), để mở
        # epoch mới (seed mới + xoay dataset field, giữ nguyên họ đóng/eval_cache) thay vì dừng
        # ngay. Trả True = đã reseed thật (run() chạy tiếp); False/None = không reseed được
        # (dừng như cũ, `no_more_ideas`). Composition root (build_closed_loop) nối xuống
        # GPIdeaSource.reseed_epoch(). None -> bỏ qua (tương thích ngược, dừng ngay khi rỗng
        # như hành vi trước B1).
        self.on_epoch_reseed = on_epoch_reseed
        # WS3 T3.1/T3.2: hàng đợi seed non-PV (frontier/alt-data/fundamental/hypothesis) CHƯA
        # được đi thẳng `_sim_direct` trong phiên — build_closed_loop nạp một lần từ direct_
        # cores (đã lọc known_fields/avoided_hashes/verified_fields tại nguồn). Rút dần mỗi
        # batch (T3.1: bổ sung cho đủ sàn FRONTIER_MIN_FRACTION) thay vì dồn hết vào 1 batch
        # đầu rồi cạn — đúng vấn đề T3.1 muốn sửa (PROGRESS Session 16). deque để popleft() O(1).
        self.frontier_reserve: "deque[ShortlistCandidate]" = deque(frontier_reserve or [])
        # Review Important #2 (T3, 2026-07-19): callable (list[str]) -> None — gọi với danh
        # sách expr THÔ sắp được lộ ra khỏi `frontier_reserve` (dùng nốt khi cạn / bổ sung
        # topup), TRƯỚC khi refiner tự sim tuần tự từng cái. Composition root (build_closed_
        # loop) bind sẵn `presim_batch_into_cache` (Task 6, DÙNG CHUNG với `AltDataIdeaSource`,
        # không nhân đôi logic) với simulator/sim_config/presim_cache thật — kết quả ghi vào
        # CÙNG `presim_cache` mà `LocalTunerRefiner._sim_direct` đọc lại (khoá = expr thô),
        # khôi phục lợi ích multi-sim (round-trip) đã mất khi T3.1 tách frontier/fundamental/
        # hypothesis ra khỏi `AltDataIdeaSource`. None (mặc định) -> bỏ qua, refiner tự sim
        # tuần tự đơn như hiện tại (tương thích ngược, KHÔNG BẮT BUỘC composition root nối).
        self.frontier_presim_fn = frontier_presim_fn

    def run(self) -> ClosedLoopReport:
        """Lặp: next_batch → mỗi ý tưởng refine_and_sim → record_brain_sim → đếm. Dừng khi
        batch rỗng (cạn ý tưởng), đạt max_ideas, hoặc refiner ném QuotaExhausted (hết quota
        Brain). Bỏ qua expr đã thấy trong phiên (tránh refine trùng)."""
        ideas_tried = 0
        sims_used = 0
        n_passed = 0
        n_abandoned = 0
        # Task 3: đếm SỐ SIM Brain thật (cộng dồn outcome.sims_used) đã dùng bởi candidate
        # origin "gp" — cap ngân sách riêng cho GP, không đụng curated/alt_data/combiner.
        gp_sims_used = 0
        # A1: cờ đã bắn on_gp_budget_exhausted(True) trong phiên này chưa — đảm bảo callback
        # chỉ gọi ĐÚNG MỘT LẦN dù nhiều candidate GP sau đó vẫn rơi vào nhánh gp_budget.
        gp_budget_da_bao = False
        # B1: đếm số epoch đã reseed (chỉ để log) + cờ "batch VỪA rỗng ngay sau một reseed"
        # (đảm bảo tối đa 1 lần reseed liên tiếp cho mỗi lần rỗng thật — rỗng NGAY SAU reseed
        # nghĩa là epoch mới cũng cạn ngay lập tức -> cạn tuyệt đối, dừng thật thay vì reseed
        # vô hạn). Reset về False mỗi khi batch KHÔNG rỗng (còn ý tưởng để chạy).
        so_epoch = 0
        vua_reseed = False
        # seen chứa DEDUP KEY (canonical fold Pha 1.2), không phải chuỗi thô. Nạp avoid-list
        # bền = UNION 3 tập (avoided_hashes ∪ avoided_hashes_original ∪ dedup(avoided_exprs))
        # qua `compute_avoided_hashes` — hàm DÙNG CHUNG với `build_closed_loop` (Finding #2
        # review) để lọc core tại NGUỒN (Curated/AltDataIdeaSource) KHỚP đúng tập seen ở đây,
        # tránh khe hở hash-space (core đã sim-fail phiên trước lọt qua lọc nguồn rồi mới bị
        # chặn ở đây — SAU KHI đã đốt quota thật).
        seen: set[str] = compute_avoided_hashes(self.repo, self.dedup_key_fn)
        # Thu thập expr đạt Power Pool nhưng KHÔNG đạt Regular — để tóm tắt cuối phiên (không
        # đổi ClosedLoopReport public: chỉ log, tránh rủi ro cho consumer đang đọc report).
        power_pool_only: list[str] = []
        # Family budget state (Pha 2.2): đếm số ứng viên đã refine + số pass mỗi họ.
        fam_tried: dict[str, int] = {}
        fam_passed: dict[str, int] = {}
        closed_families: set[str] = set()

        def _report(stop_reason: str) -> ClosedLoopReport:
            if power_pool_only:
                # RC8 fix: "Power Pool eligible" chỉ là cờ CẤU TRÚC (is_power_pool: Sharpe≥1.0,
                # ≤8 operator, ≤3 field, self_corr≤0.5) — KHÔNG phải xác nhận nộp được. Các
                # alpha này KHÔNG đạt ngưỡng Regular (Sharpe cần ~1.58+, xem
                # config/thresholds.py IS_LADDER) và tool này CHƯA có đường nộp Power Pool tự
                # động (không auth/sim/submit thật ở đây). Nêu rõ hành động tiếp theo: người
                # dùng phải tự xem lại từng alpha trên WQ Brain và tự quyết định nộp hay không.
                logger.info(
                    "⭐ Tóm tắt Power Pool: {} ứng viên đạt CẤU TRÚC Power Pool (Sharpe≥1.0, "
                    "≤8 operator, ≤3 field, self_corr≤0.5) nhưng KHÔNG đạt ngưỡng Regular "
                    "(Sharpe cần ~1.58+). Đây KHÔNG phải xác nhận nộp được — cần khớp thêm "
                    "Power Pool Theme tuần hiện tại + mô tả ≥100 ký tự. Kiểm tra và nộp qua "
                    "đường pure Power Pool:\n"
                    "   ./venv/Scripts/python.exe main.py submit --power-pool  (dry-run)\n"
                    "   ./venv/Scripts/python.exe main.py submit --power-pool --no-dry-run\n"
                    "Ứng viên: {}",
                    len(power_pool_only),
                    ", ".join(_short(e, 40) for e in power_pool_only),
                )
            # Task 8: khối "SẴN SÀNG NỘP" — PHÂN BIỆT RẠCH RÒI với "Power Pool eligible" ở
            # trên (đó chỉ là cờ CẤU TRÚC, KHÔNG xác nhận nộp được — commit e27821d). Khối
            # này LÀ xác nhận nộp được Regular thật: status=passed (WQ đã chấm Sharpe/Fitness/
            # Turnover/IS-Ladder) + failed_checks=[] (không WQ check nào tự FAIL) + self_corr
            # ĐÃ VERIFY < SELF_CORR_MAX (config/thresholds.py, không hardcode). Guard
            # getattr+callable: repo fake/cũ thiếu method này (nhiều test dùng fake tối giản)
            # vẫn chạy được, coi như 0 alpha sẵn sàng — tương thích ngược, giống pattern
            # avoided_hashes_original ở đầu run().
            _submit_ready_fn = getattr(self.repo, "submit_ready_alphas", None)
            ready = _submit_ready_fn(SELF_CORR_MAX) if callable(_submit_ready_fn) else []
            if ready:
                detail = "\n".join(
                    f"   • {r.wq_alpha_id}  Sharpe={_fmt(r.sharpe)}  self_corr={r.self_corr:.2f}"
                    f"  {_short(r.expression, 60)}"
                    for r in ready
                )
                logger.info(
                    "🚀 SẴN SÀNG NỘP: {} alpha đạt CẢ BỐN điều kiện Regular thật (status=passed, "
                    "failed_checks=[], Sharpe/Fitness đạt ngưỡng NỘP THẬT, self_corr<{:.2f} đã "
                    "verify) — KHÁC \"Power Pool eligible\" ở trên (đó chỉ là cờ CẤU TRÚC, không "
                    "xác nhận nộp được), khối này LÀ xác nhận nộp được. Nộp ngay bằng lệnh:\n"
                    "   ./venv/Scripts/python.exe main.py submit --no-dry-run\n{}",
                    len(ready), SELF_CORR_MAX, detail,
                )
            else:
                logger.info(
                    "🚀 SẴN SÀNG NỘP: 0 alpha sẵn sàng (chưa có alpha nào đạt cả status=passed "
                    "+ failed_checks=[] + Sharpe/Fitness đạt ngưỡng NỘP THẬT + self_corr<{:.2f} "
                    "đã verify trong DB). Muốn kiểm tra "
                    "thủ công: ./venv/Scripts/python.exe main.py submit --dry-run",
                    SELF_CORR_MAX,
                )
            # Yêu cầu 2026-07-18: chấm PP-ready ĐẦY ĐỦ (cấu trúc + theme tuần + description)
            # ngay cuối phiên — người vận hành không phải tự gõ `submit --power-pool` mới biết.
            if self.pp_ready_fn is not None:
                try:
                    pp_cands = list(self.pp_ready_fn() or [])
                except Exception as exc:  # noqa: BLE001 - báo cáo phụ không được phá report chính
                    logger.warning("Không chấm được PP-ready cuối phiên: {}", exc)
                    pp_cands = []
                pp_ready = [c for c in pp_cands if not getattr(c, "skip_reason", "")]
                pp_skipped = [c for c in pp_cands if getattr(c, "skip_reason", "")]
                if pp_ready:
                    pp_detail = "\n".join(
                        f"   • {c.wq_alpha_id}  Sharpe={_fmt(c.sharpe)}  {_short(c.expression, 60)}"
                        for c in pp_ready
                    )
                    logger.info(
                        "⭐ PP SẴN SÀNG NỘP: {} alpha đạt ĐỦ cấu trúc Power Pool + khớp theme "
                        "tuần hiện tại + mô tả >=100 ký tự. Nộp (quota 1 pure PP/ngày):\n"
                        "   ./venv/Scripts/python.exe main.py submit --power-pool --no-dry-run\n{}",
                        len(pp_ready), pp_detail,
                    )
                else:
                    logger.info(
                        "⭐ PP SẴN SÀNG NỘP: 0 (chưa ứng viên nào đạt đủ cấu trúc + theme + mô tả)."
                    )
                if pp_skipped:
                    logger.info(
                        "   ({} ứng viên bị bỏ qua — lý do đầu: {})",
                        len(pp_skipped), pp_skipped[0].skip_reason,
                    )
            return ClosedLoopReport(
                ideas_tried, sims_used, n_passed, n_abandoned, stop_reason,
                rho_sharpe=self.calibration_tracker.last_rho if self.calibration_tracker else None,
            )

        while True:
            logger.info("⏳ Sinh batch ý tưởng (GP + decorrelate)…")
            _gen_t0 = time.perf_counter()
            batch = self.idea_source.next_batch()
            gen_batch_ms = (time.perf_counter() - _gen_t0) * 1000.0
            # Sửa Important MỚI (re-review T3, vòng 2): T3.2 (xoay reserve theo bão hoà pool)
            # PHẢI chạy TRƯỚC bất kỳ nhánh nào bên dưới PEEK `frontier_reserve` để build batch
            # (dùng nốt khi cạn / bổ sung topup) — nếu rotate mutate deque SAU khi đã peek,
            # snapshot `batch` giữ thứ tự CŨ trong khi `popleft()` trễ ở for-loop rút theo VỊ
            # TRÍ trên deque đã bị xoay: lệch khỏi `cand` đang xử lý -> candidate ĐÃ refine vẫn
            # có thể còn trong reserve (tồn tại kép) VÀ candidate CHƯA xử lý bị pop nhầm, mất
            # vĩnh viễn không log (đúng lớp lỗi Important #1 vừa sửa, tái phát qua đường khác).
            # Rotate 1 LẦN DUY NHẤT ở đây, TRƯỚC mọi peek — cả 2 nhánh dưới nhìn CÙNG một thứ
            # tự ổn định, và deque KHÔNG bị mutate lại nữa cho tới khi for-loop tự popleft().
            if self.frontier_reserve and self.family_fn is not None:
                pool_exprs = _read_pool_exprs(self.repo)
                if pool_exprs:
                    family_share = compute_family_pool_share(
                        [self.family_fn(e) for e in pool_exprs]
                    )
                    rotate_reserve_by_saturation(
                        self.frontier_reserve, self.family_fn, family_share,
                        FRONTIER_SATURATION_K,
                    )
            # Sửa Important #1 (review T3, 2026-07-19): CHỈ PEEK (không pop) candidate lấy từ
            # `frontier_reserve` ở cả 2 nhánh dưới (dùng nốt khi cạn / bổ sung topup) —
            # `pending_from_reserve` đếm số phần tử Ở CUỐI `batch` còn "nợ" một popleft() thật
            # sự. Việc rút THẬT khỏi hàng đợi chỉ xảy ra trong for-loop bên dưới, ĐÚNG lúc
            # candidate đã qua gate `max_ideas` (chắc chắn sẽ được xử lý tiếp) — tránh mất
            # trắng, không log, không dấu vết khi max_ideas chạm NGAY giữa lúc đang rút (trước
            # fix: `batch = list(reserve); reserve.clear()` xoá NGAY LẬP TỨC, để rồi for-loop
            # mới phát hiện đã hết ngân sách và trả về mà chưa từng xử lý candidate đó).
            pending_from_reserve = 0
            used_drain_khi_can = False
            if not batch and self.frontier_reserve:
                # T3.1: idea_source cạn NHƯNG hàng đợi non-PV còn — dùng NỐT thay vì dừng
                # ngay/reseed (đúng "không chặn batch": còn gì thì dùng hết, kể cả khi idea_
                # source chính đã hết ý tưởng của riêng nó). PEEK toàn bộ — xem comment trên.
                batch = list(self.frontier_reserve)
                pending_from_reserve = len(batch)
                used_drain_khi_can = True
                if self.frontier_presim_fn is not None:
                    self.frontier_presim_fn([c.expr for c in batch])
                logger.info(
                    "🌐 idea_source cạn — dùng nốt {} candidate non-PV còn lại trong reserve.",
                    len(batch),
                )
            if not batch:
                # B1: batch rỗng chưa chắc đã cạn tuyệt đối — thử mở epoch mới (seed mới +
                # xoay dataset field ưu tiên) MỘT LẦN; rỗng NGAY SAU reseed đó (vua_reseed vẫn
                # True vì chưa có batch không-rỗng nào reset nó) mới coi là cạn tuyệt đối.
                if (not vua_reseed) and self.on_epoch_reseed is not None and self.on_epoch_reseed():
                    vua_reseed = True
                    so_epoch += 1
                    # Ngân sách sim GP (Task 3/A1) là NGÂN SÁCH THEO PHIÊN NGHIÊN CỨU, nhưng
                    # epoch mới coi như một đợt tìm kiếm mới (seed mới + có thể xoay field) —
                    # reset để candidate gp lại được sim ở epoch mới, KHÔNG bị khoá bởi trần đã
                    # chạm ở epoch trước. gp_budget_da_bao (cờ một-lần của callback A1) cũng
                    # phải reset — nếu không, epoch mới chạm trần lần nữa sẽ KHÔNG bắn lại
                    # on_gp_budget_exhausted(True) (cờ tưởng đã bắn từ epoch trước), khiến
                    # GPIdeaSource của epoch mới không được báo tắt tiến hoá khi thật sự chạm trần.
                    gp_sims_used = 0
                    gp_budget_da_bao = False
                    logger.info(
                        "🔄 Epoch #{}: reseed (batch rỗng) — seed mới + xoay dataset, giữ họ đóng.",
                        so_epoch,
                    )
                    continue
                logger.info("Cạn ý tưởng (batch rỗng) — dừng vòng kín.")
                return _report("no_more_ideas")
            vua_reseed = False
            # WS3 T3.1: sàn quota đa dạng mỗi batch — CHỈ áp dụng khi có cả reserve LẪN
            # family_fn (không biết family thì không thể phân biệt PV/non-PV, thà bỏ qua còn
            # hơn đoán sai và tiêu hao reserve vô ích). T3.2 (xoay theo bão hoà) đã chạy Ở TRÊN
            # (đầu vòng lặp, TRƯỚC nhánh "dùng nốt") — KHÔNG rotate lại ở đây (xem comment
            # "Sửa Important MỚI" ở đầu vòng lặp: rotate 2 lần trong cùng 1 batch sẽ lại làm
            # lệch snapshot batch đã peek TRƯỚC nếu nhánh "dùng nốt" đã chạy).
            if self.frontier_reserve and self.family_fn is not None:
                # T3.1: chỉ topup khi nhánh "dùng nốt" (đã lấy TOÀN BỘ reserve) chưa chạy —
                # tránh cộng dồn 2 nguồn cùng lúc.
                if not used_drain_khi_can:
                    non_pv = sum(
                        1 for c in batch if self.family_fn(c.expr) not in _PV_FAMILY_LABELS
                    )
                    topup = compute_frontier_topup(
                        len(batch), non_pv, FRONTIER_MIN_FRACTION, len(self.frontier_reserve),
                    )
                    if topup > 0:
                        # PEEK (không pop) — xem comment "Sửa Important #1" ở đầu vòng lặp.
                        added = list(self.frontier_reserve)[:topup]
                        if self.frontier_presim_fn is not None:
                            self.frontier_presim_fn([c.expr for c in added])
                        batch = list(batch) + added
                        pending_from_reserve += topup
                        logger.info(
                            "🌐 Sàn quota đa dạng: bổ sung {} candidate non-PV từ reserve "
                            "(sẽ rút thật khi tới lượt xử lý).", topup,
                        )
            # Chi phí sinh (GP/decorrelate) là của CẢ batch; phân bổ đều cho mỗi ứng viên để
            # điền gen_ms (refiner không biết chi phí này). Xấp xỉ đủ tốt cho funnel timing.
            gen_ms_each = gen_batch_ms / len(batch)
            logger.info("📦 Batch: {} ứng viên qua sàng lọc/decorrelate.", len(batch))
            # Vị trí (chỉ số) TỪ ĐÂU trong `batch` là candidate "nợ" một popleft() thật sự —
            # luôn nằm ở ĐUÔI batch (nhánh dùng-nốt/topup ở trên luôn APPEND vào cuối).
            _reserve_start_idx = len(batch) - pending_from_reserve
            for _idx, cand in enumerate(batch):
                if self.max_ideas is not None and ideas_tried >= self.max_ideas:
                    logger.info("Đạt trần max_ideas={} — dừng.", self.max_ideas)
                    return _report("no_more_ideas")
                if pending_from_reserve > 0 and _idx >= _reserve_start_idx:
                    # Đã qua gate max_ideas ở trên -> XÁC NHẬN candidate này sẽ được xử lý
                    # tiếp (dedup/family-đóng có thể vẫn loại nó, nhưng đó là tiêu thụ CÓ CHỦ
                    # Ý + có log, khác hẳn mất trắng vì cắt ngang) -> rút THẬT khỏi hàng đợi
                    # bây giờ (không rút sớm hơn — đó chính là bug Important #1).
                    self.frontier_reserve.popleft()
                key = self.dedup_key_fn(cand.expr)
                if key in seen:
                    logger.info("↩︎ Bỏ ý tưởng trùng phiên/avoid-list: {}", _short(cand.expr))
                    if self.session_summary is not None:
                        self.session_summary.record_dup_blocked()
                    continue
                seen.add(key)
                # Family budget: bỏ candidate thuộc họ đã ĐÓNG (đã cạn ngân sách mà 0 pass).
                fam = self.family_fn(cand.expr) if self.family_fn is not None else None
                if fam is not None and fam in closed_families:
                    logger.info("↩︎ Bỏ ý tưởng thuộc họ đã đóng [{}]: {}", fam, _short(cand.expr))
                    if self.session_summary is not None:
                        self.session_summary.record_dup_blocked()
                    continue
                origin = getattr(cand, "origin", "gp")
                # Task 3: candidate origin "gp" đã chạm trần ngân sách sim GP/phiên -> KHÔNG
                # refine+sim (0 sim thật) — outcome trung thực để funnel CSV/session_summary
                # ghi rõ ứng viên bị chặn vì ngân sách, không phải bị `continue` âm thầm như
                # dedup/family-closed. Candidate origin khác (curated/alt_data/combiner) không
                # bị cap này -> luôn đi tiếp đường refine+sim bình thường.
                if (
                    origin == "gp"
                    and self.max_gp_sims is not None
                    and gp_sims_used >= self.max_gp_sims
                ):
                    logger.info(
                        "🚧 Ứng viên GP #{} chạm trần ngân sách sim ({} sim) — bỏ qua, ưu "
                        "tiên quota cho seed/combiner: {}",
                        ideas_tried + 1, self.max_gp_sims, _short(cand.expr),
                    )
                    # A1: báo TRẦN ĐÃ CHẠM đúng 1 lần/phiên (composition root nối xuống
                    # GPIdeaSource.set_gp_budget_exhausted -> next_batch bỏ hẳn tiến hoá GP
                    # thay vì chạy xong rồi mới bị vứt ở đây).
                    if not gp_budget_da_bao and self.on_gp_budget_exhausted is not None:
                        gp_budget_da_bao = True
                        self.on_gp_budget_exhausted(True)
                    outcome = IdeaOutcome(
                        expr=cand.expr, canonical_hash=key, passed=False, wq_alpha_id=None,
                        sharpe=None, fitness=None, turnover=None, self_corr=None,
                        sims_used=0, stop_reason="gp_budget", source=origin,
                        stage_reached="gp_budget", fail_check="GP_BUDGET",
                        family=fam or "", dedup_key=key, is_brain_sim=False,
                    )
                else:
                    logger.info(
                        "🔎 Ý tưởng #{}: refine+sim {}", ideas_tried + 1, _short(cand.expr),
                    )
                    try:
                        outcome = self.refiner.refine_and_sim(cand)
                    except QuotaExhausted:
                        logger.info(
                            "Hết quota Brain — dừng vòng kín ({} sim đã dùng).", sims_used,
                        )
                        return _report("quota")
                # Điền gen_ms (chi phí sinh batch phân bổ) nếu refiner chưa set — refiner đo
                # backtest_ms/sim_ms, còn gen_ms thuộc tầng ClosedLoop (Fix gap Pha 0).
                if getattr(outcome, "gen_ms", None) is None:
                    outcome = replace(outcome, gen_ms=gen_ms_each)
                # Finding #4 (review): stamp `origin` của candidate (curated/gp/alt_data/
                # combiner) vào outcome — refiner không biết origin (chỉ ClosedLoop có, đọc từ
                # candidate) nên mọi outcome qua đường tune LUÔN có source="gp_local_tuner" bất
                # kể nguồn ý tưởng thật, làm mất khả năng đo tiêu chí nghiệm thu "≥60% sim
                # thuộc seed/hypothesis/combiner". Cùng pattern guard với gen_ms ở trên.
                if getattr(outcome, "origin", None) is None:
                    outcome = replace(outcome, origin=origin)
                self.repo.record_brain_sim(
                    canonical_hash=outcome.canonical_hash, expr_string=outcome.expr,
                    wq_alpha_id=outcome.wq_alpha_id, region=self.region,
                    universe=self.universe, sharpe=outcome.sharpe, fitness=outcome.fitness,
                    turnover=outcome.turnover, self_corr=outcome.self_corr,
                    status="passed" if outcome.passed else "failed",
                )
                # Persist hash GỐC (pre-tune, `key` tính ở trên TRƯỚC khi refiner tune) để
                # phiên SAU pre-check khớp đúng không gian hash (Task 6 fix). CHỈ ghi khi
                # outcome đã thực sự sim Brain (`is_brain_sim`) — sim tốn thật là bằng chứng
                # thật (pass hay fail đều đáng nhớ, khớp semantics `seen.add(key)` trong-phiên
                # ở trên vốn cũng chặn trùng bất kể pass/fail). Outcome bị gate LOCAL (local_floor/
                # presim_reject/depth/sub_universe...) KHÔNG có bằng chứng Brain thật gì cả — cấm
                # vĩnh viễn sẽ làm cạn seed novelty (RC1: ALT_DATA/FUNDAMENTAL core bị gate 1 lần
                # là mất luôn cơ hội thử lại ở phiên sau, kể cả khi conditioning/config đã đổi).
                # Guard: repo fake/cũ thiếu method vẫn chạy được (tương thích ngược).
                if outcome.is_brain_sim:
                    record_avoided_hash = getattr(self.repo, "record_avoided_hash", None)
                    if callable(record_avoided_hash):
                        record_avoided_hash(key)
                ideas_tried += 1
                # Log CSV mọi ý tưởng có outcome (kể cả bị gate 0-sim) — Task 3.
                if self.alpha_logger is not None:
                    self.alpha_logger.log(ideas_tried, outcome)
                if self.session_summary is not None:
                    self.session_summary.record(outcome)
                sims_used += outcome.sims_used
                # Đếm theo SỐ SIM thật (outcome.sims_used), KHÔNG phải số candidate (review
                # fix): với --refiner llm, 1 candidate gp có thể đốt nhiều sim (patience loop)
                # — đếm theo candidate thì cap 3 vẫn có thể cho lọt 15 sim thật. Outcome bị
                # gate local/gp_budget có sims_used=0 nên tự nhiên không cộng gì.
                if origin == "gp":
                    gp_sims_used += outcome.sims_used
                if outcome.passed:
                    n_passed += 1
                else:
                    n_abandoned += 1
                    if getattr(outcome, "power_pool_eligible", False):
                        power_pool_only.append(outcome.expr)
                # Family budget (Pha 2.2): cập nhật đếm + đóng họ nếu đã cạn ngân sách mà 0 pass.
                if fam is not None:
                    fam_tried[fam] = fam_tried.get(fam, 0) + 1
                    if outcome.passed:
                        fam_passed[fam] = fam_passed.get(fam, 0) + 1
                    if (
                        self.max_per_family is not None
                        and fam_tried[fam] >= self.max_per_family
                        and fam_passed.get(fam, 0) == 0
                    ):
                        closed_families.add(fam)
                        logger.info(
                            "🚪 Đóng họ [{}]: đã thử {} ứng viên, 0 pass — chuyển ngân sách.",
                            fam, fam_tried[fam],
                        )
                        if self.on_family_closed is not None:
                            self.on_family_closed(set(closed_families))
                # Kết quả 1 dòng: 0 sim = bị gate local chặn trước khi đốt quota Brain. In kèm
                # lý do (stop_reason) để phân biệt local_floor (Sharpe/turnover) vs sub_universe.
                if outcome.sims_used == 0:
                    logger.info("   → ⚠ bị gate local chặn [{}] (0 sim Brain).", outcome.stop_reason)
                else:
                    logger.info(
                        "   → {} Sharpe={} fit={} TO={} self_corr={} ({} sim)",
                        "✅ PASSED" if outcome.passed else "✗ failed",
                        _fmt(outcome.sharpe), _fmt(outcome.fitness), _fmt(outcome.turnover),
                        _fmt(outcome.self_corr), outcome.sims_used,
                    )
                if getattr(outcome, "power_pool_eligible", False):
                    # RC8 fix: nêu rõ đây là cờ CẤU TRÚC, không phải "đã nộp được" — alpha vẫn
                    # dưới ngưỡng Regular và chưa có đường nộp Power Pool tự động trong tool này.
                    logger.info(
                        "   ⭐ Đạt CẤU TRÚC Power Pool (Sharpe≥1.0, ≤8 op, ≤3 field, "
                        "self_corr≤0.5) — KHÔNG phải đã nộp được, cần tự xem lại trên WQ Brain."
                    )
                logger.info(
                    "   Σ {} ý tưởng / {} sim / {} pass / {} bỏ.",
                    ideas_tried, sims_used, n_passed, n_abandoned,
                )
                if self.calibration_tracker is not None:
                    self.calibration_tracker.maybe_calibrate(sims_used)
