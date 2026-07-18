"""C2: test tái lập — song song ≡ tuần tự (chốt chặn chất lượng Pha C).

Mở rộng parity test của C1 (``test_gp_engine_song_song_giong_het_tuan_tu``, CHỈ
``n_generations=1``) sang **2 thế hệ** với pop nhỏ (10) để phủ tương tác cache-hit/elitism
ĐA THẾ HỆ mà bài test C1 (một thế hệ duy nhất) chưa từng chạm tới: ở thế hệ 2, offspring có
thể trùng ``canonical_hash`` với cá thể đã eval ở thế hệ 1 (qua ``dedup_population``/NSGA-II
giữ lại cha) — ``_prefetch_parallel`` (A3) phải HIT cache đúng y hệt đường tuần tự, nếu thứ
tự nhận kết quả future / rng dùng chung / cache lệch một ly thì fitness thế hệ 2 sẽ trôi.

Mỗi engine (tuần tự và song song) dựng ``eval_cache`` RIÊNG (dict mới) + repo SQLite
in-memory RIÊNG — KHÔNG share state giữa hai lần chạy: nếu share, engine chạy sau có thể HIT
cache/pool do engine chạy trước ghi, che giấu lệch thật giữa hai đường thực thi (parity giả
tạo). Registry CHIA SẺ một object cho cả ba nơi dùng (engine tuần tự, engine song song,
initializer của worker) theo đúng cách composition root thật (``ClosedLoopGPIdeaSource``)
nối dây — ``OperatorRegistry`` chỉ là bảng tra cứu bất biến trong một phiên, không mang
trạng thái ảnh hưởng kết quả nên chia sẻ an toàn (khác ``eval_cache``/repo là nơi TÍCH LŨY
kết quả xuyên cá thể/thế hệ, phải tách riêng).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.operators_local  # noqa: F401  (side-effect: nạp operator vào registry)
from src.backtest.config import Neutralization, PortfolioConfig
from src.gp.engine import GPEngine, GPRunResult
from src.gp.parallel_eval import khoi_tao_worker
from src.lang.registry import default_registry
from src.storage.db import init_db
from src.storage.repository import MiniBrainRepository


def _repo_moi() -> MiniBrainRepository:
    """Repository SQLite in-memory MỚI mỗi lần gọi — mỗi engine (tuần tự/song song) một bản
    riêng, không share pool self-corr (pool ảnh hưởng gate + ``pool_corr_penalty``)."""
    engine = create_engine("sqlite:///:memory:", future=True)
    init_db(engine)
    sf = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return MiniBrainRepository(sf)


def _cau_hinh() -> PortfolioConfig:
    return PortfolioConfig(
        neutralization=Neutralization.NONE, decay=0, truncation=0.10,
        scale_book=1.0, delay=1,
    )


def _chay_engine(
    small_panel,  # noqa: ANN001 — MarketData của fixture integration (xem tests/conftest.py)
    *,
    n_jobs: int,
    seed: int,
    registry,  # noqa: ANN001 — OperatorRegistry, chia sẻ giữa lời gọi seq/par (xem docstring module)
    executor: ProcessPoolExecutor | None = None,
) -> GPRunResult:
    """Dựng + chạy một ``GPEngine`` ĐỘC LẬP: repo riêng, ``eval_cache`` riêng (dict mới) —
    pop nhỏ (10) x 2 thế hệ, đủ phủ dedup/elitism đa thế hệ nhưng vẫn nhanh cho CI."""
    eng = GPEngine(
        data=small_panel, repo=_repo_moi(), config=_cau_hinh(), registry=registry,
        pop_size=10, n_generations=2, seed=seed,
        n_jobs=n_jobs, executor=executor, eval_cache={},
    )
    return eng.run()


def test_song_song_va_tuan_tu_cho_cung_ket_qua(small_panel) -> None:  # noqa: ANN001
    """Chốt chặn chất lượng Pha C: cùng seed + cùng data -> quần thể cuối GIỐNG HỆT (tập
    canonical_hash + sharpe_deflated làm tròn 12 chữ số) giữa ``n_jobs=1`` và ``n_jobs=2``,
    xuyên 2 thế hệ (phủ tương tác cache-hit/elitism đa thế hệ mà C1 chưa phủ)."""
    registry = default_registry()

    r_seq = _chay_engine(small_panel, n_jobs=1, seed=123, registry=registry)

    executor = ProcessPoolExecutor(
        max_workers=2, initializer=khoi_tao_worker,
        initargs=(small_panel, _cau_hinh(), registry),
    )
    try:
        r_par = _chay_engine(
            small_panel, n_jobs=2, seed=123, registry=registry, executor=executor,
        )
    finally:
        executor.shutdown(wait=True)

    def dau_van(res: GPRunResult) -> list[tuple[str, float]]:
        return sorted(
            (i.canonical_hash(), round(i.fitness.sharpe_deflated, 12))
            for i in res.final_population if i.fitness is not None
        )

    dv_seq = dau_van(r_seq)
    dv_par = dau_van(r_par)
    assert dv_seq, "quần thể cuối tuần tự rỗng — fixture/test hỏng, không phải phép so sánh parity"
    assert dv_seq == dv_par
