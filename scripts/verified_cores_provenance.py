"""Provenance cho VERIFIED_CORES (src/app/closed_loop_adapters.py, Task 7 phần 3).

Bối cảnh: `VERIFIED_CORES` mang nhãn "Sharpe ~1.5+ trên Brain" (docstring, commit d481fe1)
nhưng KHÔNG kèm universe/decay/neutralization gốc — nghi vấn config drift đã ghi nhận
(SIM_DEFAULTS.universe hiện = TOP3000, có phiên CSV cũ chạy TOP1000; xem
docs/tailieu/review20260710/IMPROVEMENT_SPEC_v2.md mục C7). Đừng tin nhãn cũ mù quáng —
script này in RÕ (a) nhãn cũ đang claim gì và (b) settings sim SẼ dùng nếu core này được
sim NGAY BÂY GIỜ qua đường mặc định, để người dùng tự so sánh / quyết định re-sim.

Mặc định DRY-RUN tuyệt đối: chỉ build + in bảng, KHÔNG gọi mạng, KHÔNG chạm Simulator.
`--run-live` cần thêm `--i-understand-this-sends-real-sims` mới không bị từ chối ngay —
nhưng NGAY CẢ KHI xác nhận, script vẫn KHÔNG tự đăng nhập/khởi tạo phiên Brain hay gọi
`simulator.simulate()` (wiring live để lại TODO tường minh, đúng ràng buộc "không tự
auth/submit"). Người dùng muốn sim thật phải tự truyền một `Simulator` ĐÃ AUTH qua tham số
`simulator=` khi gọi `run()` bằng Python trực tiếp — CLI chỉ dừng ở cảnh báo.

Chạy:  venv/Scripts/python -m scripts.verified_cores_provenance
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from loguru import logger

from src.app.closed_loop_adapters import VERIFIED_CORES
from src.simulation.config import SimConfig

# Nhãn "đã kiểm chứng" duy nhất còn lưu trong code (docstring VERIFIED_CORES) — CHỈ có
# Sharpe xấp xỉ, không có universe/decay/neutralization gốc kèm theo. Đây CHÍNH LÀ lỗ hổng
# khiến script provenance này cần tồn tại: không thể tái lập "config đã đạt 1.57" một cách
# chắc chắn, chỉ có thể build config SẼ DÙNG nếu sim lại core hôm nay.
CLAIMED_LABEL = (
    "Sharpe ~1.5+ tren Brain (docstring VERIFIED_CORES, commit d481fe1) -- "
    "KHONG luu universe/decay/neutralization goc; nghi van config drift "
    "(xem docs/tailieu/review20260710/IMPROVEMENT_SPEC_v2.md muc C7), dung tin nhan cu mu quang"
)


@dataclass(frozen=True, slots=True)
class ProvenanceRow:
    """Một dòng bảng: core + nhãn cũ (không đầy đủ) + settings SẼ DÙNG nếu sim ngay bây giờ."""

    expr: str
    claimed: str
    intended_settings: dict


def build_provenance(
    cores: tuple[str, ...] = VERIFIED_CORES,
    *,
    region: str = "USA",
    universe: str = "TOP3000",
) -> list[ProvenanceRow]:
    """Dựng bảng provenance THUẦN LOCAL (không backtest, không gọi mạng).

    `intended_settings` là config MẶC ĐỊNH (`SimConfig.default`) — đúng config mà đường sim
    thẳng/sim-lần-đầu (vd `LocalTunerRefiner._sim_direct`, hoặc trước khi `local_tuner.tune`
    điều chỉnh decay/truncation/neutralization) sẽ gửi cho
    `Simulator.simulate(expr, settings=...)` nếu core này được sim NGAY BÂY GIỜ."""
    cfg = SimConfig.default(region=region, universe=universe)
    settings = cfg.to_settings()
    return [
        ProvenanceRow(expr=e, claimed=CLAIMED_LABEL, intended_settings=dict(settings))
        for e in cores
    ]


def format_table(rows: list[ProvenanceRow]) -> str:
    """In bảng người đọc được: mỗi core kèm nhãn cũ + settings dự kiến gửi Brain."""
    lines = [
        "VERIFIED_CORES provenance -- core -> nhan cu (claimed) -> settings se dung neu sim ngay bay gio",
        "=" * 100,
    ]
    for i, row in enumerate(rows, 1):
        lines.append(f"[{i}] expr: {row.expr}")
        lines.append(f"    claimed : {row.claimed}")
        lines.append(f"    settings: {row.intended_settings}")
        lines.append("-" * 100)
    return "\n".join(lines)


def run(
    *,
    run_live: bool = False,
    confirmed: bool = False,
    simulator: object | None = None,
    region: str = "USA",
    universe: str = "TOP3000",
) -> str:
    """In + trả bảng provenance (text). Mặc định (`run_live=False`) TUYỆT ĐỐI không gọi
    `simulator` dù có truyền vào (tham số `simulator` chỉ tồn tại để test/tool inject fake
    và xác nhận nó KHÔNG bị gọi). `run_live=True` mà thiếu `confirmed` -> in cảnh báo, từ
    chối, KHÔNG sim. `run_live=True` + `confirmed=True` -> vẫn KHÔNG tự gọi `simulator`
    (wiring live để lại TODO tường minh — không tự đăng nhập/nộp bài Brain)."""
    rows = build_provenance(region=region, universe=universe)
    table = format_table(rows)
    logger.info("Provenance VERIFIED_CORES ({} core) -- DRY-RUN, khong goi sim.", len(rows))
    print(table)

    if not run_live:
        return table

    if not confirmed:
        msg = (
            "--run-live doi hoi them --i-understand-this-sends-real-sims de xac nhan "
            "(sim that dot quota Brain that). Bo qua, KHONG sim."
        )
        logger.warning(msg)
        print(msg)
        return table

    # TODO(live-wiring): sim thật cần một Simulator ĐÃ AUTH (WQBrainClient phiên thật) --
    # script này KHÔNG tự đăng nhập/khởi tạo phiên (giữ nguyên ràng buộc không tự auth/nộp
    # bài). Muốn sim thật: tự dựng Simulator đã auth rồi gọi
    # `run(run_live=True, confirmed=True, simulator=<simulator đã auth>)` từ Python, đọc
    # `rows = build_provenance(...)` rồi tự vòng lặp `simulator.simulate(row.expr,
    # settings=row.intended_settings)` -- cố ý KHÔNG tự động hoá bước cuối này ở đây.
    msg = (
        "--run-live: da xac nhan nhung wiring simulator that CHUA duoc noi day tu CLI nay "
        "(TODO co chu). Tu goi build_provenance()/simulator.simulate(...) tu code Python "
        "neu that su can sim that -- CLI nay dung o day, KHONG tu goi simulator."
    )
    logger.warning(msg)
    print(msg)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "In bang provenance VERIFIED_CORES (dry-run mac dinh, KHONG gui sim that)"
        )
    )
    parser.add_argument(
        "--run-live", action="store_true",
        help="Yeu cau sim that (van bi tu choi neu thieu co xac nhan ben duoi)",
    )
    parser.add_argument(
        "--i-understand-this-sends-real-sims", action="store_true", dest="confirmed",
        help="Xac nhan hieu ro --run-live dot quota Brain that",
    )
    parser.add_argument("--region", default="USA")
    parser.add_argument("--universe", default="TOP3000")
    args = parser.parse_args()
    run(
        run_live=args.run_live, confirmed=args.confirmed,
        region=args.region, universe=args.universe,
    )


if __name__ == "__main__":
    main()
