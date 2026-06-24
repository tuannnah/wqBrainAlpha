"""Chạy ground-truth: simulate 60 biểu thức đã thẩm định -> lưu DB account đích.

Tái dùng src.simulation.Simulator (POST/poll/fetch + rate-limit + chặn auth chết)
và src.data.WQBrainClient (đăng nhập qua WQ_EMAIL/WQ_PASSWORD, cache .wq_session).
KHÔNG dùng wqb-mcp (sai đường dẫn, từng trả 400/403).

Chạy TUẦN TỰ (max_concurrent=1) để không vượt giới hạn slot của tài khoản basic.
Mỗi sim xong (sharpe != null) -> AlphaRepository.save_simulation ngay (partial-safe).

Dùng: venv/Scripts/python.exe scripts/run_groundtruth.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values  # noqa: E402

# --- Nạp credential từ .env (blocker đã gỡ: dùng client của repo, không MCP) ---
env = dotenv_values(ROOT / ".env")
os.environ["WQ_EMAIL"] = env["WQ_EMAIL"].strip()
os.environ["WQ_PASSWORD"] = env["WQ_PASSWORD"]

from src.data.client import WQBrainClient  # noqa: E402
from src.simulation.simulator import AuthExpiredError, Simulator  # noqa: E402
from src.simulation.rate_limiter import RateLimiter  # noqa: E402
from src.simulation.simulator import SimulationError  # noqa: E402
from src.storage.db import make_engine, init_db, make_session_factory  # noqa: E402
from src.storage.repository import AlphaRepository  # noqa: E402

import httpx  # noqa: E402

DB_URL = "sqlite:///wq_alpha_phtrang1229_gmail_com.db"
REGION = "USA"
UNIVERSE = "TOP3000"
CONFIG_KEY = "USA|TOP3000|delay1|groundtruth"  # khớp persist_groundtruth.py
MANIFEST = ROOT / ".superpowers/sdd/groundtruth-manifest.json"
REPORT = ROOT / ".superpowers/sdd/groundtruth-report.md"
STATE = ROOT / ".superpowers/sdd/groundtruth-state.json"

TARGET_DONE = 50
MAX_CONSECUTIVE_ERRORS = 3  # stop-condition sau khi đã có >=1 success

# Settings khớp Phase 0 fixture + 50 sims hiện có của tuananhpo13.
SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "NONE",
    "truncation": 0.0,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
    "testPeriod": "P0Y0M",
    "maxTrade": "OFF",
}


def _looks_like_http_error(raw: dict) -> bool:
    """Phân biệt lỗi tầng POST/HTTP (quota/403/400) với lỗi sim-level (expr xấu).

    Simulator đặt raw['error']='status=ERROR: ...' cho sim ERROR; còn lỗi POST đặt
    resp.text vào raw['error']. Biểu thức ở đây đều đã thẩm định nên error thực tế
    gần như luôn là vấn đề tài khoản/quota -> đếm để dừng theo protocol."""
    err = str(raw.get("error", ""))
    return not err.startswith("status=")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exprs = [(v["expr"], v["family"]) for v in manifest["valid"]]

    engine = make_engine(DB_URL)
    init_db(engine)
    repo = AlphaRepository(make_session_factory(engine))

    client = WQBrainClient(email=os.environ["WQ_EMAIL"], password=os.environ["WQ_PASSWORD"])
    client.authenticate()  # dùng .wq_session nếu còn hạn

    # max_concurrent=1 -> tuần tự; min_delay=3s giữa 2 POST cho lịch sự.
    simulator = Simulator(client, rate_limiter=RateLimiter(max_concurrent=1, min_delay=3.0))

    n_success, n_error = 0, 0
    consecutive_errors = 0
    status_out = "DONE"
    sharpes: list[float] = []

    for i, (expr, family) in enumerate(exprs, 1):
        if n_success >= TARGET_DONE:
            print(f"[stop] Đã đạt {TARGET_DONE} sim non-null sharpe.")
            break

        # Resume: bỏ qua expr đã có sim non-error trong DB (tránh sim lại khi chạy lần 2).
        if repo.get_cached_simulation(expr, CONFIG_KEY) is not None:
            print(f"[{i}/60] SKIP (đã sim) | {expr}")
            continue

        # Retry lỗi mạng tạm thời (ReadTimeout/poll timeout) — KHÔNG crash cả run.
        result = None
        for attempt in range(1, 4):
            try:
                result = simulator.simulate(expr, settings=SETTINGS)
                break
            except AuthExpiredError as exc:
                print(f"[BLOCKED] {exc}")
                status_out = "BLOCKED"
                break
            except (httpx.HTTPError, SimulationError, TimeoutError) as exc:
                print(f"[{i}/60] timeout/lỗi mạng lần {attempt}/3 ({type(exc).__name__}) — thử lại sau 15s")
                time.sleep(15)
        if status_out == "BLOCKED":
            break
        if result is None:
            # 3 lần đều timeout -> coi như lỗi HTTP, ghi failure, đếm để dừng nếu liên tiếp.
            repo.record_failure(expression=expr, category="sim_error",
                                reason="3x network timeout", source=family)
            n_error += 1
            consecutive_errors += 1
            print(f"[{i}/60] ERR (3x timeout) | {expr}")
            if n_success >= 1 and consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"[stop] {consecutive_errors} lỗi liên tiếp — dừng.")
                status_out = "DONE_WITH_CONCERNS"
                break
            continue

        if result.status != "error" and result.sharpe is not None:
            repo.save_simulation(
                result=result, region=REGION, universe=UNIVERSE,
                source=family, config_key=CONFIG_KEY,
            )
            n_success += 1
            consecutive_errors = 0
            sharpes.append(result.sharpe)
            print(f"[{i}/60] OK sharpe={result.sharpe:.3f} ({n_success} done) | {expr}")
        else:
            repo.record_failure(
                expression=expr, category="sim_error",
                reason=str(result.raw.get("error", "no metrics")), source=family,
            )
            n_error += 1
            if _looks_like_http_error(result.raw):
                consecutive_errors += 1
            print(f"[{i}/60] ERR ({result.raw.get('error', '')[:80]}) | {expr}")
            if n_success >= 1 and consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"[stop] {consecutive_errors} lỗi HTTP liên tiếp sau khi đã có sim — dừng.")
                status_out = "DONE_WITH_CONCERNS"
                break

        # Lịch sự: nghỉ thêm sau mỗi 10 lượt submit.
        if i % 10 == 0:
            time.sleep(10)

    # --- Tổng kết từ DB (nguồn sự thật) ---
    from src.storage.models import SimulationModel
    sf = make_session_factory(engine)
    s = sf()
    try:
        rows = (
            s.query(SimulationModel.sharpe)
            .filter(SimulationModel.sharpe.isnot(None))
            .all()
        )
        db_sharpes = sorted(r[0] for r in rows)
    finally:
        s.close()

    n_db = len(db_sharpes)
    if status_out == "DONE" and n_db < TARGET_DONE and n_success < TARGET_DONE:
        # Hết 60 expr mà chưa đủ 50 — vẫn DONE (partial hữu ích) theo stop-condition.
        status_out = "DONE"

    dist = ""
    if db_sharpes:
        mn = db_sharpes[0]
        mx = db_sharpes[-1]
        med = db_sharpes[len(db_sharpes) // 2]
        dist = f"min={mn:.3f} median={med:.3f} max={mx:.3f}"

    # --- Ghi nối report (không ghi đè) ---
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## Resume run {ts}\n")
        f.write(f"- Trạng thái: {status_out}\n")
        f.write(f"- Đường dẫn DB: {DB_URL}\n")
        f.write(f"- Sim non-null sharpe trong DB: {n_db}\n")
        f.write(f"- Sim mới chạy thành công phiên này: {n_success}; lỗi: {n_error}\n")
        if dist:
            f.write(f"- Phân bố sharpe (DB): {dist}\n")

    # --- Cập nhật state.json ---
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state["status"] = status_out
    state["done_count"] = n_db
    state["last_run"] = ts
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSTATUS={status_out} DB_SIMS_NONNULL_SHARPE={n_db} dist=({dist})")


if __name__ == "__main__":
    main()
