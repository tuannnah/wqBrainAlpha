"""tools/verify_datasets.py — Verify LIVE dataset/field cho account hiện tại (Task 7).

Bối cảnh: log phiên 2026-07-12 cho thấy field-validity guard (`closed_loop_adapters.
build_closed_loop(known_fields=...)`) chặn seed short-interest (`days_to_cover`/`shares_short`
trong `src/generation/hypothesis_seeds.py`) vì 2 field này không có trong catalog cache của
account -> họ orthogonal "short_interest" CHƯA BAO GIỜ được thử trên Brain thật. Cardinal rule
#1 của dự án là KHÔNG bịa field: script này gọi thẳng API đọc để biết CHÍNH XÁC dataset/field
nào tồn tại cho account/scope hiện tại, thay vì suy đoán từ quy ước đặt tên.

Việc làm:
1. GET /data-sets cho scope USA/TOP3000/delay-1 -> toàn bộ dataset khả dụng.
2. Lọc dataset thuộc nhóm quan tâm cho seed hypothesis mới (Task 6, commit 2d6bdf8): short
   interest, news, earnings, insider, analyst, option, sentiment — match theo id/tên/category/
   subcategory, KHÔNG phân biệt hoa thường.
3. GET /data-fields?dataset.id=... cho từng dataset quan tâm -> field id + coverage.
4. Ghi `logs/verified_fields_<YYYYMMDD>.json` (`{dataset_id: [{"id":.., "coverage":..}, ...]}`)
   + in bảng tóm tắt ra console.

CHỈ gọi API đọc (GET /data-sets, GET /data-fields) — KHÔNG tạo simulation, an toàn quota.

Chạy (cần session còn hạn, đăng nhập qua `run.bat` -> mục 1 nếu chưa có/đã hết hạn):
    ./venv/Scripts/python.exe tools/verify_datasets.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

# Đảm bảo in được tiếng Việt có dấu trên console Windows (cp1252 mặc định) — khớp cách main.py
# tự reconfigure stdout/stderr; thiếu bước này script raise UnicodeEncodeError ngay khi print
# thông báo tiếng Việt (đã xác nhận khi chạy thật lần đầu, xem báo cáo Task 7).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from src.data.client import WQBrainClient  # noqa: E402

REGION = "USA"
UNIVERSE = "TOP3000"
DELAY = 1
INSTRUMENT_TYPE = "EQUITY"
PAGE_SIZE = 50

# Nhóm dataset quan tâm cho seed hypothesis mới (Task 6) — match theo id/tên/category/
# subcategory của dataset, không phân biệt hoa thường (đúng brief Task 7).
INTEREST_KEYWORDS: tuple[str, ...] = (
    "short interest",
    "shortinterest",
    "news",
    "earnings",
    "insider",
    "analyst",
    "option",
    "sentiment",
)


# --------------------------------------------------------------- phần thuần (test được)
def _dataset_search_text(dataset: dict) -> str:
    """Gộp id/tên/category/subcategory của 1 dataset thành 1 chuỗi thường để so khớp từ khoá.

    `category`/`subcategory` trong response WQ Brain thường là dict {"id":.., "name":..} nhưng
    xử lý cả trường hợp chuỗi thô để không vỡ nếu API trả dạng khác."""
    parts = [str(dataset.get("id", "")), str(dataset.get("name", ""))]
    for key in ("category", "subcategory"):
        val = dataset.get(key)
        if isinstance(val, dict):
            parts.append(str(val.get("id", "")))
            parts.append(str(val.get("name", "")))
        elif val is not None:
            parts.append(str(val))
    return " ".join(parts).lower()


def is_interest_dataset(dataset: dict, keywords: Sequence[str] = INTEREST_KEYWORDS) -> bool:
    """True nếu dataset thuộc 1 trong các nhóm quan tâm (so khớp không phân biệt hoa thường)."""
    text = _dataset_search_text(dataset)
    return any(kw in text for kw in keywords)


def filter_interest_datasets(
    datasets: list[dict], keywords: Sequence[str] = INTEREST_KEYWORDS
) -> list[dict]:
    """Tập con dataset thuộc nhóm quan tâm, giữ nguyên thứ tự trong `datasets`."""
    return [ds for ds in datasets if is_interest_dataset(ds, keywords)]


def field_entry(raw_field: dict) -> dict:
    """{"id":.., "coverage":..} từ 1 field JSON thô của /data-fields.

    `coverage` = None nếu API không trả (field mới/chưa có coverage) — vẫn ghi vào JSON để
    người đọc biết field tồn tại nhưng chưa có số liệu coverage, không âm thầm bỏ qua."""
    return {"id": raw_field.get("id", ""), "coverage": raw_field.get("coverage")}


def build_verified_json(
    datasets: list[dict], fields_by_dataset: dict[str, list[dict]]
) -> dict[str, list[dict]]:
    """Dựng {dataset_id: [{"id":.., "coverage":..}, ...]} theo đúng thứ tự `datasets`.

    Dataset không có trong `fields_by_dataset` (chưa tải được/lỗi giữa chừng) -> danh sách
    rỗng, KHÔNG bị lược khỏi kết quả (để người đọc thấy rõ dataset đó chưa xác nhận field)."""
    return {ds.get("id", ""): fields_by_dataset.get(ds.get("id", ""), []) for ds in datasets}


def render_summary_table(verified: dict[str, list[dict]]) -> str:
    """Bảng text tóm tắt: dataset | số field verify | vài field mẫu đầu."""
    header = f"{'dataset':30s} {'#field':>7s}  field mẫu (tối đa 3)"
    lines = [header, "-" * len(header)]
    for ds_id, entries in verified.items():
        sample = ", ".join(e.get("id", "") for e in entries[:3])
        lines.append(f"{ds_id:30s} {len(entries):7d}  {sample}")
    return "\n".join(lines)


def _print_login_guidance() -> None:
    print()
    print("Không xác thực được với WorldQuant BRAIN (session hết hạn hoặc chưa đăng nhập).")
    print("Hãy đăng nhập trước: chạy `run.bat` -> chọn mục 1 (Đăng nhập), rồi chạy lại:")
    print("    ./venv/Scripts/python.exe tools/verify_datasets.py")


# --------------------------------------------------------------- phần I/O (gọi API thật)
def fetch_datasets(
    client, region: str = REGION, universe: str = UNIVERSE, delay: int = DELAY,
    instrument_type: str = INSTRUMENT_TYPE, page_size: int = PAGE_SIZE,
) -> list[dict]:
    """GET /data-sets phân trang cho 1 scope — trả danh sách dataset thô (dict)."""
    offset = 0
    out: list[dict] = []
    while True:
        resp = client.get(
            "/data-sets",
            params={
                "instrumentType": instrument_type, "region": region, "delay": delay,
                "universe": universe, "limit": page_size, "offset": offset,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        if not results:
            break
        out.extend(results)
        offset += page_size
        total = payload.get("count")
        if total is not None and offset >= total:
            break
    return out


def fetch_fields_for_dataset(
    client, dataset_id: str, region: str = REGION, universe: str = UNIVERSE, delay: int = DELAY,
    instrument_type: str = INSTRUMENT_TYPE, page_size: int = PAGE_SIZE,
) -> list[dict]:
    """GET /data-fields?dataset.id=... phân trang — trả [{"id":.., "coverage":..}, ...]."""
    offset = 0
    out: list[dict] = []
    while True:
        resp = client.get(
            "/data-fields",
            params={
                "instrumentType": instrument_type, "region": region, "delay": delay,
                "universe": universe, "dataset.id": dataset_id, "limit": page_size, "offset": offset,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        if not results:
            break
        out.extend(field_entry(r) for r in results)
        offset += page_size
        total = payload.get("count")
        if total is not None and offset >= total:
            break
    return out


def _run(client) -> int:
    datasets = fetch_datasets(client)
    interest = filter_interest_datasets(datasets)
    print(f"Tổng dataset scope {REGION}/{UNIVERSE}/delay={DELAY}: {len(datasets)}")
    print(f"Dataset thuộc nhóm quan tâm ({', '.join(INTEREST_KEYWORDS)}): {len(interest)}")
    for ds in interest:
        print(f"  - {ds.get('id', '')}: {ds.get('name', '')}")

    fields_by_dataset: dict[str, list[dict]] = {}
    for ds in interest:
        ds_id = ds.get("id", "")
        fields_by_dataset[ds_id] = fetch_fields_for_dataset(client, ds_id)

    verified = build_verified_json(interest, fields_by_dataset)
    print()
    print(render_summary_table(verified))

    out_path = ROOT / "logs" / f"verified_fields_{date.today():%Y%m%d}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[đã ghi] {out_path}")
    return 0


def _build_client() -> WQBrainClient:
    return WQBrainClient(settings.wq_email or "", settings.wq_password or "")


def main(client=None) -> int:
    """Điểm vào script. Nhận `client` tuỳ chọn (để test bơm client giả) — mặc định dựng
    WQBrainClient thật (nạp session từ `.wq_session` nếu có, KHÔNG tự đăng nhập tương tác)."""
    client = client if client is not None else _build_client()

    if not client.is_session_valid():
        _print_login_guidance()
        return 1
    # Đã xác nhận session hợp lệ ở trên -> đánh dấu authenticated để client thật không tự gọi
    # lại authenticate() (có thể bật luồng QR tương tác) nếu tình cờ dính 401 giữa chừng.
    if hasattr(client, "_authenticated"):
        client._authenticated = True

    try:
        return _run(client)
    except Exception as exc:  # noqa: BLE001 — mọi lỗi mạng/API giữa chừng: in gọn, không traceback
        print(f"\nLỗi khi gọi API WQ Brain: {exc}")
        _print_login_guidance()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
