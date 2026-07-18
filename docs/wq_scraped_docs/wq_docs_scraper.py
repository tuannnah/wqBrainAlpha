#!/usr/bin/env python3
"""Snapshot tài liệu WorldQuant Brain về local (markdown + SQLite) cho MiniBrain/RAG.

Có HAI chế độ:

1) API (khuyên dùng) — `--api`
   Dùng lại WQBrainClient của dự án: đăng nhập bằng WQ_EMAIL/WQ_PASSWORD trong .env
   (tự xử lý QR/persona nếu tài khoản yêu cầu), rồi tải tài liệu qua REST API:
       GET /tutorials              -> cây danh mục + trang
       GET /tutorial-pages/{id}    -> nội dung từng trang (HTML)
   Không cần trình duyệt/playwright.

       pip install markdownify beautifulsoup4      # httpx/dotenv/loguru đã có sẵn
       python docs/wq_scraped_docs/wq_docs_scraper.py --api

2) BROWSER (dự phòng) — `--login` rồi `--crawl`
   Render SPA bằng headless Chromium (đăng nhập thủ công 1 lần). Chỉ dùng khi API
   đổi cấu trúc.

       pip install playwright markdownify beautifulsoup4
       python -m playwright install chromium
       python docs/wq_scraped_docs/wq_docs_scraper.py --login
       python docs/wq_scraped_docs/wq_docs_scraper.py --crawl

Kết quả (đặt tại thư mục hiện hành):
    out/docs/<slug>.md        một file markdown / trang
    out/tutorials.json        cây /tutorials thô (để tra cứu / debug cấu trúc)
    out/docs.sqlite           bảng docs(url, path, title, markdown, fetched_at)
"""

import argparse
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

# Console Windows mặc định cp1252 -> vỡ khi in tiếng Việt/emoji. Ép UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

BASE = "https://platform.worldquantbrain.com"      # frontend SPA (chế độ browser)
API_BASE = "https://api.worldquantbrain.com"       # REST API (chế độ --api)
DOCS_ROOT = f"{BASE}/learn/documentation"
STATE_FILE = "wq_state.json"                        # session playwright (chế độ browser)
SESSION_FILE = ".wq_session"                        # session WQBrainClient (chế độ --api)
OUT_DIR = Path("out")
DOCS_DIR = OUT_DIR / "docs"
DB_PATH = OUT_DIR / "docs.sqlite"
TUTORIALS_JSON = OUT_DIR / "tutorials.json"

# Chế độ browser: nhận diện link tài liệu + container bài viết.
NAV_LINK_SELECTOR = 'a[href^="/learn/documentation/"]'
CONTENT_SELECTORS = ["main", "article", '[class*="content"]', '[class*="doc"]']
RATE_LIMIT_S = 0.35                                 # nghỉ giữa các request cho lịch sự

# Khóa JSON chứa tên/tiêu đề node trong cây /tutorials (danh mục lẫn trang).
TITLE_KEYS = ("name", "title", "label")


# ------------------------------------------------------------- tiện ích chung
def init_db() -> sqlite3.Connection:
    OUT_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS docs(
               url TEXT PRIMARY KEY, path TEXT, title TEXT,
               markdown TEXT, fetched_at TEXT)"""
    )
    return conn


def _slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "untitled"


def _html_to_markdown(html: str) -> str:
    """HTML -> markdown ATX; nếu không phải HTML thì trả nguyên văn."""
    if not html:
        return ""
    if "<" not in html:                              # đã là text/markdown thuần
        return html.strip()
    from markdownify import markdownify as md       # import lazy
    markdown = md(html, heading_style="ATX")
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def _markdown_table(data: list) -> str:
    if not data:
        return ""
    rows = [[str(cell).replace("\n", "<br>").replace("|", "\\|") for cell in row] for row in data]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_block(block: dict) -> str:
    """Một block trong `content` (list) của /tutorial-pages/{id} -> đoạn markdown.

    Cấu trúc thực tế (dò qua nhiều trang mẫu): mỗi block có `type` +`value`:
      TEXT -> value là HTML; HEADING -> {level, content}; IMAGE -> {title, url};
      TABLE -> {data, firstRowIsTableHeader}; EQUATION -> chuỗi LaTeX;
      SIMULATION_EXAMPLE -> {settings, type, regular/...}.
    Loại lạ (chưa gặp) -> dump JSON để KHÔNG âm thầm mất nội dung.
    """
    btype = block.get("type")
    value = block.get("value")

    if btype == "TEXT":
        return _html_to_markdown(value if isinstance(value, str) else "")

    if btype == "HEADING" and isinstance(value, dict):
        try:
            level = max(1, min(6, int(value.get("level", 2))))
        except (TypeError, ValueError):
            level = 2
        return f"{'#' * level} {value.get('content', '')}".strip()

    if btype == "IMAGE" and isinstance(value, dict):
        alt = value.get("title") or "image"
        url = value.get("url", "")
        return f"![{alt}]({url})" if url else ""

    if btype == "TABLE" and isinstance(value, dict):
        return _markdown_table(value.get("data") or [])

    if btype == "EQUATION":
        return f"$$\n{value}\n$$" if isinstance(value, str) else ""

    if btype == "SIMULATION_EXAMPLE" and isinstance(value, dict):
        settings = value.get("settings") or {}
        expr = value.get("regular") or value.get(str(value.get("type", "")).lower()) or ""
        settings_line = ", ".join(f"{k}={v}" for k, v in settings.items())
        return f"```text\n# Simulation settings: {settings_line}\n{expr}\n```"

    # Loại chưa biết: giữ nguyên dưới dạng JSON để không mất dữ liệu.
    import json as _json

    return f"```json\n// block type chưa hỗ trợ: {btype}\n{_json.dumps(block, ensure_ascii=False, indent=1)}\n```"


def _render_content(content) -> str:
    if isinstance(content, str):
        return _html_to_markdown(content)
    if isinstance(content, list):
        parts = [_render_block(b) for b in content if isinstance(b, dict)]
        markdown = "\n\n".join(p for p in parts if p)
        return re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return ""


# --------------------------------------------------------------- chế độ API
def _project_root() -> Path:
    # docs/wq_scraped_docs/wq_docs_scraper.py -> gốc dự án là 2 cấp trên.
    return Path(__file__).resolve().parents[2]


def _load_env() -> tuple[str, str]:
    root = _project_root()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass
    import os

    email = os.environ.get("WQ_EMAIL") or os.environ.get("WQB_EMAIL") or ""
    password = os.environ.get("WQ_PASSWORD") or os.environ.get("WQB_PASSWORD") or ""
    if not email:
        email = input("WQ_EMAIL: ").strip()
    if not password:
        import getpass

        password = getpass.getpass("WQ_PASSWORD: ").strip()
    return email, password


def _wait_for_qr_scan(prompt: str, seconds: int = 60) -> None:
    """Thay cho input(): môi trường không có stdin tương tác (EOFError khi gọi
    input() qua tool chạy lệnh nền) nên không thể chờ người dùng nhấn Enter.
    Thay vào đó chỉ CHỜ một khoảng đủ để quét QR bằng điện thoại rồi thử lại
    (WQBrainClient tự lặp lại việc này tối đa 3 lần -> tổng ~3*seconds)."""
    print(f"⏳ Không có bàn phím tương tác — tự chờ {seconds}s để bạn quét QR bằng app WorldQuant BRAIN...")
    time.sleep(seconds)


def _make_client():
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.data.client import WQBrainClient

    email, password = _load_env()
    return WQBrainClient(
        email,
        password,
        session_file=root / SESSION_FILE,
        confirmation_input=_wait_for_qr_scan,
    )


def _first_str(d: dict, keys) -> str:
    """Giá trị chuỗi dài nhất trong các khóa ứng viên (nội dung phong phú nhất)."""
    best = ""
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best


def _walk_pages(node, trail, out):
    """Duyệt đệ quy cây /tutorials, thu (id, tên, đường-dẫn-danh-mục)."""
    if isinstance(node, dict):
        name = _first_str(node, TITLE_KEYS)
        nid = node.get("id")
        if nid is not None and (isinstance(nid, str) or isinstance(nid, int)):
            out.append((str(nid), name, list(trail)))
        new_trail = trail + ([name] if name else [])
        for v in node.values():
            _walk_pages(v, new_trail, out)
    elif isinstance(node, list):
        for item in node:
            _walk_pages(item, trail, out)


def api_crawl():
    import json

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    conn = init_db()
    client = _make_client()
    client.authenticate()                            # xử lý cả QR/persona nếu cần

    resp = client.get("/tutorials")
    resp.raise_for_status()
    tutorials = resp.json()
    TUTORIALS_JSON.write_text(json.dumps(tutorials, ensure_ascii=False, indent=2), encoding="utf-8")

    candidates: list[tuple[str, str, list[str]]] = []
    _walk_pages(tutorials, [], candidates)
    # Dedupe theo id, giữ lần gặp đầu (đường dẫn danh mục đầy đủ nhất).
    seen: dict[str, tuple[str, list[str]]] = {}
    for nid, name, trail in candidates:
        if nid not in seen:
            seen[nid] = (name, trail)
    print(f"Tìm thấy {len(seen)} id ứng viên trong /tutorials")

    ok = used_slugs = 0
    total = len(seen)
    for i, (nid, (name, trail)) in enumerate(seen.items(), 1):
        try:
            r = client.get(f"/tutorial-pages/{nid}")
            if r.status_code != 200:
                print(f"[{i}/{total}] SKIP  {nid} (HTTP {r.status_code}) — có thể là danh mục")
                time.sleep(RATE_LIMIT_S)
                continue
            page = r.json()
            title = page.get("title") or _first_str(page, TITLE_KEYS) or name or nid
            markdown = _render_content(page.get("content"))
            if not markdown:
                print(f"[{i}/{total}] EMPTY {nid} ({title})")
                time.sleep(RATE_LIMIT_S)
                continue

            parts = [_slugify(t) for t in trail if t] or ["_"]
            parts.append(_slugify(title))
            slug = "/".join(parts)
            fpath = DOCS_DIR / f"{slug}.md"
            if fpath.exists():                       # tránh đè trùng tên
                slug = f"{slug}-{nid}"
                fpath = DOCS_DIR / f"{slug}.md"
            fpath.parent.mkdir(parents=True, exist_ok=True)

            url = f"{API_BASE}/tutorial-pages/{nid}"
            header = " / ".join([t for t in trail if t] + [title])
            fpath.write_text(f"# {header}\n\n<{url}>\n\n{markdown}\n", encoding="utf-8")
            conn.execute(
                "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?)",
                (url, slug, title, markdown, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            ok += 1
            used_slugs += 1
            print(f"[{i}/{total}] OK    {slug}  ({len(markdown)} ký tự)")
        except Exception as e:
            print(f"[{i}/{total}] ERROR {nid}: {e}")
        time.sleep(RATE_LIMIT_S)

    client.close()
    conn.close()
    print(f"\nXong. Tải được {ok} trang. Markdown ở {DOCS_DIR}/, DB ở {DB_PATH}, cây ở {TUTORIALS_JSON}")


# ---------------------------------------------------------- chế độ BROWSER
def login():
    """Mở trình duyệt thật, để user đăng nhập, rồi lưu session."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(f"{BASE}/sign-in", wait_until="networkidle")
        print("Đăng nhập trong cửa sổ trình duyệt, xong rồi nhấn Enter ở đây...")
        input()
        ctx.storage_state(path=STATE_FILE)
        print(f"Đã lưu session vào {STATE_FILE}")
        browser.close()


def _pick_content_html(page) -> str | None:
    for sel in CONTENT_SELECTORS:
        el = page.query_selector(sel)
        if el:
            html = el.inner_html().strip()
            if len(html) > 200:
                return html
    return None


def discover_urls(page) -> list[str]:
    page.goto(DOCS_ROOT, wait_until="networkidle")
    page.wait_for_timeout(2000)
    hrefs = page.eval_on_selector_all(
        NAV_LINK_SELECTOR, "els => els.map(e => e.getAttribute('href'))"
    )
    urls = sorted({urljoin(BASE, h) for h in hrefs if h})
    urls.insert(0, DOCS_ROOT)
    return sorted(set(urls))


def _slug_for(url: str) -> str:
    path = urlparse(url).path.replace("/learn/documentation", "").strip("/")
    return path or "index"


def crawl():
    from playwright.sync_api import sync_playwright

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    conn = init_db()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(storage_state=STATE_FILE)
        page = ctx.new_page()

        urls = discover_urls(page)
        print(f"Tìm thấy {len(urls)} trang tài liệu")

        for i, url in enumerate(urls, 1):
            try:
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(1200)
                html = _pick_content_html(page)
                if not html:
                    print(f"[{i}/{len(urls)}] EMPTY  {url}")
                    continue
                title = (page.title() or "").replace(" | WorldQuant BRAIN", "").strip()
                markdown = _html_to_markdown(html)

                slug = _slug_for(url)
                fpath = DOCS_DIR / f"{slug}.md"
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(f"# {title}\n\n<{url}>\n\n{markdown}\n", encoding="utf-8")

                conn.execute(
                    "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?)",
                    (url, slug, title, markdown, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                print(f"[{i}/{len(urls)}] OK     {slug}  ({len(markdown)} ký tự)")
            except Exception as e:
                print(f"[{i}/{len(urls)}] ERROR  {url}: {e}")
            time.sleep(RATE_LIMIT_S)

        browser.close()
    conn.close()
    print(f"\nXong. Markdown ở {DOCS_DIR}/, DB ở {DB_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", action="store_true", help="tải tài liệu qua REST API dùng .env (khuyên dùng)")
    ap.add_argument("--login", action="store_true", help="[browser] đăng nhập thủ công, lưu session")
    ap.add_argument("--crawl", action="store_true", help="[browser] crawl SPA bằng session đã lưu")
    args = ap.parse_args()
    if args.api:
        api_crawl()
    elif args.login:
        login()
    elif args.crawl:
        if not Path(STATE_FILE).exists():
            raise SystemExit("Chưa có session browser — chạy --login trước.")
        crawl()
    else:
        ap.print_help()
