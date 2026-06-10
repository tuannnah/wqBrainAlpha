# Tiến Độ Triển Khai — Autonomous Alpha Research

> File này theo dõi tiến độ code dự án "Nghiên cứu Alpha tự động".
> Thiết kế: [`specs/...autonomous-alpha-research-design.md`](superpowers/specs/2026-06-09-autonomous-alpha-research-design.md)
> Kế hoạch: [`plans/...autonomous-alpha-research.md`](superpowers/plans/2026-06-09-autonomous-alpha-research.md)

Cập nhật lần cuối: 2026-06-10

## Quyết Định Thiết Kế (sau buổi brainstorm 2026-06-10)

- Giữ **nguyên thiết kế đầy đủ** theo spec, không cắt scope.
- Dataset/field **kéo từ WorldQuant Brain về SQLite** (multi-snapshot có nhãn), người dùng không cần biết data.
- DeepSeek sinh ý tưởng + alpha gốc + biến thể. Cần `DEEPSEEK_API_KEY`.
- Validate đầy đủ: lark parser + fingerprint + similarity + type/scope/settings.
- **Không auto-submit** — alpha đạt chuẩn vào hàng chờ duyệt (`PENDING_REVIEW`).
- Lượt chạy dừng khi đủ **N=10** alpha qualify hoặc lệnh `quit`.

## Tổng Quan Phase

| Phase | Nội dung | Trạng thái |
|---|---|---|
| ✅ Đăng nhập tương tác | getpass + xác thực QR (tính năng riêng) | Hoàn thành (đã commit) |
| ✅ Phase 1 | Storage + WorldQuant metadata | Hoàn thành (38 test OK) |
| ✅ Phase 2 | DeepSeek generation + expression validation | Hoàn thành (59 test OK) |
| ⬜ Phase 3 | Research engine + logging + stop control | Chưa bắt đầu |
| ⬜ Phase 4 | CLI migration + packaging + docs | Chưa bắt đầu |

## Phase 1 — Storage & Metadata

| Task | Mô tả | File chính | Trạng thái |
|---|---|---|---|
| 1 | Models, config, account paths | `research_models.py`, `research_config.py`, `research_config.json`, `account_storage.py` | ✅ |
| 2 | Metadata snapshot store + FTS5 | `metadata_store.py` | ✅ |
| 3 | WorldQuant client (metadata + simulation) | `worldquant_client.py` | ✅ |
| 4 | Đồng bộ metadata resumable | `metadata_sync.py` | ✅ |
| 5 | Kiểm chứng Phase 1 | — | ✅ |

## Phase 2 — DeepSeek & Validation

| Task | Mô tả | File chính | Trạng thái |
|---|---|---|---|
| 1 | Research DB + audit trail | `research_store.py` | ✅ |
| 2 | DeepSeek JSON client | `deepseek_client.py` | ✅ |
| 3 | Candidate selection cục bộ | `candidate_selector.py` | ✅ |
| 4 | FASTEXPR parser + fingerprint | `expression_parser.py` | ✅ |
| 5 | Validation theo metadata | `expression_validator.py` | ✅ |
| 6 | Qualification + quality gate | `qualification.py` | ✅ |
| 7 | Kiểm chứng Phase 2 | — | ✅ |

## Phase 3 — Engine & Control

Xem [`plans/...phase-3-engine-control.md`](superpowers/plans/2026-06-09-alpha-research-phase-3-engine-control.md) — chi tiết bổ sung khi tới Phase 3.

## Phase 4 — CLI & Packaging

Xem [`plans/...phase-4-cli-packaging.md`](superpowers/plans/2026-06-09-alpha-research-phase-4-cli-packaging.md) — chi tiết bổ sung khi tới Phase 4.

## Nhật Ký

- **2026-06-10**: Brainstorm xong, chốt giữ nguyên thiết kế. Bắt đầu Phase 1. Baseline: 18 test cũ pass, `pip check` sạch.
- **2026-06-10**: Hoàn thành Phase 1 (5 task, TDD, mỗi task 1 commit). `brain_batch_alpha.py` giờ kế thừa `WorldQuantClient`. Tổng 38 test OK, compile + pip check + diff sạch.
- **2026-06-11**: Hoàn thành Phase 2 (7 task). Đã cài `lark`. Thêm research_store, deepseek_client, candidate_selector, expression_parser, expression_validator, qualification. Tổng 59 test OK.
