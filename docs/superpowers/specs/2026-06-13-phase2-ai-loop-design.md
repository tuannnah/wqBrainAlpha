# GĐ2 — Vòng lặp AI cơ bản (giả thuyết + tinh chỉnh tham lam)

> Spec triển khai GIAI ĐOẠN 2 của `tailieu/BUILD_GUIDE_AI_alpha_tool.md`.
> Tiền đề: GĐ0, GĐ1 đã xong (login, fetch fields/operators, simulate, pre-filter,
> rate limiter). Nhánh: `phase2-ai-loop`.

## Mục tiêu

Một vòng lặp AI hoàn chỉnh chạy được qua CLI: cho một **hướng nghiên cứu**, sinh
giả thuyết có cấu trúc → mô tả → biểu thức FASTEXPR → mô phỏng thật → chấm điểm
đa chiều → tinh chỉnh tham lam nhắm chiều yếu nhất, lặp tới khi chạm trần
simulation hoặc hết cải thiện. Tích luỹ "alpha zoo" và "bộ nhớ thất bại".

Chưa làm: MCTS (GĐ6), AST-originality (GĐ3 — chỉ chừa hook), config-space tuning
(GĐ5). Nhánh template/GA hiện có **giữ nguyên, không phá**.

## Nguyên tắc

- Mọi lời gọi LLM và WQ đi qua interface tiêm được (dependency injection) để test
  bằng fake, không gọi mạng thật trong test.
- Phạt mềm cộng vào điểm; chỉ cú pháp sai + dưới ngưỡng cứng mới là cửa chặn cứng.
- Simulation là tài nguyên đắt nhất: cache theo hash (biểu thức), trần sim cấu
  hình được; cache hit không tính vào trần.

## Kiến trúc & module

Luồng mỗi vòng:
```
research direction
  → Hypothesis(observation, background, economic_rationale, implementation_spec)
  → description (bắt buộc) → expression FASTEXPR
  → pre-filter cú pháp (đã có) + hook độc đáo (chừa cho GĐ3)
  → simulate (cache theo hash) → ScoreVector đa chiều
  → hard-gate filter → đạt? lưu + vào zoo : ghi failure
  → chọn chiều yếu nhất → refine → simulate → giữ best
  (lặp tới khi chạm trần sim hoặc hết cải thiện)
```

Module mới trong `src/llm/`:

| Module | Nhiệm vụ | Task |
|--------|----------|------|
| `hypothesis.py` | `Hypothesis` dataclass + `HypothesisGenerator` sinh giả thuyết 4 phần từ research direction (JSON an toàn) | T2.3 |
| `translator.py` | `AlphaTranslator`: `Hypothesis → description → expression`, bắt buộc qua mô tả, lặp repair cú pháp qua pre-filter | T2.4, T2.5 |
| `refiner.py` | `AlphaRefiner`: nhận alpha+metrics+chiều yếu → mô tả cải tiến → biểu thức mới | T2.12 |
| `loop.py` | `RefinementLoop` orchestrator greedy: trần sim, cache, callback tiến độ | T2.14 |

Tái dùng: `DeepSeekClient` (T2.1/T2.2 đã xong), `PreFilter`, `Simulator`,
`generate_ideas` (giữ ở generator.py).

## Scoring / filter (`src/scoring/`)

- `ScoreVector` dataclass: điểm chuẩn hoá từng chiều `{sharpe, fitness,
  turnover_fit, drawdown_fit}` + `total`. `score()` cũ vẫn trả `total` (GA dùng) —
  không phá. (T2.8)
- `weakest_dimension(vector) -> str`: chọn chiều có điểm chuẩn hoá thấp nhất
  (có thể đặt trọng số ưu tiên). (T2.11)
- `filter.passes()` = cửa chặn cứng (giữ nguyên). Phạt mềm nằm trong ScoreVector.
  Ghi rõ trong docstring hard-gate vs soft-penalty. (T2.9)

## Dữ liệu (`src/storage/`)

- `AlphaModel` thêm: `hypothesis` (Text/JSON), `description` (Text),
  `parent_id` (String, nullable — lineage tinh chỉnh). (T2.6)
- `SimulationModel` thêm: `expr_hash` (String, index) để cache sim theo hash. (T1.15)
- `FailureModel` mới: `id, expression, category, reason, source, created_at`.
  category ∈ {`syntax`, `low_score`, `hypothesis_mismatch`}. (T2.13)
- Alpha zoo (T2.10) = **view**: `AlphaRepository.zoo(limit)` trả alpha đã pass
  (join simulation status=passed, sort theo score) kèm hypothesis/description —
  không bảng mới.
- Migration nhẹ idempotent trong `init_db`: ALTER thêm cột thiếu cho DB cũ
  (đọc `PRAGMA table_info`, thêm cột chưa có). Không nhân đôi.

Repository thêm:
- `save_alpha(..., hypothesis=None, description=None, parent_id=None)`.
- `get_cached_simulation(expr) -> SimulationModel | None` (theo expr_hash).
- `save_simulation(..., expr_hash=...)`.
- `record_failure(expression, category, reason, source)`.
- `recent_failures(limit)`, `zoo(limit)`.

## Vòng greedy (`loop.py`)

```
@dataclass AlphaCandidate: hypothesis, description, expression, parent_id=None
@dataclass LoopProgress: sims_used, best_total, phase, detail   # cho callback
@dataclass LoopResult: best_candidate, best_vector, history, zoo_added, failures, sims_used

class RefinementLoop:
    __init__(hypothesis_gen, translator, refiner, simulator, prefilter,
             repo, score_vector_fn, hard_filter_fn,
             max_simulations=20, no_improve_patience=3, region, universe)
    run(direction, on_progress=None) -> LoopResult
```

Logic `run`:
1. Sinh hypothesis → translate → candidate gốc.
2. `evaluate(candidate)`:
   - pre-filter cú pháp; lỗi → `record_failure(syntax)` → trả None.
   - cache theo hash: hit → dùng lại metrics (không tăng sim count).
   - hết trần sim → trả None (không gọi WQ).
   - simulate → tăng count → ScoreVector → lưu sim (kèm expr_hash).
   - hard filter: đạt → `save_alpha` (kèm bộ ba) + đánh dấu zoo; trượt →
     `record_failure(low_score)`.
3. Greedy: giữ `best`. Mỗi vòng: `weak = weakest_dimension(best_vector)` →
   `refiner.refine(best, metrics, weak)` → candidate mới (parent_id=best) →
   evaluate. Tốt hơn → cập nhật best, reset patience; không → patience++.
   Dừng khi hết trần sim hoặc patience vượt `no_improve_patience`.
4. `on_progress` phát sự kiện cho CLI (sim count, best total, chiều đang nhắm).

Trần sim mặc định 20; cache hit không tính. (đã chốt với người dùng)

## CLI (demo GĐ2)

- Lệnh: `python main.py research --direction "..." --max-sims 20 [--no-improve 3]`.
- Wizard `start`: thêm mục "Nghiên cứu alpha bằng AI (giả thuyết + tinh chỉnh)".
- Hiển thị `rich.Progress` (giống GA) + kết quả cuối: hypothesis 4 phần, mô tả,
  biểu thức, metrics, số alpha thêm vào zoo, số lần sim.

## Kế hoạch test (TDD, dùng fake)

Fakes (mở rộng `tests/fakes.py`):
- `FakeDeepSeek`: hàng đợi nội dung JSON trả về theo lượt `complete()`.
- `FakeSimulator`: map expr→metrics hoặc trả tuần tự; đếm số lần gọi.

Test theo module:
- `hypothesis`: parse JSON 4 phần; thiếu phần → vẫn an toàn (không crash).
- `translator`: bắt buộc qua description; repair cú pháp khi expr lỗi rồi sửa được.
- `scoring`: ScoreVector các chiều; `weakest_dimension` chọn đúng chiều thấp nhất.
- `refiner`: prompt chứa đúng chiều yếu cần cải thiện; trả expr mới hợp lệ.
- `loop`: tôn trọng `max_simulations`; best total cải thiện qua các vòng;
  zoo tích luỹ alpha pass; failure được ghi; cache → không sim trùng.
- `storage`: lưu/đọc bộ ba (hypothesis, description, parent_id); `record_failure`
  + `recent_failures`; `get_cached_simulation` hit theo hash; `zoo()` trả alpha pass;
  migration ALTER cột không nhân đôi.

## Acceptance (theo guide)

- Cho một hướng nghiên cứu → sinh alpha hợp lệ kèm giả thuyết + mô tả; vòng tự sửa
  cú pháp hoạt động.
- Vòng lặp chạy trọn nhiều vòng, điểm tốt nhất cải thiện theo thời gian (log/chứng
  cứ); alpha zoo tích luỹ; tận dụng cache không sim trùng.
- Toàn bộ test (cũ + mới) pass; không phá nhánh template/GA.

## Thứ tự triển khai (mỗi bước 1 commit, TDD)

1. Storage: models + migration + repository methods (test trước).
2. Scoring: ScoreVector + weakest_dimension (test trước).
3. `hypothesis.py` (test trước).
4. `translator.py` (test trước).
5. `refiner.py` (test trước).
6. `loop.py` RefinementLoop (test trước).
7. CLI `research` + mục wizard; chạy `pytest -q` toàn bộ.
