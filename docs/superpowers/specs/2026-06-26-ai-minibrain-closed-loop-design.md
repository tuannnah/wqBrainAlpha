# Vòng kín AI + MiniBrain → Brain SIM — Design

> Spec brainstorm 2026-06-26. Hợp nhất hai mặt của tool thành MỘT vòng kín tự động: menu +
> đăng nhập + nạp data → GP (MiniBrain) sinh ý tưởng + AI tăng cường → gate local +
> decorrelate → AI cải tiến mỗi ý tưởng ≤5 lần → đẩy Brain SIM (không trần) → lưu DB +
> feedback cải thiện engine → lặp đến hết quota. Tận dụng tối đa thành phần P0–P8 + luồng
> LLM/sim sẵn có; phần mới chủ yếu là orchestrator nối dây + menu + cầu DB + tối ưu prompt.

## Goal

Một công cụ dùng AI để **tạo công thức alpha** chạy vòng kín tự động: GPEngine (MiniBrain)
là trục sinh, AI (LLM) tăng cường (gieo ý tưởng + cải tiến công thức), mọi ứng viên qua gate
local + decorrelate trước khi tốn quota, ứng viên đạt được đẩy lên WorldQuant Brain SIM, kết
quả SIM lưu DB và feed ngược để cải thiện các engine. Vòng chạy đến khi Brain hết quota.

## Quyết định đã chốt (brainstorm)

1. **Kiến trúc:** MiniBrain GP làm trục sinh, AI tăng cường (không phải AI làm trục).
2. **Tự động hoàn toàn, không trần quota** — chạy đến khi Brain báo hết quota thì dừng.
3. **Patience = 5:** một ý tưởng cải tiến 5 lần không khá hơn → bỏ, lấy ý tưởng mới.
4. **Bốn feedback đều vào v1:** pool self-corr decorrelate; avoid-list + dead-field; tự tái
   calibrate ρ sau mỗi N sim; AI học từ outcome SIM (prompt tối ưu đẩy ý tưởng mới ít tương
   quan — vì **trùng thì không nộp được**).
5. **AI backend cấu hình `.env`:** `LLM_BACKEND` ∈ {deepseek, claude-cli, codex-cli} + model
   + quality. (DeepSeek hiện lỗi 402 → mặc định CLI miễn phí.)

## Mô hình hợp nhất (cốt lõi)

GP và RefinementLoop chia vai, không cạnh tranh:

- **GP + AI-seed = nguồn ý tưởng (trục sinh):** `GPEngine.run()` tiến hóa quần thể bare signal
  core local (không tốn quota); `GPSeedGenerator` (P7.8) + AI bơm hướng/ý tưởng mới, ưu tiên
  lạ và ít tương quan.
- **`build_shortlist` (P8) = lọc + decorrelate pool-aware** → danh sách "ý tưởng" ứng viên
  (cores) xếp hạng.
- **`RefinementLoop` + `AlphaRefiner` + referee (có sẵn) = bộ AI-cải-tiến + SIM + feedback:**
  với mỗi core ứng viên, AI refine ≤5 lần; mỗi biến thể qua `local_gate_fn` (gate local bắt
  buộc, D9) + decorrelate; không khá hơn sau 5 lần → referee 'abandon' → lấy core kế / AI đề
  xuất hướng mới (reseed qua `idea_generator`). Biến thể đạt → `Simulator` đẩy Brain SIM →
  kết quả persist.

→ Khớp "GP làm trục, AI tăng cường" + "tự động, bỏ sau 5 lần, chạy đến hết quota". GP là
nguồn nguyên liệu sinh; RefinementLoop là bộ refine+sim+feedback tiêu thụ luồng đó.

## Vòng một chu kỳ (data flow)

```
[Login + data]
   ↓ (cache-aware: thiếu→fetch API; có→load; option tải mới)
[GP sinh core + AI seed]  ── GPEngine.run() + GPSeedGenerator/AI
   ↓ final_population (đã eval local + persist)
[short-list]              ── build_shortlist(top_k, max_corr, pool) → cores ứng viên
   ↓ mỗi core là một "ý tưởng"
[AI refine ≤5/idea]       ── RefinementLoop: AlphaRefiner đề xuất biến thể
   ↓ mỗi biến thể: score_one (gate local) + decorrelate; referee patience=5
[đạt gate + ít tương quan] → [Brain SIM]  ── Simulator + WQBrainClient (KHÔNG trần)
   ↓ kết quả SIM
[persist DB + feedback]   ── (a)(b)(c)(d) bên dưới
   ↓
[lặp] đến khi Brain báo hết quota → dừng gọn, mọi thứ đã persist.
```

## Components

### Mới
- **`src/pipeline/closed_loop.py`** — orchestrator `ClosedLoop` (network-agnostic, nhận
  dependency injected): drive GP→shortlist→RefinementLoop-per-idea→sim→feedback một chu kỳ,
  lặp đến quota-exhausted. Trả `ClosedLoopReport` (số idea thử, số sim, số pass Brain, ρ mới,
  lý do dừng). Thuần điều phối — KHÔNG tự tạo client/DB (test bằng fake).
- **Cầu DB liên kết** evaluation MiniBrain ↔ kết quả SIM Brain: khi đẩy SIM, ghi kết quả
  (wq_alpha_id, sharpe/fitness/turnover Brain, self_corr Brain nếu có) gắn với expression
  MiniBrain (qua canonical_hash hoặc bảng bridge `sim_link`). Cho phép feedback so local↔Brain.
- **Mục menu mới** trong `main.py start`: "Chạy vòng kín AI+MiniBrain" → khởi tạo
  `ClosedLoop` từ phiên đăng nhập + panel + AI backend, chạy đến hết quota (Ctrl+C dừng tay).
- **Tối ưu prompt novelty** trong `AlphaRefiner`/idea prompt: đưa outcome SIM gần đây + pool
  hiện tại vào ngữ cảnh, hướng AI sinh công thức MỚI / ít tương quan (anti-crowding).

### Tận dụng lại (không sửa lõi trừ khi cần nối)
`GPEngine`/`GPSeedGenerator` (P7); `score_one`/`build_shortlist` (P8); `RefinementLoop`/
`AlphaRefiner`/referee + `local_gate_fn=score_local_gate` (có sẵn, đã enforce gate-trước-sim
D9); `Simulator`/`WQBrainClient`; `PoolCorrelation` + dead-field + `CalibrationHarness`;
`fetch-fields`/`fetch-operators` cache-aware + `_menu_login`.

## Bốn feedback (v1)

| # | Cơ chế | Nối vào | Ghi chú |
|---|--------|---------|---------|
| a | Pool self-corr decorrelate | alpha pass-SIM → PnL vào pool → `build_shortlist`/gate decorrelate vòng sau | Khung pool local đã có; ưu tiên dùng PnL thật khi có |
| b | Avoid-list + dead-field | fail gate local / fail SIM / abandon-sau-5 → ghi để GP/AI không sinh lại; field Brain từ chối → blacklist | dead-field model đã có (P5); avoid-list theo canonical_hash |
| c | Tự tái calibrate ρ | sau mỗi N sim mới → `CalibrationHarness` tính lại ρ Spearman (local vs Brain) | cảnh báo nếu ρ tụt dưới ngưỡng (độ tin ranking) |
| d | AI học từ SIM | outcome SIM gần đây + pool → prompt `AlphaRefiner`/idea | **trọng tâm: đẩy ý tưởng mới ít tương quan** (trùng = không nộp được) |

## Menu + cấu hình
- `run.ps1` → `main.py start`: Đăng nhập → tự load fields/operators (cache; thiếu→fetch;
  mục "tải lại" sẵn có) → **mục mới chạy vòng kín**.
- `.env`: `LLM_BACKEND` (deepseek | claude-cli | codex-cli) + model + quality; panel qua
  `--market-data-dir` (mặc định `data/market_yf2010`).

## Error handling
- Brain hết quota / lỗi sim → dừng vòng gọn (không crash), persist mọi thứ đã làm, in
  `ClosedLoopReport` lý do dừng.
- AI backend lỗi (vd DeepSeek 402) → báo rõ + dừng/đổi backend theo `.env`, không treo.
- Candidate parse/eval/TypeError/all-NaN → gate local trả fail (đã có ở `score_one`), không
  tốn sim.
- Ctrl+C → dừng tay, alpha đã sinh/sim vẫn lưu DB.

## Testing strategy
- Unit: `ClosedLoop` với fake GPEngine + fake refiner + fake simulator → kiểm luồng:
  short-list→refine≤5→abandon→reseed→promote-on-pass→persist→feedback; patience=5 dừng đúng;
  quota-exhausted dừng gọn. Không gọi mạng/AI thật.
- Cầu DB: ghi/đọc liên kết evaluation↔sim result; feedback đọc đúng.
- Feedback: pool lớn dần loại candidate trùng; avoid-list chặn tái sinh; calibrate chạy lại
  sau N sim; prompt chứa outcome+pool.
- Tích hợp (đánh dấu, có thể skip nếu thiếu mạng/AI): 1 chu kỳ nhỏ end-to-end với panel thật
  + fake simulator (không đốt Brain thật trong test).

## Constraints (kế thừa)
Python 3.12, full type hints, mypy --strict trên file mới, ruff sạch. No look-ahead /
delay-1 / stage separation. Thresholds chỉ ở `config/thresholds.py`. Determinism qua seed
inject. **Dependency rule (chốt rõ):** `src/pipeline/closed_loop.py` KHÔNG import `src.llm`/
`src.simulation` — nó nhận mọi dependency qua **Protocol structural** (giống `generate_many`
nhận `gp_engine`): `_RefinesIdeas` (refine 1 core ≤ patience lần), `_SimulatesAlpha` (đẩy SIM
Brain), `_GeneratesPopulation` (GP). Việc DỰNG cụ thể `RefinementLoop`/`AlphaRefiner`/
`Simulator`/`WQBrainClient` (đều ở `src.llm`/`src.simulation`) nằm ở **`main.py` (tầng app)**
rồi inject vào `ClosedLoop` — giữ B1 sạch. **Tiếng Việt giữ dấu** trong mọi docstring/comment
mới.

## Quy mô & phân rã
Lớn → plan chia phase: (1) cầu DB liên kết evaluation↔sim + persist kết quả SIM; (2)
orchestrator `ClosedLoop` nối GP→shortlist→RefinementLoop→sim (fake sim trong test); (3) bốn
feedback; (4) menu + `.env` AI config; (5) tối ưu prompt novelty; (6) review/merge/push +
chạy thử thật. Mỗi phase là deliverable test được độc lập.

## Self-review (đối chiếu yêu cầu người dùng)
- (1) menu + đăng nhập → §Menu, tận dụng `_menu_login`. ✔
- (2) data field/operator cache + tải mới tùy ý → §Menu cache-aware. ✔
- (3) MiniBrain + ghép AI sinh ý tưởng/công thức/cải tiến → §Mô hình hợp nhất (GP+seed +
  AlphaRefiner). ✔
- (4) chấm điểm, đạt thì đẩy Brain SIM → §Vòng (gate local → SIM). ✔
- (5) nhận kết quả SIM → DB → cải thiện engine → §Bốn feedback + cầu DB. ✔
- (vòng) tự động đến hết quota, bỏ ý tưởng sau 5 lần → §Quyết định 2/3. ✔
