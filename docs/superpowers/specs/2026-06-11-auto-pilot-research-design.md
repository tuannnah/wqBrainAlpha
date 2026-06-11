# Thiết kế: Auto-Pilot Nghiên cứu Alpha (research-only)

> Spec cho chế độ chạy tự động ("auto-pilot") của WQ Auto-Alpha Tool. Mục tiêu:
> một lệnh duy nhất chạy toàn bộ vòng nghiên cứu — sinh → lọc → mô phỏng → giữ
> lại alpha tốt & khác nhau — tới khi đủ số lượng mong muốn, KHÔNG tự nộp.

## 1. Mục tiêu & phạm vi

**Mục tiêu:** Bấm một nút (`python main.py auto`) để tool tự chạy vòng nghiên cứu
alpha và dừng khi tìm đủ K alpha vừa **đạt ngưỡng** vừa **khác nhau về cấu trúc**.

**Trong phạm vi:**
- Tự chuẩn bị: đăng nhập (tái dùng session), fetch fields/operators (dùng cache).
- Vòng lặp sinh → pre-filter → chống trùng → simulate → chấm điểm → lọc → giữ.
- Sinh alpha bằng **DeepSeek (LLM) làm seed + GA tiến hóa**.
- **Chống trùng cấu trúc** bằng "vân tay" (fingerprint) — bước local, trước simulate.
- **Cache mô phỏng theo hash** — không simulate lại biểu thức đã chạy.
- Dừng khi đủ K alpha tốt & khác nhau, hoặc người dùng gõ `quit`.

**Ngoài phạm vi (làm sau / người dùng tự làm):**
- ❌ Correlation check — người dùng tự kiểm tra thủ công.
- ❌ Tự động nộp (submit) — người dùng review rồi tự `submit`.
- ❌ Chạy mô phỏng song song — để sau (Giai đoạn C của module simulation).
- ❌ IS/OOS — chưa thuộc phạm vi bản này.

## 2. Pipeline (chi phí tăng dần: rẻ→đắt)

```
[1] CHUẨN BỊ (một lần, dùng cache nếu có)
    Đăng nhập (.wq_session) → Fetch fields (DB) → Fetch operators (DB)

[2] VÒNG LẶP NGHIÊN CỨU — tới khi đủ K alpha tốt & khác nhau, hoặc 'quit'
    ◀── RẺ (local, miễn phí) ──▶          ◀──── ĐẮT (gọi WQ) ────▶
    SINH ─▶ PRE-FILTER ─▶ CHỐNG TRÙNG ─▶ SIMULATE ─▶ LỌC ─▶ GIỮ DB
   LLM+GA   (cú pháp)     (vân tay)      (cache hash) (ngưỡng)
                          trùng → BỎ                  đạt → giữ
                          (chưa tốn quota)

[3] KẾT THÚC: in bảng K alpha (đạt ngưỡng, đa dạng). KHÔNG tự nộp.
```

**Thứ tự cố ý:** mọi bước lọc rẻ (pre-filter, chống trùng) chạy **trước** bước
đắt nhất (simulate) để không phí quota mô phỏng đồ hỏng hoặc trùng.

## 3. Thành phần

| Module | Nhiệm vụ | Trạng thái |
|---|---|---|
| `AutoPilot` (`src/pipeline/autopilot.py`) | Điều phối 3 chặng; quản vòng lặp + điều kiện dừng; lắng nghe `quit`; in tiến độ từng chặng | **Mới** |
| `DedupChecker` (`src/pipeline/dedup.py`) | Tạo fingerprint cấu trúc từ AST; so với tập đã giữ (và DB); trùng → loại | **Mới** |
| `CachedSimulator` (`src/pipeline/cached_simulator.py`) | Bọc `Simulator`: tra DB theo (hash expr + scope) trước khi gọi WQ; chưa có thì simulate rồi lưu | **Mới** |
| `FieldRepository` / `OperatorRepository` | fetch-once + cache | Tái dùng |
| `LLMAlphaGenerator` + `GeneticOptimizer` | sinh ý tưởng (DeepSeek) + tiến hóa | Tái dùng |
| `scorer` + `FilterThresholds` (`passes`) | chấm điểm + lọc "alpha tốt" | Tái dùng |
| `AlphaRepository` | lưu alpha/kết quả | Tái dùng (mở rộng) |

### 3.1 Thuật toán chống trùng (DedupChecker)

Mục tiêu: hai biểu thức **viết khác nhau nhưng cùng bản chất** được coi là trùng.

1. Parse biểu thức thành AST (`ast_utils.parse_expression`).
2. **Chuẩn hóa về dạng chính tắc (canonical):**
   - Render lại từ AST để bỏ khác biệt khoảng trắng/định dạng.
   - Với toán tử **giao hoán** (`+`, `*`, và hàm đối xứng như `ts_corr`), **sắp xếp
     các toán hạng theo thứ tự chính tắc** để `a + b` ≡ `b + a`.
3. Hash chuỗi chính tắc → **fingerprint** (vd SHA-1 hex).
4. Alpha mới trùng nếu fingerprint đã có trong: (a) tập alpha đã giữ trong lần
   chạy này, hoặc (b) bảng fingerprint trong DB (các lần chạy trước).
5. Lưu fingerprint khi giữ một alpha tốt → các lần sau không giữ lại đồ trùng.

> Phạm vi bản này: chống trùng **cấu trúc** (đủ theo yêu cầu). Chống trùng theo
> correlation KHÔNG nằm trong phạm vi.

### 3.2 Cache mô phỏng (CachedSimulator)

- Khóa cache: `(fingerprint, region, universe, delay)` — `fingerprint` dùng đúng
  hàm chính tắc hóa của `DedupChecker` (§3.1), nên hai biểu thức tương đương về
  cấu trúc cũng dùng chung kết quả cache.
- Trước khi gọi WQ: tra bảng `simulations` theo khóa. Có → trả `SimulationResult`
  dựng lại từ DB (không gọi WQ). Chưa → `Simulator.simulate()` rồi lưu kèm
  `expr_fingerprint`.
- Tiết kiệm quota khi GA gặp lại cùng (hoặc tương đương) biểu thức qua các thế hệ.

## 4. Mô hình dữ liệu (thay đổi)

- Bảng `simulations` (đã có): thêm cột `expr_fingerprint` (để cache theo dạng
  chính tắc + truy vấn nhanh). Khóa tra cache logic: fingerprint + scope.
- Bảng mới `alpha_fingerprints`: `fingerprint` (PK), `expression`, `created_at`
  — để chống trùng xuyên suốt nhiều lần chạy.

## 5. Điều kiện dừng & điều khiển

- **Đủ K alpha** vừa đạt ngưỡng (`FilterThresholds`: Sharpe≥1.25, Fitness>1.0,
  Turnover 0.01–0.70, Drawdown<0.20) vừa **fingerprint khác nhau** → tự dừng.
- **`quit`**: luồng nền lắng nghe bàn phím; gõ `quit`+Enter → dừng an toàn sau
  alpha đang xử lý (không cắt giữa chừng).
- Không có trần cứng số simulation (người dùng tự canh quota) — theo yêu cầu.
- `--target K` (mặc định 10) đặt số alpha cần tìm.

## 6. Xử lý lỗi (tái dùng cơ chế có sẵn)

- Session hết hạn giữa chừng → client tự re-auth (đã có).
- Rate limit 429 → backoff theo `Retry-After` (đã có).
- Simulation timeout → đánh dấu `error`, bỏ qua, **không dừng vòng lặp**.
- Alpha bị WQ từ chối/cú pháp → ghi log lý do, bỏ qua, đi tiếp.
- **Nguyên tắc: một alpha lỗi không bao giờ làm dừng cả pipeline.**

## 7. Điểm vào (người dùng "bấm" ở đâu)

1. CLI: `python main.py auto --target 10` (tùy chọn `--region/--universe/--delay`,
   `--seed-llm/--no-seed-llm`).
2. Wizard (`run.bat`): thêm mục **"9) Chạy tự động (auto-pilot)"**.

Khi chạy, in tiến độ theo từng chặng của pipeline (§2) và đếm số alpha tốt đã tìm.

## 8. Kiểm thử (mock, không gọi WQ thật)

- `DedupChecker`: `a + b` ≡ `b + a`; khác khoảng trắng ≡ nhau; biểu thức khác
  bản chất ≠ nhau; trùng với DB bị loại.
- `CachedSimulator`: lần 2 cùng biểu thức KHÔNG gọi simulator; khác scope thì có.
- `AutoPilot`: với fake generator/simulator, dừng đúng khi đủ K alpha distinct;
  alpha lỗi không làm dừng; `quit` dừng được.

## 9. Giai đoạn triển khai (theo MOTA_module_simulation §5)

- A. CachedSimulator + DedupChecker (tuần tự) — làm trước, dễ kiểm chứng.
- B. AutoPilot loop + điều kiện dừng + `quit`.
- C. CLI `auto` + mục wizard.
- (Song song hóa: KHÔNG thuộc bản này.)
