# Thiết kế: Engine sinh alpha hiệp đồng (AlphaGen-adapted)

Ngày: 2026-06-20
Trạng thái: đã duyệt hướng A

## Bối cảnh & vấn đề

Log `logs/wq_alpha_2026-06-19.log` cho thấy chất lượng alpha sim rất thấp:

1. **Ngân sách sim bị đốt vào rác**: `rank(close)` ×360, `rank(rank(open))`,
   `rank(foo_bar)`, `rank(a,b,c)`, `ts_mean(volume,volume)`...
2. **Vòng GA/hybrid thoái hóa**: bơm lặp đúng 1 biến thể `ts_mean(volume,5)`
   hàng giờ cho mọi "chiều yếu".
3. **LLM bịa cả field lẫn metric**: các "hướng" kèm `sharpe=2.1, fitness=0.92`
   là số LLM tự bịa trong text, không đo thật; gán sai ngữ nghĩa field; field
   chết (`mdl77_2gdna_cfroi`, `composite_sentiment_score_2`) vẫn lọt cổng.
4. **Bug scorer cốt lõi**: sim lỗi → `normalize` trả default
   `sharpe=0, fitness=0, drawdown=1.0` → `score = 0.12`. Alpha lỗi được chấm
   0.12 điểm ngang một alpha thật tầm thường → GA mất gradient.

Gốc rễ: **không có vòng phản hồi dựa trên metric đo thật** và **không có áp lực
độc đáo bên trong vòng tiến hóa**.

## Mục tiêu

Alpha vừa **độc đáo** (correlation thấp với pool/zoo) vừa **qua chuẩn nộp WQ**
(sharpe/fitness/turnover đạt ngưỡng).

## Xương sống lý thuyết

**AlphaGen (Yu et al., KDD 2023)** — tối ưu cả một *tập* alpha hiệp đồng: phần
thưởng của ứng viên mới = mức tăng IC của tín hiệu tổng hợp khi thêm nó vào
pool. Công thức này thưởng alpha vừa dự báo tốt vừa ít tương quan với pool.

**Bản phỏng theo (adaptation):** reward của paper cần IC tính trên return cục bộ
tốc độ cao; ở đây hàm đánh giá bị chặn bởi WQ-sim (chậm, tốn quota). Nên:
- **Chất lượng standalone** lấy từ WQ-sim thật (`score_vector(result).total`).
- **Đóng góp biên/độc đáo (proxy IC hiệp đồng)** đo bằng **AST-similarity cục
  bộ** (`decorrelation.ReferenceZoo`, field-aware) — miễn phí, chạy local, áp
  ngay trong vòng GA. Tùy chọn dùng correlation PnL thật (`get_alpha_pnl`) ở
  cổng nộp cuối, không dùng trong vòng để tránh quota.

## Kiến trúc — 4 thành phần (mỗi phần 1 commit, TDD)

### Thành phần 1 — Hàm mục tiêu pool-aware `src/scoring/synergy.py` (làm trước)

`SynergyScorer` là callable drop-in cho `scorer` callback của GA
(`value = self.scorer(result)`).

```
def __call__(result) -> float:
    if getattr(result, "status", None) == "error":
        return NEG_INF                      # FIX bug 0.12: loại hẳn sim lỗi
    base = score_vector(result).total        # chất lượng standalone [0,1]
    orig = self.zoo.originality(result.expression)  # [0,1], 1=độc đáo
    reward = base * (orig ** self.beta)      # beta điều chỉnh sức ép decorrelation
    if status == "passed":
        self.zoo.add(result.expression)      # pool online: ứng viên sau bị phạt
    return reward                            #   nếu giống alpha tốt đã tìm thấy
```

- `beta` (mặc định 1.0): >1 ép độc đáo mạnh hơn.
- Pool online: chỉ add alpha `passed` → xây dần "tập alpha tốt đã tìm" để GA bị
  đẩy ra vùng decorrelated (cơ chế hiệp đồng của paper).
- Interface: phụ thuộc duy nhất vào `ReferenceZoo` (có `.originality`, `.add`) và
  `score_vector`. Test độc lập bằng `SimulationResult` giả + zoo giả.

### Thành phần 2 — Grounding sinh ý tưởng (`src/llm/generator`, prompt)

- Nhồi metadata field thật (id, mô tả, dataset, type) từ DB vào prompt; LLM chỉ
  chọn field trong catalog.
- Vứt mọi `sharpe=/fitness=` LLM tự bịa khi parse output — chỉ WQ-sim định nghĩa
  chất lượng.
- `VERIFIED_FIELDS` (hardcode 30 field) → provider tra cứu DB động.

### Thành phần 3 — Chống thoái hóa & kỷ luật ngân sách (`src/optimization`)

- **Novelty archive toàn cục**: không sim lại / không inject lại expression đã
  thấy (chặn `ts_mean(volume,5)` lặp và `rank(close)` ×360).
- Seed pool **không bao giờ** sụp về `rank(close)`: bắt buộc seed từ
  `NOVEL_ALPHAS` + template families khi LLM tắt.
- Prefilter chặn cây tầm thường/sai arity trước khi tốn lượt sim.

### Thành phần 4 — Tinh chỉnh có trí nhớ (`src/llm/refiner`)

- Refiner nhận lịch sử biến thể đã thử của cùng nhánh → cấm lặp; nhắm chiều yếu
  đo thật từ sim, không từ số bịa.

## Thứ tự thực thi

1 (objective, đòn bẩy cao nhất + độc lập) → 3 (chống thoái hóa) → 2 (grounding)
→ 4 (refiner). Mỗi thành phần: viết test trước (TDD), 1 commit, giao tiếp tiếng
Việt.

## Rủi ro & quyết định

- **Không dùng `get_alpha_pnl` trong vòng** (tránh quota) — độc đáo trong vòng
  dùng AST-similarity cục bộ; PnL thật để dành cổng nộp cuối (ngoài phạm vi đợt
  này).
- Pool online mutate khi evaluate → phụ thuộc thứ tự; chấp nhận (đúng hành vi
  online của AlphaGen), test cố định seed để xác định.
