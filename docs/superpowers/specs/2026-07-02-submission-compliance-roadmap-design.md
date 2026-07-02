# Submission Compliance Roadmap — thiết kế + backlog

> Spec brainstorm 2026-07-02. Đọc toàn bộ `docs/worldquantbrain/docs/consultant-information/`
> (14 file) + đối chiếu với code hiện tại (`src/submission/`, `src/simulation/config.py`,
> `src/storage/models.py`). Mục tiêu cuối cùng không đổi: **sinh ra công thức (alpha) THỰC SỰ
> nộp được** trên WQ Brain, không chỉ pass gate local. Doc này có 2 phần:
> 1. Thiết kế chi tiết **sub-project C** (hạ tầng set properties/tags) — đã duyệt, sẵn sàng
>    viết plan triển khai ngay.
> 2. **Backlog/roadmap** các sub-project còn lại — mỗi mục đủ chi tiết để brainstorm riêng khi
>    tới lượt, không viết plan ngay.

## Vì sao cần doc này

Rà tài liệu `consultant-information` phát hiện code hiện tại có khoảng cách lớn với điều kiện
nộp thật của WQ Brain (xem bảng gap ở mỗi mục backlog). Việc này không thể làm trong 1 lần —
cần tách nhỏ, làm dần, mỗi phần tự đứng vững và có thể nối vào pipeline `miniquant` sau.

## Đính chính (2026-07-02, lúc triển khai sub-project B)

Khi bắt tay viết plan B, phát hiện `src/simulation/simulator.py:293-320` (đã có từ trước,
xác nhận qua `tests/test_simulator.py:42,67`) — WQ Brain tự trả `is.checks` (list tên test +
PASS/FAIL, vd `"LOW_SHARPE"`) ngay trong `GET /alphas/{id}` sau khi sim xong, và
`SimulationResult.status` đã tính từ đó (`failed` nếu bất kỳ check nào FAIL). Nghĩa là **ngưỡng
Sharpe/Fitness/Turnover/Weight/IS-Ladder thật — đúng theo tier tài khoản (consultant vs user),
đúng theo region/delay — WQ đã tự chấm sẵn**, không cần dò lại bằng bảng số cứng (rủi ro sai khi
WQ đổi ngưỡng theo quý, và không biết chắc tier tài khoản). Phạm vi sub-project B dưới đây đã
được sửa lại cho đúng phát hiện này — KHÔNG còn mục "IS Ladder Sharpe (xấp xỉ local)" hay
"bảng ngưỡng Sharpe/Fitness theo region/delay" như dự kiến ban đầu nữa.

## Gap audit — hiện trạng code (2026-07-02)

| Vùng | File | Hiện trạng | Thiếu gì so với tài liệu thật |
| --- | --- | --- | --- |
| Ngưỡng chọn ứng viên nộp | `src/submission/manager.py:31-33` | `MIN_SHARPE=1.5`, `MIN_FITNESS=1.2` hard-code, không phân biệt D0/D1 | Consultant thật: Sharpe D1≥1.58/D0≥2.69, Fitness D1≥1/D0≥1.5 ([consultant-submission-tests.md](../../worldquantbrain/docs/consultant-information/consultant-submission-tests.md)) |
| Turnover | — | Không có gate | Phải 1%–70% (Superalpha: 2%–40%) |
| Weight test | — | Không có gate | Max weight/1 stock <10% (consultant), <8% (user thường) |
| Self-correlation | `src/submission/correlation.py:9` | `MAX_SELF_CORR=0.70`, reject cứng nếu >0.7 | Đúng ngưỡng, nhưng thiếu nhánh "hoặc Sharpe cao hơn alpha trùng ≥10%" — code hiện tại không có đường thoát này |
| Prod-correlation | — | Không có | Consultant bị test riêng với TOÀN BỘ pool BRAIN, không chỉ pool của mình — chưa có endpoint/logic nào gọi |
| IS Ladder Sharpe | — | Không có | Test lặp 2→10 năm, ngưỡng PASS/FAIL theo bảng D0/D1 — chưa xấp xỉ local |
| Set properties/tags | — | Không có method nào, submit() không set gì | Cần cho Power Pool (mô tả Idea/Rationale ≥100 ký tự) và mọi mô tả alpha khác |
| Power Pool eligibility | — | Không có | Đếm operator/field unique, Power Pool Correlation, theme matching — chưa tồn tại |
| Single Dataset Alphas | — | Không có | Có sẵn `DataFieldModel.dataset_id` trong DB nhưng chưa dùng để phát hiện single-dataset alpha |
| Simulation settings | `src/simulation/config.py` (`SimConfig`) | Chỉ có region/universe/delay/neutralization/decay/truncation | Thiếu `pasteurization`, `nan_handling`, `test_period`, `max_trade`, `max_position` — một số **bắt buộc** theo region (ASI/JPN/HKG/TWN/KOR phải `maxTrade=ON`) |

---

## Phần 1 — Sub-project C: hạ tầng set properties/tags (THIẾT KẾ CHI TIẾT, sẵn sàng plan)

### Mục tiêu
Có một hàm Python nội bộ, đã test, gọi được `PATCH /alphas/{wq_alpha_id}` để set
`name/color/tags/regular_desc/combo_desc/selection_desc` — làm nền cho mọi sub-project sau cần
ghi mô tả/tag lên alpha (Power Pool Idea-Rationale, custom tag Osmosis, v.v.). **Chưa nối vào
luồng `submit()` hay CLI nào** — nội dung thật + thời điểm gọi do sub-project A/B quyết định
khi tới lượt, rồi gói lại thành 1 luồng e2e trong `miniquant`.

### Endpoint thật (tham chiếu từ package `wqb-mcp` đã cài, KHÔNG phải code trong repo này)
```
PATCH /alphas/{alpha_id}
{
  "name": "...",            # optional
  "color": "...",           # optional
  "tags": ["..."],          # optional
  "selectionDesc": "...",   # optional
  "comboDesc": "...",       # optional
  "regular": {"description": "..."}   # optional, LỒNG NGOẶC — không phải "regularDesc" phẳng
}
```
Field nào không cần set thì bỏ khỏi payload (không gửi `null`).

### Component 1 — `src/data/client.py`
Thêm:
```python
def patch(self, path: str, **kwargs) -> httpx.Response:
    return self._request("PATCH", path, **kwargs)
```
Đặt ngay cạnh `get()`/`post()` (dòng ~326-330). Tái dùng `_request()` sẵn có — thừa hưởng
retry 429 + tự re-authenticate 401. Không cần logic mới.

### Component 2 — `src/submission/manager.py`
Thêm dataclass + method vào `SubmissionManager`:
```python
@dataclass
class PropertiesResult:
    wq_alpha_id: str
    status: str  # ok/unchanged/error
    detail: str = ""

def set_properties(
    self, wq_alpha_id: str, *, name=None, tags=None,
    regular_desc=None, combo_desc=None, selection_desc=None, color=None,
) -> PropertiesResult: ...
```
Hành vi:
1. Dựng payload (bỏ field `None`/rỗng, lồng đúng `regular.description`).
2. **Idempotency**: so `tags` + `regular_desc` mới với giá trị đã lưu gần nhất trong DB cho
   `wq_alpha_id` (nếu có) — giống hệt thì **không gọi API**, trả `status="unchanged"`.
3. Gọi `client.patch(...)`. Bắt exception (không raise ra ngoài — giữ triết lý phòng thủ như
   `submit()`).
4. HTTP 2xx → `status="ok"`; khác → `status="error"`.
5. Luôn ghi DB (kể cả lỗi, để có audit trail) — xem Component 3.

### Component 3 — DB (`src/storage/models.py`, `SubmissionModel`)
Thêm 3 cột nullable vào `SubmissionModel`:
```python
tags = Column(Text)              # JSON-encoded list[str]
regular_desc = Column(Text)
properties_set_at = Column(DateTime)  # chỉ set khi status="ok"
```
Quy tắc ghi (trong `set_properties()`, KHÔNG dùng lại `_record()` của `submit()` vì khác
schema ý nghĩa `status`):
- Nếu đã có row `SubmissionModel` cho `wq_alpha_id` này (đã từng `submit()`) → **update** row
  mới nhất (`order_by(submitted_at.desc())`) — không tạo row mới, giữ 1-row-per-attempt cho
  submit nhưng cho phép "đính" properties vào lần submit gần nhất.
- Nếu **chưa có row nào** (set properties trước khi từng submit) → **insert** row mới với
  `status="properties_set"` — giá trị `status` MỚI, khác `submitted/rejected/error`, để không
  lọt vào `select_candidates()` (đang filter cứng `status == "submitted"` — không đổi logic
  đó).

### Testing (TDD bắt buộc — viết test trước)
File: `tests/unit/test_submission_manager.py` (mở rộng) + có thể tách
`tests/unit/test_alpha_client_patch.py` cho phần `client.patch()`.
Case cần có:
1. `client.patch()` gọi đúng method="PATCH", đúng path, đúng `json=`.
2. `set_properties()` dựng payload đúng shape — field `None` bị loại, `regular_desc` lồng
   đúng `{"regular": {"description": ...}}`.
3. Có row `SubmissionModel` cũ cho `wq_alpha_id` → **update** row đó (không insert thêm).
4. Không có row nào → **insert** row mới `status="properties_set"`.
5. Gọi 2 lần payload giống hệt → lần 2 không gọi `client.patch` (assert mock không được gọi
   thêm), trả `status="unchanged"`.
6. `client.patch` raise exception → `status="error"`, không crash, vẫn ghi được DB (không có
   `properties_set_at`).
7. HTTP non-2xx → `status="error"`.

### Ngoài phạm vi sub-project C (cố ý)
- Nội dung Idea/Rationale thật, cách sinh mô tả ≥100 ký tự — **sub-project A**.
- Cách/khi nào gắn tag `PowerPoolSelected`, cách lấy danh sách Power Pool Theme thật từ WQ
  Brain (chưa thấy trong tài liệu đã đọc, cần dò thêm — có thể không đi qua endpoint
  `PATCH /alphas/{id}` này mà là 1 bước riêng lúc submit "pure Power Pool") — **sub-project A**.
- Nối `set_properties()` vào `submit()`/pipeline thật — **sub-project A/B**, lúc đó mới đóng
  gói thành luồng e2e trong `miniquant`.

---

## Phần 2 — Backlog: các sub-project còn lại

Thứ tự đề xuất bên dưới dựa trên phụ thuộc: **B (đúng ngưỡng nền) nên làm trước A/D** vì Power
Pool eligibility và Single Dataset Alphas đều dựa trên gate Sharpe/Fitness/Turnover đúng chuẩn
consultant trước đã. C (ở trên) không phụ thuộc gì, làm được ngay.

### Sub-project B — Sửa đúng ngưỡng nộp consultant (ĐÃ SỬA LẠI PHẠM VI — xem "Đính chính" ở đầu doc)
**Hiện trạng đúng sau khi đọc kỹ code**: `SimulationResult.status` (từ `is.checks` do chính WQ
Brain trả về) ĐÃ phản ánh đúng Sharpe/Fitness/Turnover/Weight/IS-Ladder theo tier tài khoản
thật — `SubmissionManager.select_candidates()` lọc `status == "passed"` là đã đúng các gate
đó. Cái SAI thật sự là lớp filter THỪA/CỨNG phía trên (`MIN_SHARPE=1.5`, `MIN_FITNESS=1.2`) —
vì `status=="passed"` đã đảm bảo các ngưỡng thật (vd D1 sharpe>1.58 > 1.5), filter cứng này là
no-op vô hại cho D1 nhưng gây hiểu lầm về mặt thiết kế (đọc code tưởng đây là gate chính).

Việc cần làm (đã thu hẹp, không còn dựng bảng ngưỡng/IS-Ladder local nữa):
1. Xoá `MIN_SHARPE`/`MIN_FITNESS`/tham số override tương ứng khỏi `SubmissionManager` — dựa
   hẳn vào `status=="passed"` (đã đúng, tier-aware, không cần tự đoán số).
2. Lưu lại DANH SÁCH TÊN CHECK bị FAIL (không chỉ boolean status) — thêm
   `SimulationResult.failed_checks`, cột `SimulationModel.failed_checks` — để debug/audit biết
   chính xác vì sao 1 sim không "passed" (vd `LOW_SHARPE`, `CONCENTRATED_WEIGHT`...) thay vì chỉ
   biết "failed" chung chung.
3. Self-correlation: dự định thêm nhánh chấp nhận "Sharpe ≥10% so với alpha trùng nhất" nhưng
   **KHÔNG triển khai** — cần biết ID + Sharpe của alpha gây correlation cao, mà response
   `/correlations/self` hiện parse qua `CorrelationChecker._extract_max()` không rõ field nào
   là alpha-id (đoán sai sẽ match nhầm alpha) — để nguyên reject cứng >0.7 (an toàn), ghi nhận
   vào mục "cần verify" ở cuối doc.
4. Prod-correlation (consultant-only) — vẫn giữ nguyên, chưa làm (endpoint chưa xác nhận).

**KHÔNG còn cần**: bảng ngưỡng Sharpe/Fitness theo `(region, delay)`, gate Turnover riêng, gate
Weight riêng, IS Ladder Sharpe xấp xỉ local — tất cả đã có sẵn qua `status`/`is.checks` của WQ.

### Sub-project A — Power Pool Alphas
Phụ thuộc: sub-project B (cần Sharpe≥1.0 đã là ngưỡng thấp nhất trong mọi loại — không phụ
thuộc nhiều — nhưng nên có gate Turnover/self-corr chuẩn trước để không đề xuất alpha sai).
Cũng phụ thuộc sub-project C (hạ tầng set properties) đã xong.

Việc cần làm:
1. **Đếm operator unique** trong 1 biểu thức (loại trừ `ts_backfill`/`group_backfill`) ≤8, và
   **đếm data field unique** (loại trừ 6 grouping field: country/industry/subindustry/
   currency/market/sector/exchange — *lưu ý: danh sách 6 field ở power-pool-alphas.md khác nhẹ
   với danh sách 5 field ở single-dataset-alphas.md ("exchange" thừa/thiếu tuỳ chỗ) — cần đọc
   kỹ khi cài đặt, không copy nhầm danh sách*) ≤3. Cần 1 hàm AST-walk mới — gợi ý đặt cạnh
   `src/decorrelation/similarity.py` (đã có parser cho expression) hoặc file riêng
   `src/scoring/power_pool.py`.
2. **Power Pool Correlation** <0.5 — cần endpoint riêng (không phải self-correlation thường
   0.7) — WQ Brain có khái niệm "Power Pool Correlation" riêng theo tài liệu, endpoint thật
   chưa xác nhận, cần dò.
3. **Power Pool Theme matching** — đây là phần mở nhất: **chưa tìm thấy danh sách Theme thật ở
   đâu trong tài liệu đã đọc lẫn code `wqb-mcp`** (đã grep "theme" trong `wqb_mcp` — chỉ ra kết
   quả là "theme" filter của `/data-sets` API, không phải Power Pool Theme). Việc đầu tiên khi
   brainstorm sub-project A là **dò ra nguồn danh sách theme thật** (có thể qua
   `get_platform_setting_options`, hoặc 1 endpoint `/competitions`/`/themes` chưa biết, hoặc
   chỉ có trên UI — cần đăng nhập thật và xem network tab).
4. **Mô tả Idea/Rationale ≥100 ký tự** theo template (Idea / Rationale for data used /
   Rationale for operators used) — có thể tận dụng `AlphaModel.hypothesis` (đã có 4 phần: quan
   sát/nền tảng/lý giải KT/triển khai từ `HypothesisGenerator`) làm nguyên liệu cho LLM viết
   lại đúng khuôn Power Pool, rồi gọi `set_properties(regular_desc=...)` (sub-project C).
5. **Quota riêng**: 10 pure Power Pool/tháng + 1/ngày (loại trừ [Power Pool+ATOM/Regular]) —
   cần track riêng, không dùng chung `DAILY_QUOTA` hiện tại của `SubmissionManager`.
6. Gắn tag `PowerPoolSelected` khi nộp — qua `set_properties(tags=[...])`, nhưng **chưa rõ**
   tag này do WQ tự gắn sau khi đủ điều kiện hay do mình chủ động set trước khi submit — cần
   verify khi chạy thật.

### Sub-project D — Single Dataset Alphas
Nhỏ, độc lập với A, phụ thuộc B (dùng ngưỡng Last-2Y-Sharpe D0/D1 thay vì IS Ladder đầy đủ).
Việc cần làm:
1. Phát hiện alpha chỉ dùng field từ 1 `dataset_id` (trừ 5 grouping field:
   country/exchange/market/sector/industry/subindustry — **không có "currency" ở danh sách
   này, khác Power Pool** — xem lưu ý danh sách field ở mục A). `DataFieldModel.dataset_id` đã
   có sẵn trong DB (`src/storage/models.py`, thấy dùng ở `list_fields` CLI) — chỉ cần join.
2. Operator `inst_pnl()`/`convert()` tính như đang dùng dataset `pv1` — cần liệt kê 2 operator
   này đặc biệt khi xác định "single dataset".
3. Nếu single-dataset: dùng ngưỡng Last-2Y-Sharpe (D1≥2.38, D0≥3.96, nhân 0.85 nếu
   turnover<30%) thay vì chạy full IS Ladder — đơn giản hơn IS Ladder của sub-project B.

### Sub-project E — Simulation settings còn thiếu trong `SimConfig`
Độc lập, có thể làm song song bất kỳ lúc nào. Việc cần làm:
1. Thêm field vào `SimConfig`: `pasteurization` (default "ON"), `nan_handling` (default
   "OFF"), `test_period` (default 0 năm), `max_trade` (default "OFF"), `max_position`
   (default "OFF") — hiện `to_settings()`/constructor không có các field này.
2. **Ràng buộc bắt buộc theo region**: `max_trade="ON"` bắt buộc cho ASI/JPN/HKG/KOR/TW — nên
   validate trong `SimConfig.__post_init__` hoặc 1 factory `SimConfig.for_region()` để tránh
   quên set khi mở rộng sang region mới.
3. `max_position` khuyến nghị (không bắt buộc) cho USA/ASI/EUR.

### Sub-project F — Osmosis allocation (khuyến nghị: KHÔNG tự động hoá giai đoạn này)
Đây là phân bổ điểm **hàng tuần, cấp portfolio** (không phải cấp 1 alpha) — chốt vào 23:59 EST
Chủ nhật, ảnh hưởng toàn bộ pool đã nộp trước đó, đòi hỏi nhìn toàn cảnh performance +
correlation nội bộ pool để quyết định phân bổ. Rủi ro tự động hoá sai cao (tài liệu nhấn mạnh
"tránh đảo lộn đột ngột", cần đánh giá qua nhiều tuần) và giá trị with-automation không rõ so
với công sức. **Đề xuất: để việc này làm thủ công trên UI, không đưa vào roadmap kỹ thuật đợt
này** — ghi nhận lại để không quên, nhưng không brainstorm tiếp trừ khi có yêu cầu rõ.

### Sub-project G — BRAIN Genius tracking (nice-to-have, không phải gate cứng)
Không chặn nộp — là hệ thống xếp hạng theo quý (Gold/Expert/Master/Grand Master) dựa trên số
signal nộp, số "pyramid" (region×delay×dataset-category, cần ≥3 alpha/pyramid), và Combined
Alpha Performance. Có thể làm 1 dashboard/report đơn giản sau này (đếm pyramid đã hình thành,
avg distinct operators/fields mỗi alpha — thấp hơn thì tốt cho tiêu chí tie-break) nhưng
**không phải điều kiện nộp được hay không** — độ ưu tiên thấp nhất, chỉ làm nếu còn dư sức sau
A–E.

---

## Việc cần verify khi có phiên chạy thật (không thể xác nhận chỉ bằng đọc tài liệu)
- Endpoint/field thật cho Power Pool Correlation (khác self-correlation 0.7 thường).
- Endpoint/cách set Power Pool Theme khi submit "pure Power Pool Alpha".
- Endpoint Prod-Correlation cho consultant (có thể là biến thể của
  `/alphas/{id}/correlations/self` với tham số khác, hoặc endpoint riêng).
- Tag `PowerPoolSelected` do WQ tự gắn hay do client chủ động set trước khi submit.
- Response sim thật có sẵn dữ liệu Weight test không, hay phải gọi thêm endpoint riêng.

## Thứ tự triển khai đề xuất
C (đã sẵn sàng plan) → B (sửa ngưỡng nền, không phụ thuộc gì mới) → D (nhỏ, ăn theo B) → A
(Power Pool, ăn theo B+C, nhiều điểm cần dò API thật) → E (độc lập, chèn bất cứ lúc nào) → G
(optional, làm sau cùng) → F (không làm, chỉ ghi nhận).
