# Hướng dẫn build: Tool sinh Alpha bằng AI trên WorldQuant Brain

> Tài liệu cho **Claude Code** thực thi. Chia thành các giai đoạn, mỗi giai đoạn gồm nhiều task nhỏ với mô tả và tiêu chí hoàn thành. Làm tuần tự, không nhảy giai đoạn. **Không cần viết code mẫu trong tài liệu này** — Claude Code tự quyết định triển khai, miễn đạt mô tả và acceptance của từng task.

---

## Bối cảnh & mục tiêu

Xây một tool tự động sinh, mô phỏng, đánh giá và nộp alpha trên nền tảng WorldQuant Brain, với hướng tiếp cận cốt lõi là **dùng AI (LLM) sinh alpha theo giả thuyết, tinh chỉnh theo phản hồi backtest, và ép tính độc đáo để chống alpha decay**.

Thành công = sinh ra được alpha chất lượng cao (vượt ngưỡng đa chỉ số), độc đáo (correlation thấp), và bền (không decay nhanh), sẵn sàng nộp lên WQ Brain.

---

## Nguyên tắc chung cho Claude Code

- Python 3.11+. Không hardcode credentials hay API key — đọc từ `.env`.
- Mỗi giai đoạn phải chạy được độc lập, demo được qua CLI khi kết thúc.
- Trước khi viết logic parse bất kỳ response nào của WQ Brain, **gọi API thật và log nguyên response trước**, xác nhận format rồi mới viết parser. Không đoán format.
- Tôn trọng rate limit và quota của WQ Brain. Luôn có delay, retry với backoff.
- Mọi bước tốn quota (simulation) phải được bảo vệ bằng pre-filter và cache.
- Log đầy đủ mọi API call, mọi simulation, mọi lần gọi LLM (kèm token/chi phí).
- Bật từng cơ chế một khi tuning. Không bật nhiều ràng buộc cùng lúc rồi chỉnh đồng thời.
- Commit theo từng task với message rõ ràng.

---

## Cơ sở phương pháp (để Claude Code hiểu "tại sao")

Thiết kế dựa trên hai công trình chính, có thể đối chiếu code:

- **AlphaAgent** (KDD 2025, github.com/RndmVariableQ/AlphaAgent): vòng lặp ba agent (giả thuyết → công thức → đánh giá), ba cơ chế chống decay (ép độc đáo bằng AST, căn chỉnh giả thuyết, kiểm soát độ phức tạp). Thiết kế đầy đủ của tool gần như tương ứng với framework này.
- **Navigating the Alpha Jungle** (arXiv 2505.11122): tinh chỉnh nhắm vào chiều yếu nhất, và cơ chế tránh nhánh con phổ biến để giữ đa dạng.

Ba phương pháp trụ cột, hoạt động ở ba giai đoạn khác nhau (không xung đột):
1. **Sinh theo giả thuyết** — quyết định tạo ra cái gì.
2. **Tinh chỉnh theo feedback** — quyết định cải thiện thế nào.
3. **Ép decorrelation** — quyết định giữ lại cái gì.

Quy tắc thiết kế xuyên suốt: các ràng buộc (độ độc đáo, khớp giả thuyết, độ phức tạp) phần lớn là **điểm phạt mềm cộng vào điểm tổng**, KHÔNG phải cửa chặn cứng. Chỉ vài thứ rõ ràng mới là cửa chặn cứng (cú pháp sai, trùng cấu trúc gần y hệt, độ phức tạp vượt trần). Mục đích: tránh siết quá tay khiến không alpha nào sống sót.

---

## Lộ trình phân tầng (tổng quan)

```
GĐ0  Setup
GĐ1  Đăng nhập + lấy data/operators + simulate được          ← nền tảng, làm trước
GĐ2  Vòng lặp AI cơ bản: giả thuyết + tinh chỉnh tham lam     ← Tầng 1
GĐ3  Ép decorrelation: AST-originality làm pre-filter          ← Tầng 2 (đáng giá nhất)
GĐ4  Hypothesis-alignment + complexity control                ← Tầng 3
GĐ5  Tinh chỉnh cấu hình (neutralization/decay/truncation)    ← giai đoạn 2 của search
GĐ6  (tùy chọn) Nâng cấp: MCTS / GA / multi-model
GĐ7  Submission manager + dashboard
```

Không bắt đầu GĐ2 khi GĐ1 chưa simulate được một alpha thật và lưu metrics. Mọi phần AI đều phụ thuộc vào simulation chạy ổn định.

---

# GIAI ĐOẠN 0 — Setup

- **T0.1** Khởi tạo project Python (venv), cấu trúc thư mục theo các module: data, simulation, generation, scoring, decorrelation, optimization, llm, submission, storage, dashboard.
- **T0.2** Tạo `requirements.txt` với các thư viện cần (HTTP client, websocket, ORM/SQLite, settings, logging, CLI, retry; phần AI thêm SDK gọi LLM; phần tối ưu thêm thư viện Bayesian/GA; dashboard).
- **T0.3** Tạo `.env.example` và settings đọc từ `.env` (credentials WQ, API key LLM, database URL, các default region/universe/delay, ngưỡng).
- **T0.4** Thiết lập `.gitignore` (`.env`, file session, file DB, cache). Thiết lập logging chung.

**Acceptance:** cài dependencies không lỗi; cấu trúc thư mục đầy đủ; settings nạp được từ `.env`.

---

# GIAI ĐOẠN 1 — Đăng nhập + lấy data + simulate

> Mục tiêu cuối: chạy CLI simulate một biểu thức đơn giản và nhận về metrics thật từ WQ Brain, lưu DB.

## Đăng nhập (session persistence)

- **T1.1** Viết WQ Brain client: đăng nhập qua Basic Auth, lưu session cookie ra file để tái dùng giữa các lần chạy.
- **T1.2** Kiểm tra session còn hạn bằng một request nhẹ; nếu còn thì bỏ qua đăng nhập lại.
- **T1.3** Tự re-authenticate khi gặp 401 giữa chừng, rồi retry request một lần.
- **T1.4** Xử lý trường hợp tài khoản cần biometric: log link xác thực cho người dùng mở thủ công lần đầu, dừng gọn (không crash).

**Acceptance:** đăng nhập thành công tạo file session; chạy lần hai bỏ qua đăng nhập; tự re-auth khi hết hạn.

## Lấy data fields & operators (tải một lần, cache trong DB)

- **T1.5** Tạo bảng theo dõi trạng thái fetch (`fetch_state`): mỗi tổ hợp (region, universe, delay) một dòng, ghi số lượng, thời điểm fetch, trạng thái.
- **T1.6** Viết repository data fields: lần đầu fetch toàn bộ qua API (phân trang), lưu DB; lần sau đọc từ DB, KHÔNG gọi API.
- **T1.7** Thêm cơ chế reload: chỉ fetch lại khi người dùng ép (`--reload`), hoặc cache quá hạn (TTL). Khi reload thì ghi đè, không nhân đôi.
- **T1.8** Làm tương tự cho operators (fetch một lần vào DB). Operators ít đổi nên TTL dài hơn.
- **T1.9** Lệnh "probe": gọi API thật in nguyên JSON một trang để xác nhận format trước khi tin parser.

**Acceptance:** fetch lần đầu lưu vài trăm field; lần hai load từ DB không gọi API (kiểm qua log); reload ghi đè không nhân đôi; mỗi tổ hợp (region, universe, delay) là cache riêng.

## Simulation (mốc quan trọng nhất GĐ1)

- **T1.10** Viết pre-filter cú pháp cơ bản: kiểm tra cân bằng ngoặc, operator/field tồn tại trong DB, độ sâu/độ phức tạp trong giới hạn. Chạy local, trước mọi simulation.
- **T1.11** Viết simulator: gửi yêu cầu simulation (biểu thức + cấu hình), nhận đường dẫn theo dõi, poll cho tới khi hoàn tất (tôn trọng Retry-After), lấy metrics.
- **T1.12** Parse metrics từ kết quả: Sharpe, Fitness, Turnover, Drawdown, Margin, số vị thế, phân biệt In-Sample và Out-of-Sample.
- **T1.13** Lưu kết quả vào DB: liên kết alpha, cấu hình đã dùng, các metrics, trạng thái, và **toàn bộ kết quả thô** (để phân tích lại không cần sim lại).
- **T1.14** Viết rate limiter + retry với backoff (429/500/503). Bắt đầu bằng chế độ tuần tự một alpha một lần (dễ debug).
- **T1.15** Thêm cache theo hash của (biểu thức + cấu hình): đã sim thì không sim lại, đọc từ DB.

**Acceptance:** simulate một biểu thức đơn giản trả metrics thật, lưu DB; sim lại cùng biểu thức lấy từ cache không gọi API; lỗi một alpha không làm dừng pipeline.

---

# GIAI ĐOẠN 2 — Vòng lặp AI cơ bản (Tầng 1: giả thuyết + tinh chỉnh tham lam)

> Mục tiêu: một vòng lặp AI hoàn chỉnh, chạy được. Chưa cần MCTS, chưa cần AST-originality.

## Lớp gọi LLM (trừu tượng hóa)

- **T2.1** Viết lớp client LLM trừu tượng cho phép gọi model qua API kiểu OpenAI (để sau dễ đổi/route nhiều model). Một hàm gọi chung nhận system + user prompt, trả về (ưu tiên) JSON.
- **T2.2** Thêm parse JSON an toàn (bọc try/except, retry khi model trả sai format). Log token và ước tính chi phí mỗi lần gọi.

## Sinh theo giả thuyết (hai bước)

- **T2.3** Thiết kế prompt sinh **giả thuyết thị trường có cấu trúc** gồm bốn phần: quan sát, kiến thức nền (lý thuyết/trực giác tài chính), lý giải kinh tế, và đặc tả triển khai (tham số gợi ý). Cho phép truyền vào một "hướng nghiên cứu" do người dùng chỉ định.
- **T2.4** Thiết kế prompt **dịch giả thuyết → mô tả bằng lời → biểu thức FASTEXPR**. Bắt buộc đi qua bước mô tả ngôn ngữ. Ngữ cảnh prompt chứa: danh sách operators hợp lệ (từ DB), một tập con fields liên quan theo category, và vài ví dụ alpha hợp lệ (few-shot).
- **T2.5** Vòng lặp kiểm tra cú pháp: biểu thức sinh ra → pre-filter → nếu lỗi, gửi lại kèm thông báo lỗi để model tự sửa (tối đa vài lần). Vẫn lỗi thì bỏ qua, log.
- **T2.6** Lưu mỗi alpha kèm bộ ba (giả thuyết, mô tả, biểu thức) và nguồn = "llm".

**Acceptance:** cho một hướng nghiên cứu, sinh được alpha hợp lệ kèm giả thuyết và mô tả; vòng tự sửa cú pháp hoạt động.

## Chấm điểm đa chiều

- **T2.7** Viết bộ trích & chuẩn hóa metrics từ kết quả simulation.
- **T2.8** Viết hàm điểm tổng hợp đa chiều (kết hợp Sharpe, Fitness, turnover, drawdown...). Đây là vector điểm, không chỉ một con số.
- **T2.9** Viết bộ lọc đa tiêu chí (ngưỡng tối thiểu cho từng metric). Phân biệt rõ **cửa chặn cứng** (loại thẳng) và **điểm phạt mềm** (chỉ trừ điểm).

## Tinh chỉnh tham lam theo feedback

- **T2.10** Lưu một "alpha zoo": kho các alpha đã vượt ngưỡng, dùng làm ví dụ chất lượng cho lần sinh sau.
- **T2.11** Cơ chế chọn **chiều yếu nhất** của alpha tốt hiện tại để cải thiện (ví dụ Sharpe ổn nhưng turnover quá cao → nhắm turnover). Chọn có trọng số ưu tiên chiều yếu.
- **T2.12** Prompt tinh chỉnh: đưa alpha + metrics + chỉ rõ chiều cần cải thiện, yêu cầu model đề xuất cải tiến (bằng lời trước, rồi công thức). Simulate kết quả, cập nhật.
- **T2.13** Lưu cả các ca **thất bại** kèm lý do (lệch giả thuyết, lỗi cú pháp, điểm thấp) để lần sau tránh lặp lại.
- **T2.14** Vòng lặp tham lam: luôn cải thiện alpha tốt nhất hiện có, lặp tới khi đạt số vòng hoặc hết cải thiện.

**Acceptance:** vòng lặp chạy trọn vẹn nhiều vòng, điểm tốt nhất cải thiện theo thời gian (log chứng minh); alpha zoo tích lũy; tận dụng cache để không sim trùng.

---

# GIAI ĐOẠN 3 — Ép decorrelation (Tầng 2: AST-originality)

> Bước đáng giá nhất: chống decay, giảm correlation, và chạy local nên tiết kiệm quota. Bật riêng cơ chế này và quan sát tác động trước khi thêm cái khác.

- **T3.1** Viết bộ parse biểu thức thành cây cú pháp (AST): node trong là operator, lá là field/số.
- **T3.2** Viết thuật toán đo độ tương đồng giữa hai AST = kích thước (số node) của **nhánh con chung lớn nhất** (subtree đẳng cấu lớn nhất).
- **T3.3** Xây "alpha zoo tham chiếu": gồm (a) thư viện alpha công khai như Alpha101 đã dịch sang FASTEXPR, và (b) các alpha bạn đã nộp. Parse sẵn thành AST.
- **T3.4** Tính **điểm độc đáo** của alpha mới = mức tương đồng cao nhất so với toàn bộ zoo. Càng giống alpha đã biết, điểm độc đáo càng thấp.
- **T3.5** Dùng điểm độc đáo làm **pre-filter rẻ TRƯỚC khi simulate**: loại thẳng các alpha trùng cấu trúc gần y hệt (cửa chặn cứng), trừ điểm mềm cho phần còn lại. Mục đích: không tốn một lần sim nào cho alpha gần-trùng-lặp.
- **T3.6** Bổ sung biến thể "tránh nhánh con phổ biến": thống kê các subtree xuất hiện nhiều trong các alpha tốt, đưa vào prompt yêu cầu LLM tránh dùng lại, để giữ đa dạng.
- **T3.7** Ghi rõ trong tài liệu nội bộ: AST-similarity KHÁC return-correlation của WQ. AST là bộ lọc rẻ loại trùng hiển nhiên; correlation thật của WQ vẫn là kiểm tra cuối khi nộp.

**Acceptance:** tính được điểm độc đáo cho alpha bất kỳ so với zoo; alpha trùng cấu trúc cao bị loại trước khi sim; quan sát thấy alpha sinh ra đa dạng hơn.

---

# GIAI ĐOẠN 4 — Hypothesis-alignment + complexity control (Tầng 3)

- **T4.1** Viết bộ chấm **nhất quán giả thuyết–công thức** bằng một lần gọi LLM phụ: (a) mô tả có triển khai đúng giả thuyết không, (b) biểu thức có phản ánh đúng mô tả không. Ví dụ: tuyên bố về thanh khoản nhưng không có thành phần volume/spread → điểm thấp.
- **T4.2** Dùng điểm nhất quán làm bộ lọc thứ hai TRƯỚC khi sim (loại cái "nói một đằng làm một nẻo"). Cũng không tốn sim.
- **T4.3** Thêm phạt **độ phức tạp**: theo độ sâu cây, số tham số tự do, số lượng feature sử dụng. Cây càng rối càng dễ overfit/decay.
- **T4.4** Gộp ba khoản phạt (độ độc đáo, mức khớp giả thuyết, độ phức tạp) thành một số hạng điều chuẩn cộng vào điểm tổng theo công thức "điểm hiệu quả − λ·phạt". Để các trọng số cấu hình được, mặc định ở mức vừa.

**Acceptance:** alpha lệch giả thuyết bị lọc trước sim; alpha quá phức tạp bị trừ điểm; các trọng số chỉnh được và có tác động quan sát được.

---

# GIAI ĐOẠN 5 — Tinh chỉnh cấu hình (search giai đoạn hai)

> Một alpha = (biểu thức + cấu hình). Cùng biểu thức, đổi neutralization/decay/truncation cho metrics khác hẳn. Đây là không gian tìm kiếm thứ hai, nhỏ, xử lý SAU khi đã có biểu thức tốt.

- **T5.1** Tách rõ trong code hai không gian: **không gian biểu thức** (nơi LLM/GA hoạt động, dùng cấu hình mặc định cố định) và **không gian cấu hình** (neutralization, decay, truncation, delay...).
- **T5.2** Ở giai đoạn sinh/tinh chỉnh biểu thức, **cố định một cấu hình mặc định hợp lý** (ví dụ neutralization theo subindustry, decay vừa, truncation chuẩn). Không quét cấu hình ở giai đoạn này để tránh bùng nổ số lần sim.
- **T5.3** Với riêng các alpha hứa hẹn (đã vượt ngưỡng), chạy **quét cấu hình** bằng tìm kiếm Bayesian hoặc grid trên decay, truncation, và thử vài mức neutralization.
- **T5.4** Ghi nhận vai trò kép của **neutralization** như một công cụ decorrelation (mức chi tiết hơn → tương đối hơn → correlation thấp hơn), không chỉ là tinh chỉnh.
- **T5.5** Ghi nhận **decay** là núm điều khiển turnover chính (ảnh hưởng việc qua ngưỡng turnover và Fitness sau phí), và **truncation** ảnh hưởng drawdown/margin.
- **T5.6** **Bắt buộc kiểm chứng OOS** cho mọi lần quét cấu hình: chỉ giữ cấu hình nào tốt cả In-Sample lẫn Out-of-Sample. Tinh chỉnh cấu hình chỉ theo IS là một dạng overfitting.

**Acceptance:** giai đoạn sinh biểu thức dùng cấu hình cố định; quét cấu hình chỉ chạy trên alpha tốt; kết quả quét được lọc qua OOS.

---

# GIAI ĐOẠN 6 — Nâng cấp (tùy chọn)

> Chỉ làm khi các giai đoạn trên đã ổn định.

- **T6.1** Nâng vòng feedback tham lam lên **MCTS**: cho phép giữ nhiều nhánh ứng viên, cân bằng khám phá/khai thác, lan ngược điểm qua cây.
- **T6.2** Bổ sung **GA** (cây biểu thức, crossover/mutation) như một bộ sinh song song với LLM, dùng LLM làm hạt giống chất lượng cho quần thể.
- **T6.3** Thêm **multi-model routing**: dùng model rẻ nhanh cho sinh/đột biến hàng loạt, model mạnh cho suy luận khó (sinh giả thuyết, đánh giá độc đáo). Lớp trừu tượng ở T2.1 đã chuẩn bị cho việc này.
- **T6.4** Hỗ trợ đa region/universe: mỗi tổ hợp dùng cache field riêng (đã chuẩn bị ở GĐ1).

**Acceptance:** mỗi nâng cấp bật được/tắt được độc lập, không phá vỡ pipeline cơ bản.

---

# GIAI ĐOẠN 7 — Submission + Dashboard

- **T7.1** Bộ chọn alpha để nộp: lọc theo ngưỡng, sắp theo điểm, loại các alpha tương quan cao với nhau (greedy theo điểm độc đáo + correlation).
- **T7.2** Tích hợp **kiểm tra correlation thật của WQ Brain** trước khi nộp (đây là kiểm tra cuối, khác AST-similarity local).
- **T7.3** Submission manager: nộp alpha đạt chuẩn, quản lý quota nộp theo ngày, retry với backoff, lưu trạng thái (pending/live/rejected) và lý do.
- **T7.4** Dashboard: tổng quan (số alpha, pass rate, phân phối Sharpe), bảng khám phá alpha (filter/sort, xem giả thuyết + mô tả + metrics), tiến trình cải thiện theo vòng, ma trận correlation top alpha, theo dõi submission.

**Acceptance:** chọn-và-nộp chạy ở chế độ thử (dry-run) liệt kê đúng alpha; submit thật ghi trạng thái; dashboard hiển thị data thật từ DB.

---

## Các nguyên tắc cần nhớ xuyên suốt

- **Ba phương pháp ở ba giai đoạn khác nhau, không xung đột** — nhưng đừng biến tất cả thành cửa chặn cứng. Dùng điểm phạt mềm, chỉ vài thứ rõ ràng mới chặn cứng.
- **Bật từng cơ chế một khi tuning.** Bật một cái, chạy vài chục alpha, quan sát phân phối Sharpe/correlation, rồi mới thêm cái kế. Không chỉnh nhiều trọng số đồng thời.
- **AST-originality và các bộ lọc LLM chạy local, miễn phí — đặt TRƯỚC simulation** để tiết kiệm quota. Simulation là tài nguyên đắt nhất.
- **Cache mọi simulation theo hash (biểu thức + cấu hình).** Đặc biệt quan trọng khi có GA/lặp nhiều vòng.
- **Hai không gian tìm kiếm tách biệt:** biểu thức trước (cấu hình cố định), cấu hình tinh chỉnh sau (chỉ trên alpha tốt).
- **OOS là trọng tài cuối** cho cả biểu thức lẫn cấu hình. Mọi thứ chỉ đẹp ở IS đều đáng nghi.
- **Thành thật về nguồn:** vòng lặp đầy đủ bám AlphaAgent; tinh chỉnh nhắm chiều yếu bám Alpha Jungle; phần greedy + phân tầng + AST-as-prefilter + quota-aware là điều chỉnh riêng cho WQ Brain, cần tự kiểm chứng.

---

## Thứ tự thực thi tuyệt đối

```
GĐ0 → GĐ1 (phải simulate được mới đi tiếp)
    → GĐ2 (vòng lặp AI cơ bản chạy được)
    → GĐ3 (bật decorrelation, quan sát)
    → GĐ4 (alignment + complexity)
    → GĐ5 (tinh chỉnh cấu hình, kiểm OOS)
    → GĐ6 (nâng cấp tùy chọn)
    → GĐ7 (nộp + dashboard)
```

Kết thúc mỗi giai đoạn phải pass toàn bộ acceptance trước khi sang giai đoạn sau.
