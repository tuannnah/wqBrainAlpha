# Mô tả: Module Mô phỏng (Simulation) trên WorldQuant Brain

> Tài liệu mô tả luồng và thiết kế của module simulation. Không chứa code — chỉ trình bày cách hoạt động, các quyết định thiết kế, và những điều cần lưu ý khi triển khai.

---

## 1. Vai trò của module

Module simulation là cầu nối giữa các alpha do tool sinh ra và nền tảng WorldQuant Brain. Nhiệm vụ của nó: nhận vào một alpha expression, gửi lên WQ để chạy backtest, chờ kết quả, rồi trả về các chỉ số đánh giá (Sharpe, Fitness, Turnover, Drawdown, Margin...) để các bước sau chấm điểm và lọc.

Đây là **bộ phận đắt nhất và chậm nhất** của toàn hệ thống. Mỗi lần mô phỏng tiêu tốn một phần quota của tài khoản WQ và mất từ vài giây tới vài phút. Vì vậy mọi thiết kế đều xoay quanh nguyên tắc: **tiết kiệm tối đa số lần mô phỏng thực sự gọi lên server**.

---

## 2. Luồng mô phỏng thực tế

Mô phỏng trên WQ Brain không phải là một request đồng bộ trả kết quả ngay. Nó là một quy trình bất đồng bộ gồm nhiều bước:

**Bước 1 — Gửi yêu cầu.** Tool gửi alpha expression kèm cấu hình (region, universe, delay, neutralization...) lên WQ. Server không trả kết quả ngay, mà trả về một đường dẫn theo dõi tiến độ (thông qua Location header).

**Bước 2 — Theo dõi tiến độ (poll).** Tool lặp lại việc hỏi server "xong chưa?" theo chu kỳ. Server báo trạng thái đang chạy và đề xuất khoảng thời gian chờ trước lần hỏi tiếp theo (qua Retry-After header). Vòng lặp này tiếp tục cho tới khi trạng thái chuyển sang hoàn tất.

**Bước 3 — Lấy kết quả.** Khi mô phỏng xong, server trả về một định danh alpha. Tool dùng định danh này để truy vấn chi tiết và lấy ra bộ chỉ số đánh giá.

**Bước 4 — Phân tích và lưu trữ.** Tool bóc tách các chỉ số quan trọng từ kết quả thô, tính điểm tổng hợp, rồi lưu cả kết quả thô lẫn các chỉ số đã xử lý vào cơ sở dữ liệu.

Điểm cần nhấn mạnh: phần lớn thời gian của một lần mô phỏng nằm ở **bước 2 (chờ)**. Tool không "treo" để chờ mà chủ động hỏi theo nhịp server đề xuất, tránh hỏi quá dày gây lãng phí và bị giới hạn.

---

## 3. Các tình huống cần xử lý

Một lần mô phỏng có thể không diễn ra suôn sẻ. Module phải xử lý gọn các trường hợp sau mà không làm sập cả pipeline:

**Phiên đăng nhập hết hạn giữa chừng.** Nếu trong lúc poll mà session hết hạn, module tự đăng nhập lại một lần rồi tiếp tục, không bắt đầu lại từ đầu.

**Mô phỏng quá lâu (timeout).** Nếu một alpha chạy vượt quá ngưỡng thời gian hợp lý (ví dụ vài phút), module đánh dấu nó là lỗi và bỏ qua, thay vì chờ vô hạn.

**Bị giới hạn tần suất (rate limit).** Khi server báo quá tải hoặc vượt quota tạm thời, module chờ theo cơ chế lùi dần (backoff) rồi thử lại, chứ không dồn dập gửi tiếp.

**Alpha bị lỗi cú pháp hoặc bị từ chối.** Một số alpha tuy qua được pre-filter nhưng vẫn bị WQ từ chối (ví dụ vi phạm quy tắc nội bộ của nền tảng). Module ghi lại lý do lỗi để phục vụ gỡ rối, rồi tiếp tục với alpha khác.

Nguyên tắc chung: **một alpha lỗi không bao giờ được phép làm dừng cả quá trình.** Mỗi alpha được xử lý độc lập, lỗi của nó được ghi nhận và bỏ qua.

---

## 4. Tiết kiệm quota — ba lớp phòng vệ

Vì mô phỏng là tài nguyên khan hiếm, tool dựng ba lớp để không lãng phí:

**Lớp 1 — Pre-filter trước khi gửi.** Mọi alpha đều phải qua kiểm tra cú pháp tại chỗ (kiểm tra cân bằng ngoặc, operator có tồn tại, field hợp lệ, độ phức tạp trong giới hạn) trước khi được phép gọi lên server. Lớp này chạy local, miễn phí, loại bỏ phần lớn alpha hỏng.

**Lớp 2 — Cache kết quả theo nội dung.** Mỗi alpha expression được băm (hash) thành một khóa. Trước khi mô phỏng, tool kiểm tra xem expression đó (với đúng cấu hình) đã từng được chạy chưa. Nếu rồi, lấy kết quả cũ từ DB, không gọi lại server. Điều này đặc biệt quan trọng với genetic algorithm, nơi cùng một expression có thể xuất hiện lại nhiều lần qua các thế hệ.

**Lớp 3 — Kiểm soát số lượng đồng thời.** Tool giới hạn số mô phỏng chạy song song và đặt khoảng nghỉ tối thiểu giữa các yêu cầu, để không vượt giới hạn của WQ và tránh bị chặn.

---

## 5. Chiến lược triển khai theo giai đoạn

Không nên xây phiên bản phức tạp ngay từ đầu. Thứ tự hợp lý:

**Giai đoạn A — Tuần tự, một alpha một lần.** Đây là phiên bản đầu tiên cần làm. Gửi một alpha, chờ xong, lấy kết quả, lưu DB. Mục tiêu của giai đoạn này là **xác nhận luồng đúng và bóc tách chỉ số chính xác** — đặc biệt là kiểm chứng tên và ý nghĩa của từng metric trả về từ WQ. Dễ gỡ lỗi, dễ đọc log. Chưa cần quan tâm tốc độ.

**Giai đoạn B — Thêm cache.** Sau khi luồng tuần tự chạy ổn, bổ sung lớp cache theo hash. Từ đây, chạy lại cùng một alpha sẽ không tốn quota.

**Giai đoạn C — Song song hóa có kiểm soát.** Cuối cùng mới thêm khả năng chạy nhiều mô phỏng cùng lúc, kèm rate limiter và backoff. Đây là phần dễ gây lỗi nhất (bị chặn, quá tải, race condition khi ghi DB) nên để sau cùng, khi mọi thứ khác đã chắc chắn.

Lý do của thứ tự này: nếu nhảy thẳng vào song song hóa khi chưa chắc luồng cơ bản đúng, sẽ rất khó phân biệt lỗi đến từ logic mô phỏng hay từ việc quản lý concurrency.

---

## 6. Cấu hình mô phỏng

Mỗi lần mô phỏng đi kèm một bộ cấu hình quyết định alpha được backtest như thế nào. Các tham số chính:

- **Region** — thị trường (USA, EUR, ASI...).
- **Universe** — tập cổ phiếu (TOP3000, TOP2000...).
- **Delay** — độ trễ dữ liệu (0 hoặc 1).
- **Neutralization** — cách trung hòa rủi ro (theo ngành, tiểu ngành, thị trường...).
- **Decay, Truncation** — các tham số xử lý tín hiệu.

Bộ cấu hình này phải **khớp với tổ hợp đã dùng khi lấy data field**. Một alpha mô phỏng ở USA/TOP3000/delay1 phải dùng đúng các field đã cache cho tổ hợp đó. Tool nên có một bộ cấu hình mặc định, cho phép ghi đè khi cần.

---

## 7. Dữ liệu lưu lại sau mỗi mô phỏng

Với mỗi alpha đã mô phỏng, tool lưu vào cơ sở dữ liệu:

- Liên kết tới alpha expression gốc.
- Cấu hình đã dùng (region, universe, delay...).
- Các chỉ số đánh giá: Sharpe, Fitness, Turnover, Drawdown, Margin, số vị thế long/short...
- Điểm tổng hợp do tool tính.
- Trạng thái: thành công, thất bại, hay lỗi.
- Toàn bộ kết quả thô trả về từ WQ (để phân tích lại sau này mà không cần mô phỏng lại).
- Thời điểm mô phỏng.

Việc lưu cả kết quả thô là có chủ đích: sau này nếu muốn tính thêm chỉ số mới hoặc kiểm tra lại, ta dùng dữ liệu đã có thay vì tốn quota chạy lại.

---

## 8. Phân biệt In-Sample và Out-of-Sample

Một nguyên tắc đánh giá cốt lõi: kết quả mô phỏng được chia thành hai phần thời gian. Phần **In-Sample (IS)** là giai đoạn tool dùng để tìm và tối ưu alpha. Phần **Out-of-Sample (OOS)** là giai đoạn giữ riêng để kiểm tra alpha có thực sự tốt hay chỉ đang khớp ngẫu nhiên với dữ liệu quá khứ.

Một alpha có Sharpe cao ở IS nhưng sụp ở OOS là dấu hiệu của overfitting. Tool phải theo dõi cả hai và ưu tiên alpha giữ được hiệu quả ở OOS. Đây là tiêu chí phân biệt alpha thật sự có giá trị với alpha chỉ "đẹp trên giấy".

---

## 9. Trước khi triển khai — kiểm chứng format

Trước khi viết logic bóc tách kết quả, cần thực hiện một mô phỏng thật với một alpha đơn giản (ví dụ một biểu thức cơ bản) và **ghi lại toàn bộ phản hồi thô từ WQ ở mỗi bước**: phản hồi khi gửi yêu cầu, khi poll tiến độ, và khi lấy chi tiết alpha.

Mục đích là xác nhận chính xác cấu trúc dữ liệu thật — tên các trường, vị trí của từng chỉ số trong kết quả, cách server báo trạng thái và thời gian chờ. Nền tảng có thể đã thay đổi so với mô tả chung, nên không nên giả định format mà phải kiểm chứng bằng dữ liệu thật trước.

---

## Tóm tắt

Module simulation nhận alpha, gửi lên WQ qua quy trình bất đồng bộ ba bước (gửi → chờ → lấy kết quả), xử lý gọn mọi lỗi mà không làm dừng pipeline, và bảo vệ quota bằng ba lớp: pre-filter, cache, và kiểm soát đồng thời. Nên xây theo thứ tự tuần tự → cache → song song. Luôn lưu cả kết quả thô và phân biệt IS/OOS để đánh giá alpha một cách trung thực.
