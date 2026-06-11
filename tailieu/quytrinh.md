1. Lớp dữ liệu
Kết nối với WQ Brain API để lấy danh sách data fields, operators, và universe definitions. Cần lưu cache toàn bộ metadata (field names, categories, types) để engine sinh alpha có thể tham chiếu mà không cần gọi API liên tục.
Kỹ thuật: REST API client với session management, cookie/token authentication bắt chước behavior của trình duyệt khi đăng nhập Brain.

2. Lớp sinh Alpha
Đây là phần cốt lõi. Có 4 hướng tiếp cận:
Template-based: Xây dựng thư viện các pattern phổ biến như rank(ts_mean(field, d)), group_neutralize(...), sau đó điền tham số ngẫu nhiên hoặc theo grid search.
LLM-assisted: Dùng một model (có thể là Claude API) nhận vào mô tả ý tưởng alpha, danh sách operators/fields hợp lệ, và sinh ra expression. Cần few-shot prompting với các alpha mẫu hợp lệ.
Genetic Algorithm: Biểu diễn alpha expression dưới dạng cây (AST), rồi áp dụng crossover và mutation. Thư viện DEAP (Python) rất phù hợp.
Random search: Sinh ngẫu nhiên có ràng buộc — chọn operator, chọn field phù hợp với type, chọn tham số trong khoảng hợp lệ.

3. Tiền lọc & Mô phỏng
Tiền lọc: Parse expression thành AST, kiểm tra syntax trước khi nộp lên server để tiết kiệm API calls. Kiểm tra: độ sâu cây, operator nesting rules, type compatibility giữa field và operator.
Mô phỏng: WQ Brain dùng WebSocket để stream kết quả backtest. Cần:

Session management (login, giữ session)
Gửi simulation request đúng format JSON
Parse kết quả streaming về Sharpe, Fitness, Turnover, v.v.
Rate limiting để không bị block (thường ~1–3 sim/giây tùy account)


4. Chấm điểm & Lọc
Xây bộ lọc đa tiêu chí. Ngưỡng tham khảo ban đầu:
MetricNgưỡng tối thiểuSharpe≥ 1.3 (IS), ≥ 0.8 (OOS)Fitness> 1.0Turnover0.05 – 0.7 (tùy strategy)Drawdown< 20%Margin> 0Self-correlation< 0.7 với các alpha đã có

5. Vòng lặp tối ưu hóa
Sau khi có kết quả sim, feedback loop điều chỉnh:

Parameter tuning: Bayesian optimization (dùng optuna) cho các tham số số như window, decay
Operator mutation: Thay thế operator trong cây AST bằng operator tương tự
Population evolution: Giữ top-K alpha, dùng làm seed cho thế hệ tiếp theo


6. Storage
Dùng SQLite cho đơn giản ban đầu, schema gồm: bảng alphas (expression, params, metadata), bảng simulations (kết quả từng sim), bảng submissions (trạng thái nộp). Sau này nâng lên PostgreSQL nếu cần scale.

7. Submission Manager
Module cuối tự động nộp alpha đạt ngưỡng. Cần xử lý: rate limit của WQ Brain, check portfolio-level correlation (WQ giới hạn số alpha có correlation cao), và track trạng thái submission (pending, live, rejected).

Stack công nghệ gợi ý

Python là lựa chọn chính — httpx/requests cho API, websockets cho simulation streaming, DEAP cho GA, optuna cho Bayesian tuning, sqlite3/SQLAlchemy cho storage
UI: Streamlit để visualize nhanh, hoặc CLI thuần bằng rich/typer
Logging: loguru để track toàn bộ pipeline