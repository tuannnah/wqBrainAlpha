# Thiết Kế Hệ Thống Nghiên Cứu Alpha Tự Động

## Mục Tiêu

Thay cơ chế dataset và chiến lược hard-code bằng một pipeline nghiên cứu tự động:

1. Đăng nhập WorldQuant BRAIN.
2. Tạo một snapshot metadata mới hoặc chọn snapshot cũ của đúng email.
3. Tự chọn dữ liệu, tạo ý tưởng và hypothesis nghiên cứu.
4. Dùng DeepSeek tạo Alpha có cấu trúc.
5. Kiểm tra Alpha cục bộ trước khi gửi simulation.
6. Đánh giá kết quả, tạo biến thể có mục tiêu khi Alpha đủ tiềm năng.
7. Lưu Alpha đạt chuẩn vào hàng chờ kiểm tra thủ công.
8. Tiếp tục cho đến khi người dùng nhập `quit` hoặc lượt chạy tạo được 10
   Alpha mới đạt chuẩn.

Hệ thống không tự submit Alpha.

## Phạm Vi

Thiết kế bao gồm:

- đồng bộ toàn bộ metadata mà tài khoản WorldQuant có quyền truy cập;
- nhiều snapshot metadata có tên dễ nhớ cho mỗi email;
- một Research DB riêng cho mỗi email;
- tích hợp DeepSeek qua biến môi trường `DEEPSEEK_API_KEY`;
- sinh Alpha, validation, simulation, đánh giá và tạo biến thể;
- console log, rotating file log và audit log trong SQLite;
- dừng an toàn bằng lệnh `quit`;
- cấu hình các giới hạn nghiên cứu ngoài mã nguồn.

Thiết kế chưa bao gồm:

- tự động submit Alpha;
- giao diện đồ họa;
- huấn luyện hoặc fine-tune model;
- gửi toàn bộ metadata DB tới DeepSeek;
- tạo biến thể nhiều thế hệ trong cùng một lượt nghiên cứu.

## Thuật Ngữ

- **Metadata snapshot:** bản chụp dataset, data field, operator, category và
  scope tài khoản có thể dùng tại một thời điểm.
- **Research DB:** lịch sử ý tưởng, hypothesis, prompt, Alpha, simulation,
  token và trạng thái kiểm tra thủ công.
- **Ý tưởng:** chủ đề nghiên cứu rộng, ví dụ đảo chiều giá hoặc thay đổi chất
  lượng cơ bản.
- **Hypothesis:** giả thuyết kinh tế cụ thể, có thể kiểm tra bằng một Alpha.
- **Alpha gốc:** Alpha đầu tiên được tạo từ một hypothesis.
- **Alpha cha:** Alpha gốc đủ quality gate để dùng làm đầu vào tạo biến thể.
- **Biến thể:** Alpha chỉ thay đổi theo một hướng cải thiện xác định.
- **Lượt chạy:** phiên tự động bắt đầu sau khi chọn snapshot và kết thúc khi
  nhận `quit` hoặc có 10 Alpha mới đạt chuẩn.

## Kiến Trúc

### WorldQuant Client

Client hiện tại tiếp tục quản lý đăng nhập, session và simulation, nhưng không
đọc `dataset_config.py`.

Client cung cấp các API có trách nhiệm riêng:

- đọc configuration để tìm scope hợp lệ của tài khoản;
- phân trang toàn bộ `/data-sets`;
- phân trang toàn bộ `/data-fields`;
- đọc `/data-categories`;
- đọc `/operators`;
- gửi simulation và đọc kết quả Alpha.

Các endpoint và query parameter được đóng gói trong client thay vì ghép URL ở
luồng nghiệp vụ.

### Metadata Synchronizer

Khi người dùng chọn tạo DB mới:

1. Hỏi tên snapshot; nếu bỏ trống dùng `Snapshot YYYY-MM-DD HH-mm`.
2. Tạo một SQLite tạm với trạng thái `SYNCING`.
3. Đọc tất cả scope hợp lệ từ configuration của tài khoản.
4. Đồng bộ dataset và data field theo từng tổ hợp instrument, region, delay và
   universe, có phân trang.
5. Đồng bộ category và operator.
6. Loại bản ghi trùng theo khóa nghiệp vụ nhưng giữ mọi scope khả dụng.
7. Ghi số lượng, thời gian, lỗi và checkpoint đồng bộ.
8. Chạy kiểm tra toàn vẹn.
9. Đổi trạng thái thành `READY` và chuyển file tạm thành snapshot chính thức.

Snapshot lỗi có trạng thái `FAILED`, được giữ để debug nhưng không xuất hiện
trong danh sách DB có thể dùng.

Khi chọn DB cũ, chương trình chỉ liệt kê snapshot `READY` thuộc email đang đăng
nhập, kèm tên, ngày tạo và số dataset/field/operator.

### Account Storage

Email được chuẩn hóa bằng cách trim và chuyển lowercase, sau đó băm SHA-256 để
tạo account ID. Tên thư mục không chứa email thô:

```text
data/
└── accounts/
    └── <account_hash>/
        ├── metadata/
        │   ├── <snapshot_id>.sqlite
        │   └── ...
        ├── research.sqlite
        └── logs/
            └── <run_id>.log
```

Mỗi email có nhiều Metadata DB nhưng chỉ một Research DB. Tên snapshot do
người dùng nhập là nhãn trong DB; tên file dùng UUID. Tên nhãn trùng được phép.

Email và mật khẩu không được lưu vào metadata snapshot. Research DB chỉ lưu
account hash và các ID nội bộ.

### Metadata Schema

Mỗi metadata snapshot tối thiểu có các bảng:

- `snapshot`: ID, nhãn, trạng thái, thời gian, phiên bản schema và thống kê.
- `scopes`: instrument type, region, delay, universe và quyền truy cập.
- `categories`: category/subcategory lấy từ BRAIN.
- `datasets`: ID, name, description, category, themes và metadata gốc cần thiết.
- `dataset_scopes`: quan hệ dataset với các scope có thể dùng.
- `data_fields`: ID, dataset ID, description, type, category và đơn vị.
- `data_field_scopes`: coverage, date coverage và scope của field.
- `operators`: name, scope, definition, description, level và metadata type.
- `sync_events`: endpoint, scope, offset, trạng thái, số bản ghi và lỗi.

SQLite FTS5 index mô tả dataset và data field để lọc candidate cục bộ mà không
gọi LLM.

### Research Schema

Research DB tối thiểu có các bảng:

- `research_runs`: snapshot, config, thời gian, trạng thái và số Alpha đạt chuẩn.
- `ideas`: nội dung, nguồn, trạng thái, novelty key và lý do kết thúc.
- `hypotheses`: idea ID, hypothesis, rationale, dataset và từ khóa field.
- `llm_requests`: loại request, model, prompt hash, prompt đã lọc, raw response
  đã lọc, input/output/cache token, thời gian và lỗi.
- `alphas`: expression, expression hash, structural fingerprint, settings,
  dataset, hypothesis, parent ID, generation và improvement direction.
- `simulations`: Alpha ID, WorldQuant Alpha ID, metrics, checks, HTTP status,
  thời gian và raw result đã lọc.
- `qualification_results`: pass/fail, từng tiêu chí và lý do.
- `review_queue`: trạng thái `PENDING_REVIEW`, `APPROVED` hoặc `REJECTED`.
- `research_lessons`: pattern thành công/thất bại được rút gọn để dùng ở các
  request sau.
- `run_events`: các sự kiện hiển thị trên console và file log.

Mọi bản ghi con dùng foreign key. Alpha biến thể bắt buộc có `parent_id`;
Alpha gốc không có parent.

## Chọn Dữ Liệu Và Tạo Ý Tưởng

Người dùng không phải chọn dataset, field hoặc nhập ý tưởng.

Pipeline chuẩn:

1. Idea selector ưu tiên ý tưởng chưa chạy được lưu trong Research DB. Nếu
   không còn ý tưởng phù hợp, pipeline mới yêu cầu DeepSeek tạo ý tưởng.
2. Local sampler chọn một tập nhỏ dataset chưa được khai thác nhiều, có quyền
   sử dụng và phù hợp với scope hiện tại.
3. Sampler tạo catalog rút gọn gồm category, dataset ID, mô tả, loại field và
   thống kê lịch sử. Không gửi toàn bộ DB.
4. Nếu cần ý tưởng mới, DeepSeek tạo một ý tưởng rộng và các từ khóa tìm field
   từ catalog rút gọn cùng các bài học lịch sử.
5. FTS5 và bộ lọc metadata tìm 20-50 field phù hợp trong các dataset được chọn.
6. Bộ lọc operator chọn các operator phù hợp với loại `MATRIX`, `VECTOR` hoặc
   `GROUP` và quyền tài khoản.
7. DeepSeek nhận ý tưởng, candidate fields, operator subset, scope và các bài
   học lịch sử để tạo tối đa 5 hypothesis khác nhau cùng 5 Alpha gốc.

Field `VECTOR` phải được giảm chiều bằng operator vector tương thích trước khi
dùng trong operator matrix. Field `GROUP` chỉ được dùng ở vị trí group. Alpha
không được tham chiếu field hoặc operator ngoài context đã cấp.

## DeepSeek Client

API key chỉ được đọc từ `DEEPSEEK_API_KEY`. Thiếu biến môi trường là lỗi cấu
hình và pipeline không được bắt đầu.

Client dùng API tương thích OpenAI tại `https://api.deepseek.com`. Model nằm
trong config, mặc định là `deepseek-v4-pro` tại ngày 09/06/2026 và có thể thay
đổi mà không sửa code. Không dùng alias sắp hết hạn làm mặc định.

Mọi request tạo dữ liệu máy đọc được phải:

- bật JSON Output;
- yêu cầu JSON trong system prompt;
- dùng schema ứng dụng để kiểm tra response;
- giới hạn số token output;
- retry có giới hạn nếu response rỗng, JSON hỏng hoặc bị rate limit;
- ghi usage token thực tế do API trả về;
- không ghi API key vào prompt, DB hoặc log.

Response tạo Alpha tối thiểu chứa:

```json
{
  "alphas": [
    {
      "hypothesis": "Giả thuyết kinh tế có thể kiểm tra",
      "rationale": "Lý do dùng dữ liệu và operator",
      "expression": "rank(ts_delta(field_id, 20))",
      "dataset_ids": ["dataset_id"],
      "field_ids": ["field_id"],
      "operator_names": ["rank", "ts_delta"],
      "settings": {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 0,
        "neutralization": "SUBINDUSTRY",
        "truncation": 0.08
      }
    }
  ]
}
```

Settings phải thuộc configuration và scope của metadata snapshot. Tool không
tin trực tiếp settings do model trả về; mọi giá trị đều được đối chiếu với DB.

## Validation Cục Bộ

Trước simulation, mỗi Alpha đi qua các bước:

1. Parse expression thành cấu trúc cú pháp.
2. Kiểm tra dấu ngoặc, số lượng tham số và identifier.
3. Kiểm tra mọi field và operator tồn tại trong snapshot.
4. Kiểm tra quyền dataset và scope.
5. Kiểm tra type compatibility của `MATRIX`, `VECTOR` và `GROUP`.
6. Kiểm tra settings hợp lệ theo configuration.
7. Chuẩn hóa expression và tạo exact hash.
8. Tạo structural fingerprint để phát hiện Alpha quá giống.

Alpha trùng exact hash không được simulation. Alpha có độ giống vượt ngưỡng
config với Alpha đã chạy cũng không được simulation. Lý do loại vẫn được lưu.

Validation cục bộ nhằm giảm request lãng phí, nhưng WorldQuant BRAIN vẫn là
nguồn quyết định cuối cùng về compile và quyền.

## Vòng Nghiên Cứu

### Alpha Gốc

Mỗi ý tưởng có tối đa 3 lô:

- mỗi lô tạo tối đa 5 Alpha gốc;
- 5 Alpha phải thuộc các hypothesis khác nhau;
- mỗi Alpha hợp lệ được simulation một lần;
- tối đa 15 simulation gốc cho một ý tưởng.

Sau mỗi lô, tool đánh giá:

- Sharpe;
- Fitness;
- Turnover;
- Margin;
- lỗi cú pháp;
- lỗi quyền dataset;
- sub-universe check nếu BRAIN trả về.

Nếu không có Alpha đủ quality gate sau một lô, tool tạo lô Alpha gốc mới cho
cùng ý tưởng. Sau 3 lô vẫn không có Alpha cha, ý tưởng được đánh dấu
`EXHAUSTED` và tool tự chuyển sang ý tưởng mới.

Nếu một lô đã có Alpha đủ quality gate, tool không tạo thêm lô Alpha gốc cho ý
tưởng đó. Tool chọn Alpha cha và chuyển ngay sang bước tạo biến thể.

Số ý tưởng không bị giới hạn theo lượt chạy. Điều kiện dừng toàn lượt là lệnh
`quit` hoặc 10 Alpha mới đạt chuẩn.

### Quality Gate Và Chọn Alpha Cha

Mỗi ý tưởng chọn tối đa 2 Alpha cha. Một Alpha đủ điều kiện nếu:

- đã đạt chuẩn; hoặc
- đạt ít nhất tỷ lệ cấu hình, mặc định 80%, của cả ngưỡng Sharpe và Fitness;
- không có lỗi cú pháp;
- không có lỗi quyền dataset;
- Turnover không vượt hard limit;
- không thất bại ở check nghiêm trọng được cấu hình.

Nếu không có Alpha qua gate, tool không gọi DeepSeek để cải thiện. Điều này
tránh tốn token cho Alpha không có tín hiệu.

### Tạo Biến Thể

Mỗi Alpha cha có tối đa 5 biến thể. Mỗi biến thể chỉ mang một
`improvement_direction`:

- `REDUCE_TURNOVER`;
- `IMPROVE_NEUTRALIZATION`;
- `ADJUST_TRADE_WHEN`;
- `CHANGE_TIME_WINDOW`;
- `HANDLE_OUTLIER_OR_SMOOTHING`.

DeepSeek nhận expression cha, hypothesis, metrics, checks, fields/operators hợp
lệ và một hướng cải thiện cụ thể. Tool không yêu cầu model "làm tốt hơn" một
cách chung chung.

Không tạo biến thể từ biến thể trong cùng lượt. Với tối đa 2 Alpha cha, một ý
tưởng có tối đa 10 simulation biến thể. Biến thể vẫn phải qua validation,
duplicate check và similarity check trước simulation.

### Alpha Đạt Chuẩn

Qualification ưu tiên checks do BRAIN trả về và bổ sung hard limit trong
config. Alpha đạt chuẩn:

- được ghi đầy đủ simulation và lý do pass;
- được thêm vào `review_queue` với `PENDING_REVIEW`;
- được tính vào mục tiêu 10 Alpha mới của lượt hiện tại;
- không tự submit;
- vẫn có thể được chọn làm Alpha cha để tìm biến thể tốt hơn, nếu lượt chưa
  đạt điều kiện dừng.

## Điều Khiển Lượt Chạy

Khi research loop bắt đầu, một luồng console riêng chờ lệnh. Lệnh hợp lệ:

```text
quit
```

Khi nhận `quit`:

1. đặt stop flag;
2. không bắt đầu request DeepSeek hoặc simulation mới;
3. hoàn tất request/simulation đang chạy;
4. ghi kết quả và checkpoint vào DB;
5. đóng session và file log;
6. kết thúc với trạng thái `STOPPED_BY_USER`.

`Ctrl+C` vẫn được bắt để tránh mất dữ liệu nhưng không phải cách dừng chính.

Lượt chạy tự kết thúc với trạng thái `TARGET_REACHED` khi tạo được 10 Alpha
mới đạt chuẩn trong chính lượt đó. Alpha chờ duyệt từ lượt trước không được
tính.

## Log Và Quan Sát

Console hiển thị tiến trình theo dạng:

```text
[RUN] Ý tưởng 2 | Lô 1/3 | Alpha 3/5
[DEEPSEEK] Đang tạo 5 Alpha gốc...
[VALIDATE] PASS | expression_hash=...
[SIMULATION] Sharpe=1.31 Fitness=1.05 Turnover=0.42 Margin=...
[QUALIFIED] WorldQuant Alpha abc123 -> PENDING_REVIEW (4/10)
[CONTROL] Đã nhận quit, đang hoàn tất simulation hiện tại...
```

Mỗi sự kiện đồng thời được:

- in ra console;
- ghi vào rotating log file theo run ID;
- ghi dạng có cấu trúc vào `run_events`.

Log không chứa mật khẩu, HTTP Authorization header hoặc API key. Raw DeepSeek
response và WorldQuant response chỉ được lưu sau khi lọc thông tin nhạy cảm.

## Cấu Hình

Một file config không chứa secret quản lý tối thiểu:

- DeepSeek base URL, model, timeout và retry;
- số Alpha mỗi lô, mặc định 5;
- số lô mỗi ý tưởng, mặc định 3;
- số Alpha cha, mặc định 2;
- số biến thể mỗi cha, mặc định 5;
- quality gate, mặc định 80%;
- turnover hard limit;
- similarity threshold;
- số candidate field tối thiểu/tối đa;
- simulation delay và rate-limit backoff;
- mục tiêu Alpha đạt chuẩn mỗi lượt, mặc định 10;
- giới hạn kích thước raw response và log rotation.

Config được kiểm tra khi khởi động. Giá trị âm, bằng 0 không hợp lệ hoặc vượt
giới hạn an toàn phải bị từ chối với thông báo rõ ràng.

## Xử Lý Lỗi

- Metadata API lỗi tạm thời: retry có backoff từ checkpoint.
- Snapshot không hoàn chỉnh: đánh dấu `FAILED`, không dùng nghiên cứu.
- DeepSeek thiếu API key: dừng trước khi tạo run.
- DeepSeek rate limit hoặc lỗi mạng: retry có giới hạn và ghi usage nếu có.
- JSON rỗng hoặc sai schema: yêu cầu sửa một lần; sau đó đánh dấu request lỗi.
- Alpha compile error: lưu lỗi, không retry cùng expression.
- Dataset authorization error: lưu lỗi, loại Alpha khỏi quality gate.
- WorldQuant rate limit: tôn trọng `Retry-After` nếu có.
- Simulation timeout: lưu trạng thái không hoàn chỉnh để kiểm tra, không tự
  coi là fail qualification.
- SQLite lỗi ghi: dừng tạo công việc mới để tránh chạy mà không có audit trail.

## Thay Đổi Cấu Trúc Mã Nguồn

Các trách nhiệm dự kiến được tách thành:

- `worldquant_client.py`: authentication, metadata API và simulation API.
- `metadata_store.py`: schema và truy vấn metadata snapshot.
- `metadata_sync.py`: full-account synchronization và checkpoint.
- `research_store.py`: Research DB và transaction.
- `deepseek_client.py`: request JSON, retry, token usage và redaction.
- `candidate_selector.py`: chọn dataset, field và operator cục bộ.
- `expression_validator.py`: parse, type check, duplicate và similarity.
- `research_engine.py`: state machine của idea, batch, parent và variant.
- `run_control.py`: nhận `quit` và stop flag.
- `logging_setup.py`: console, rotating file và DB event logging.
- `research_config.py`: load và validate config.
- `main.py`: menu đăng nhập, tạo/chọn DB và khởi chạy engine.

`dataset_config.py` và `AlphaStrategy` hard-code không còn nằm trên runtime path
của pipeline mới. Mã cũ chỉ được giữ tạm nếu cần migration hoặc test tương
thích, sau đó xóa khi không còn caller.

## Kiểm Thử

Mọi hành vi mới được triển khai test-first.

Test unit bao gồm:

- account hash và phân vùng DB theo email;
- tạo nhiều snapshot có nhãn trùng;
- snapshot tạm chỉ trở thành `READY` sau kiểm tra toàn vẹn;
- pagination và resume checkpoint;
- FTS candidate selection;
- DeepSeek JSON parsing, empty response, retry và token usage;
- không ghi secret vào log hoặc DB;
- validation field/operator/type/settings;
- exact duplicate và near-duplicate rejection;
- ba lô Alpha gốc rồi đổi ý tưởng;
- không tạo biến thể khi không có Alpha qua quality gate;
- chọn tối đa hai Alpha cha;
- mỗi biến thể có đúng một hướng cải thiện;
- không tạo biến thể thế hệ hai;
- `quit` không bắt đầu công việc mới và vẫn lưu công việc hiện tại;
- dừng đúng khi lượt hiện tại có 10 Alpha đạt chuẩn;
- Alpha đạt chuẩn vào `PENDING_REVIEW` và không tự submit.

Test integration dùng fake WorldQuant và fake DeepSeek server/session, không
gọi dịch vụ thật. Một smoke test thủ công riêng có thể dùng tài khoản thật sau
khi toàn bộ test tự động đạt.

## Đóng Gói Và Migration

- SQLite dùng thư viện chuẩn Python.
- Nếu dùng OpenAI-compatible SDK cho DeepSeek, dependency và PyInstaller config
  phải được cập nhật; nếu dùng `requests`, client vẫn phải có schema validation
  rõ ràng.
- Thư mục `data/` và log runtime không được đóng gói vào executable.
- DB runtime nằm ở thư mục dữ liệu người dùng có quyền ghi, không mặc định nằm
  cạnh executable nếu executable ở thư mục chỉ đọc.
- README mô tả `DEEPSEEK_API_KEY`, tạo/chọn snapshot, lệnh `quit`, vị trí log
  và quy trình kiểm tra thủ công.

## Nguồn Kỹ Thuật

- WorldQuant BRAIN frontend chính thức công bố các endpoint
  `/data-sets`, `/data-fields`, `/data-categories` và `/operators`:
  <https://platform.worldquantbrain.com/data>
- Danh sách operator chính thức:
  <https://platform.worldquantbrain.com/learn/operators>
- DeepSeek API tương thích OpenAI và model hiện hành:
  <https://api-docs.deepseek.com/>
- DeepSeek JSON Output:
  <https://api-docs.deepseek.com/guides/json_mode>
- DeepSeek models và pricing:
  <https://api-docs.deepseek.com/quick_start/pricing>
