# Hướng dẫn build: WorldQuant Brain Auto-Alpha Tool

> Tài liệu này dành cho **Claude Code** thực thi. Mỗi phase là một mốc hoàn chỉnh, chạy được, có acceptance criteria rõ ràng. **Làm tuần tự, không nhảy phase.** Hoàn thành và test xong một phase rồi mới sang phase tiếp theo.

---

## Quy tắc chung cho Claude Code

1. **Ngôn ngữ:** Python 3.11+.
2. **Mỗi phase phải chạy được độc lập** — kết thúc phase là có thể demo bằng CLI.
3. **Viết test cho mỗi module** trước khi sang phase sau (`pytest`).
4. **Không hardcode credentials** — đọc từ `.env` qua `pydantic-settings`.
5. **Log đầy đủ** bằng `loguru` — mọi API call, mọi simulation đều log.
6. **Tôn trọng rate limit của WQ Brain** — không spam request, luôn có delay và retry.
7. **Commit theo từng task** với message rõ ràng (`feat:`, `fix:`, `test:`).
8. Khi không chắc về endpoint/format của WQ Brain API, **kiểm tra bằng request thật trước, log response, rồi mới viết logic parse**. Không đoán format response.

---

## Setup ban đầu (làm trước Phase 1)

### Task 0.1 — Khởi tạo project

```bash
mkdir wq-alpha-tool && cd wq-alpha-tool
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### Task 0.2 — `requirements.txt`

```
httpx>=0.27
websockets>=12.0
pydantic>=2.7
pydantic-settings>=2.3
sqlalchemy>=2.0
loguru>=0.7
typer>=0.12
rich>=13.7
python-dotenv>=1.0
tenacity>=8.3
pytest>=8.2
pytest-asyncio>=0.23
# Phase 2+
deap>=1.4
optuna>=3.6
# Phase 3 (DeepSeek dùng chuẩn OpenAI-compatible)
openai>=1.30
# Dashboard
streamlit>=1.35
pandas>=2.2
plotly>=5.22
```

### Task 0.3 — Cấu trúc thư mục

```
wq-alpha-tool/
├── .env                       # KHÔNG commit (thêm vào .gitignore)
├── .env.example               # Template, commit cái này
├── config/
│   ├── settings.py            # Pydantic settings
│   └── sim_defaults.yaml      # Default simulation config
├── src/
│   ├── data/                  # Phase 1
│   ├── simulation/            # Phase 1
│   ├── storage/               # Phase 1
│   ├── generation/            # Phase 2+
│   ├── scoring/               # Phase 2
│   ├── optimization/          # Phase 2 (GA)
│   ├── llm/                   # Phase 3 (DeepSeek)
│   └── submission/            # Phase 4
├── tests/
├── dashboard/                 # Phase 4
├── main.py                    # CLI entry
└── README.md
```

### Task 0.4 — `.env.example`

```env
# WorldQuant Brain
WQ_EMAIL=your_email@example.com
WQ_PASSWORD=your_password

# DeepSeek (Phase 3)
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Database
DATABASE_URL=sqlite:///wq_alpha.db

# Defaults
DEFAULT_REGION=USA
DEFAULT_UNIVERSE=TOP3000
DEFAULT_DELAY=1
```

**Acceptance:** `pip install -r requirements.txt` chạy không lỗi; cấu trúc thư mục đầy đủ; `.gitignore` chứa `.env`, `venv/`, `*.db`, `__pycache__/`.

---

# PHASE 1 — Đăng nhập, lấy data/fields, mô phỏng được

> Mục tiêu cuối Phase 1: chạy `python main.py simulate --expr "rank(close)"` và nhận về Sharpe/Fitness/Turnover thật từ WQ Brain.

## Task 1.1 — Settings (`config/settings.py`)

Dùng `pydantic-settings` đọc từ `.env`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    wq_email: str
    wq_password: str
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    database_url: str = "sqlite:///wq_alpha.db"
    default_region: str = "USA"
    default_universe: str = "TOP3000"
    default_delay: int = 1

    class Config:
        env_file = ".env"

settings = Settings()
```

## Task 1.2 — WQ Brain Client (`src/data/client.py`)

**Đây là module quan trọng nhất của Phase 1.** Xử lý authentication và session.

WorldQuant Brain dùng cơ chế:
- Endpoint auth: `POST https://api.worldquantbrain.com/authentication`
- Auth bằng **HTTP Basic Auth** (email + password) trong lần POST đầu.
- Response trả về session cookie (`JSESSIONID` hoặc tương tự) — lưu trong cookie jar của `httpx.Client`.
- Một số tài khoản có **biometric / persona check**: nếu response 401 kèm header `WWW-Authenticate` chứa link, cần mở link đó để xác thực (xử lý trường hợp này — log link ra cho user click thủ công lần đầu).

```python
import httpx
from loguru import logger

class WQBrainClient:
    BASE_URL = "https://api.worldquantbrain.com"

    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.client = httpx.Client(base_url=self.BASE_URL, timeout=30.0)
        self._authenticated = False

    def authenticate(self) -> None:
        """
        POST /authentication với Basic Auth.
        Lưu session cookie vào self.client.
        Xử lý case cần biometric: log URL cho user.
        """
        resp = self.client.post(
            "/authentication",
            auth=(self.email, self.password),
        )
        if resp.status_code == 201:
            self._authenticated = True
            logger.success("Đăng nhập WQ Brain thành công")
        elif resp.status_code == 401:
            # Có thể cần biometric — kiểm tra WWW-Authenticate header
            logger.error(f"Auth 401: {resp.headers.get('WWW-Authenticate')}")
            raise AuthError(...)
        else:
            raise AuthError(f"Auth thất bại: {resp.status_code} {resp.text}")

    def get(self, path: str, **kwargs) -> httpx.Response:
        """GET có tự re-auth nếu session hết hạn (401 → authenticate → retry 1 lần)"""

    def post(self, path: str, **kwargs) -> httpx.Response:
        """POST tương tự GET"""
```

**Lưu ý:** Trước khi viết logic parse, **gọi `/authentication` thật, log toàn bộ response (status, headers, body)** để xác nhận format. WQ có thể thay đổi cơ chế.

## Task 1.3 — Lấy Data Fields (`src/data/fields.py`)

Endpoint (kiểm tra lại bằng request thật):
```
GET /data-fields?region={region}&delay={delay}&universe={universe}&limit={n}&offset={offset}
```

```python
from dataclasses import dataclass

@dataclass
class DataField:
    id: str
    description: str
    type: str         # MATRIX / VECTOR / GROUP
    dataset_id: str
    region: str
    delay: int
    universe: str

class FieldRepository:
    def fetch_all(self, region: str, universe: str, delay: int) -> list[DataField]:
        """
        Phân trang qua offset/limit (mỗi trang ~50 fields).
        Cache kết quả vào DB (bảng data_fields).
        """

    def fetch_datasets(self) -> list[dict]:
        """GET /data-sets — lấy danh sách dataset categories"""
```

## Task 1.4 — Lấy Operators (`src/data/operators.py`)

```
GET /operators
```

Lưu danh sách operators kèm metadata (tên, số tham số, type input/output). Phục vụ validation và sinh alpha sau này.

## Task 1.5 — Storage cơ bản (`src/storage/`)

`models.py` (SQLAlchemy):

```python
class DataFieldModel(Base):
    __tablename__ = "data_fields"
    id = Column(String, primary_key=True)
    description = Column(Text)
    type = Column(String)
    region = Column(String)
    universe = Column(String)
    delay = Column(Integer)
    cached_at = Column(DateTime)

class AlphaModel(Base):
    __tablename__ = "alphas"
    id = Column(String, primary_key=True)
    expression = Column(Text, nullable=False)
    source = Column(String)        # template/ga/llm/random
    created_at = Column(DateTime)

class SimulationModel(Base):
    __tablename__ = "simulations"
    id = Column(String, primary_key=True)
    alpha_id = Column(String, ForeignKey("alphas.id"))
    region = Column(String)
    universe = Column(String)
    sharpe = Column(Float)
    fitness = Column(Float)
    turnover = Column(Float)
    drawdown = Column(Float)
    margin = Column(Float)
    returns = Column(Float)
    status = Column(String)        # passed/failed/error
    raw_result = Column(Text)      # full JSON
    sim_at = Column(DateTime)
```

`db.py`: tạo engine, session factory, `init_db()` để create tables.

## Task 1.6 — Simulator (`src/simulation/simulator.py`)

**Cơ chế simulation của WQ Brain:**

1. `POST /simulations` với JSON body chứa `type`, `settings`, `regular` (expression).
2. Response trả `Location` header chứa URL để **poll progress**.
3. Poll `GET {location}` lặp lại cho tới khi `status == "COMPLETE"` (hoặc đọc `Retry-After` header để biết delay).
4. Khi xong, response chứa `alpha` id → `GET /alphas/{alpha_id}` để lấy metrics (`is` block: sharpe, fitness, turnover, returns, drawdown, margin...).

```python
SIM_DEFAULTS = {
    "type": "REGULAR",
    "settings": {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 0,
        "neutralization": "SUBINDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "OFF",
        "language": "FASTEXPR",
        "visualization": False,
    },
}

class Simulator:
    def __init__(self, client: WQBrainClient):
        self.client = client

    def simulate(self, expression: str, settings: dict = None) -> SimulationResult:
        """
        1. POST /simulations  → lấy Location header
        2. Poll cho tới COMPLETE (tôn trọng Retry-After)
        3. GET /alphas/{id} → parse metrics
        4. Trả về SimulationResult
        """
```

**Quan trọng:** poll loop phải đọc header `Retry-After` (giây) nếu có, mặc định sleep 2-3s giữa các lần poll. Timeout tổng ~5 phút/simulation.

## Task 1.7 — Rate Limiter (`src/simulation/rate_limiter.py`)

```python
class RateLimiter:
    """
    - Giới hạn số simulation đồng thời (mặc định 3, tùy account tier).
    - Delay tối thiểu giữa 2 POST /simulations.
    - Dùng tenacity cho retry với exponential backoff khi gặp 429/500/503.
    """
```

Dùng thư viện `tenacity`:
```python
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_result
```

## Task 1.8 — CLI Phase 1 (`main.py`)

```bash
python main.py login                          # Test đăng nhập
python main.py fetch-fields                   # Lấy & cache data fields
python main.py fetch-operators                # Lấy & cache operators
python main.py simulate --expr "rank(close)"  # Chạy 1 simulation
python main.py simulate --expr "rank(ts_delta(close, 5))" --region USA --universe TOP3000
```

Dùng `typer` + `rich` để in bảng kết quả đẹp.

### ✅ Acceptance Phase 1

- [ ] `python main.py login` → đăng nhập thành công, in "OK".
- [ ] `python main.py fetch-fields` → lưu ≥ vài trăm fields vào DB, in số lượng.
- [ ] `python main.py simulate --expr "rank(close)"` → trả về Sharpe, Fitness, Turnover thật, lưu vào bảng `simulations`.
- [ ] Re-auth tự động khi session hết hạn.
- [ ] Có test cho client (mock response) và simulator.
- [ ] Log đầy đủ trong `logs/`.

---

# PHASE 2 — Sinh alpha + Genetic Algorithm

> Mục tiêu: tự động sinh hàng loạt alpha bằng template + GA, simulate, chấm điểm, giữ lại alpha tốt.

## Task 2.1 — AST Utilities (`src/generation/ast_utils.py`)

Biểu diễn expression dưới dạng cây để GA thao tác:

```python
@dataclass
class Node:
    op: str                    # tên operator, vd "rank", "ts_delta"
    children: list["Node | Leaf"]

@dataclass
class Leaf:
    value: str | int | float   # field name hoặc số

def to_expression(node) -> str:
    """Cây → chuỗi FASTEXPR"""

def parse_expression(expr: str) -> Node:
    """Chuỗi → cây (dùng lark hoặc parser thủ công)"""

def tree_depth(node) -> int: ...
def node_count(node) -> int: ...
def all_subtrees(node) -> list: ...   # phục vụ crossover
```

## Task 2.2 — Pre-filter (`src/simulation/pre_filter.py`)

Lọc syntax TRƯỚC khi simulate để khỏi phí quota:

```python
class PreFilter:
    def check(self, expr: str) -> tuple[bool, str]:
        """
        Trả (ok, reason).
        - Parse được thành AST không?
        - Cân bằng dấu ngoặc?
        - Operator có tồn tại trong DB operators?
        - Field có tồn tại trong DB fields?
        - Độ sâu <= max_depth (mặc định 6)?
        - Số node <= max_nodes (mặc định 30)?
        """
```

## Task 2.3 — Template Generator (`src/generation/template.py`)

```python
TEMPLATES = [
    "rank(ts_delta({field}, {d}))",
    "rank(ts_mean({field}, {d1}) - ts_mean({field}, {d2}))",
    "-rank(ts_zscore({field}, {d}))",
    "group_neutralize(rank({field}), {group})",
    "rank(ts_corr({f1}, {f2}, {d}))",
    # ... thêm 15-20 templates
]

PARAM_RANGES = {
    "d":  [5, 10, 20, 40, 60],
    "d1": [5, 10, 20],
    "d2": [20, 40, 60],
    "group": ["market", "sector", "industry", "subindustry"],
}

class TemplateGenerator:
    def generate(self, count: int) -> list[str]:
        """Random chọn template + điền field/param hợp lệ, qua pre-filter."""
```

## Task 2.4 — Scoring (`src/scoring/`)

`metrics.py`: chuẩn hóa metrics từ simulation result.

`scorer.py`:
```python
def score(m: dict) -> float:
    sharpe   = m.get("sharpe", 0)
    fitness  = m.get("fitness", 0)
    turnover = m.get("turnover", 0.5)
    drawdown = m.get("drawdown", 1.0)
    turnover_penalty = abs(turnover - 0.3)
    return (0.40*sharpe + 0.30*fitness
            + 0.15*(1 - drawdown) + 0.15*(1 - turnover_penalty))
```

`filter.py`: lọc đa tiêu chí.

| Metric | Ngưỡng |
|---|---|
| Sharpe | ≥ 1.25 |
| Fitness | > 1.0 |
| Turnover | 0.01 – 0.70 |
| Drawdown | < 0.20 |

## Task 2.5 — Genetic Algorithm (`src/optimization/evolution.py`)

**Trọng tâm của Phase 2.** Dùng `DEAP`.

**Biểu diễn cá thể (individual):** một cây AST (Node).

**Fitness:** điểm score từ simulation (cần simulate để đánh giá → cache để khỏi simulate trùng).

**Các phép biến đổi:**

```python
class GeneticOptimizer:
    def __init__(self, simulator, scorer, prefilter,
                 population_size=50, generations=20,
                 crossover_rate=0.4, mutation_rate=0.4):
        ...

    # --- Genetic operators ---
    def crossover(self, a: Node, b: Node) -> tuple[Node, Node]:
        """Hoán đổi 1 subtree ngẫu nhiên giữa a và b."""

    def mutate_field(self, ind: Node) -> Node:
        """Đổi 1 field leaf sang field cùng type."""

    def mutate_operator(self, ind: Node) -> Node:
        """Đổi operator sang operator cùng arity/type (ts_mean→ts_median...)."""

    def mutate_param(self, ind: Node) -> Node:
        """Đổi tham số số (d=5 → d=10)."""

    def mutate_wrap(self, ind: Node) -> Node:
        """Bọc thêm 1 lớp operator (x → rank(x), x → group_neutralize(x,sector))."""

    def evaluate(self, ind: Node) -> float:
        """to_expression → prefilter → simulate → score. Cache theo hash expr."""

    def run(self) -> list[Node]:
        """
        1. Khởi tạo population (dùng TemplateGenerator làm seed).
        2. Mỗi generation:
           - evaluate toàn bộ (song song trong giới hạn rate limit)
           - selection: giữ top-K (elitism) + tournament
           - sinh offspring: crossover (40%) + mutation (40%) + random (20%)
           - log best score của generation
        3. Trả top alpha sau max generation.
        """
```

**Lưu ý hiệu năng:** mỗi `evaluate` tốn 1 simulation (chậm + tốn quota). Vì vậy:
- **Cache cứng** theo hash của expression (đã simulate thì không simulate lại).
- Giới hạn population và generations hợp lý (population 30-50, gen 10-20 để bắt đầu).
- Evaluate song song nhưng tôn trọng `RateLimiter`.

## Task 2.6 — Bayesian tuning (`src/optimization/bayesian.py`) — optional trong Phase 2

Dùng `optuna` tinh chỉnh tham số số của một template tốt:

```python
def tune_template(template: str, simulator, n_trials=30):
    def objective(trial):
        d1 = trial.suggest_int("d1", 5, 60)
        d2 = trial.suggest_int("d2", d1+5, 120)
        expr = template.format(d1=d1, d2=d2)
        return simulator.simulate(expr).sharpe
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

## Task 2.7 — CLI Phase 2

```bash
python main.py generate --method template --count 100
python main.py run-ga --population 50 --generations 20 --region USA
python main.py tune --template "rank(ts_mean(close,{d1})-ts_mean(close,{d2}))"
python main.py top --n 20 --sort score
```

### ✅ Acceptance Phase 2

- [ ] Template generator sinh được 100 alpha hợp lệ qua pre-filter.
- [ ] GA chạy trọn vẹn ≥ 10 generations, best score tăng dần (log chứng minh).
- [ ] Cache hoạt động — không simulate trùng expression.
- [ ] `python main.py top` hiển thị bảng alpha tốt nhất.
- [ ] Test cho ast_utils, crossover, mutation (không cần gọi WQ thật — mock).

---

# PHASE 3 — LLM-assisted generation với DeepSeek

> Mục tiêu: dùng DeepSeek API sinh alpha từ ý tưởng ngôn ngữ tự nhiên + seed cho GA.

DeepSeek dùng API **tương thích OpenAI**, nên dùng thẳng thư viện `openai` với `base_url` trỏ về DeepSeek.

## Task 3.1 — DeepSeek Client (`src/llm/deepseek_client.py`)

```python
from openai import OpenAI

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "deepseek-chat"   # hoặc "deepseek-reasoner" cho reasoning

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"} if json_mode else None,
            temperature=1.0,
        )
        return resp.choices[0].message.content
```

## Task 3.2 — Alpha Generator bằng LLM (`src/llm/generator.py`)

```python
class LLMAlphaGenerator:
    def __init__(self, deepseek: DeepSeekClient, field_repo, operator_repo, prefilter):
        ...

    def build_system_prompt(self) -> str:
        """
        Liệt kê:
        - Cú pháp FASTEXPR cơ bản.
        - Danh sách operators hợp lệ (lấy từ DB).
        - Một subset fields liên quan (không nhồi hết — chọn ~30-50 field theo category).
        - 5-8 ví dụ alpha hợp lệ (few-shot).
        - Yêu cầu trả JSON: {"expression": "...", "rationale": "..."}
        """

    def generate(self, idea: str, n: int = 5) -> list[str]:
        """
        Gửi idea → nhận expression → pre-filter.
        Nếu fail syntax: gửi lại kèm error message (tối đa 3 retry).
        """

    def generate_ideas(self, n: int = 10) -> list[str]:
        """Cho LLM tự brainstorm n ý tưởng alpha (momentum, reversal, volume...)."""
```

**Pattern quan trọng — validation loop:**
```
1. LLM sinh expression
2. PreFilter.check(expr)
3. Nếu fail → gửi lại prompt + "Expression bị lỗi: {reason}. Sửa lại."
4. Lặp tối đa 3 lần. Vẫn fail → bỏ qua, log.
```

## Task 3.3 — Tích hợp LLM vào GA

LLM làm **seed thông minh** cho GA thay vì random:

```python
# Trong GeneticOptimizer.run():
# - 50% population khởi tạo từ LLMAlphaGenerator (đa dạng ý tưởng)
# - 50% từ TemplateGenerator
# → GA tiến hóa từ điểm xuất phát tốt hơn.
```

Ngoài ra, dùng LLM **giải thích & đề xuất mutation** cho alpha tốt: đưa alpha + metrics cho DeepSeek, hỏi "cải thiện thế nào?", parse expression mới.

## Task 3.4 — CLI Phase 3

```bash
python main.py llm-generate --idea "momentum ngắn hạn kết hợp volume" --count 5
python main.py llm-ideas --count 10
python main.py run-ga --seed-llm --population 50 --generations 20
```

### ✅ Acceptance Phase 3

- [ ] Gọi DeepSeek thành công, sinh expression hợp lệ.
- [ ] Validation loop hoạt động (tự sửa khi syntax sai).
- [ ] GA chạy được với seed từ LLM, log cho thấy alpha seed chất lượng tốt hơn random.
- [ ] Chi phí API được log (số token, ước tính cost).

---

# PHASE 4 — Submission Manager + Dashboard

> Mục tiêu: tự động nộp alpha đạt ngưỡng + dashboard theo dõi.

## Task 4.1 — Correlation Check (`src/submission/correlation.py`)

WQ Brain có endpoint check correlation của alpha với các alpha đã có:
```
GET /alphas/{alpha_id}/correlations/self
```

```python
class CorrelationChecker:
    MAX_SELF_CORR = 0.70
    def is_acceptable(self, alpha_id: str) -> bool:
        """Lấy max self-correlation, so với ngưỡng."""
```

## Task 4.2 — Submission Manager (`src/submission/manager.py`)

```python
class SubmissionManager:
    MIN_SHARPE = 1.5
    MIN_FITNESS = 1.2
    DAILY_QUOTA = 10

    def select_candidates(self) -> list[Alpha]:
        """Query DB: alpha passed ngưỡng, chưa nộp, sắp theo score."""

    def submit(self, alpha_id: str) -> SubmissionResult:
        """
        1. Check correlation (CorrelationChecker)
        2. POST submit lên WQ Brain
        3. Lưu trạng thái vào bảng submissions
        4. Retry với backoff nếu 429/503
        """

    def run_daily(self, dry_run: bool = True):
        """Chọn ≤ DAILY_QUOTA alpha tốt nhất, không trùng correlation, nộp."""
```

## Task 4.3 — Dashboard (`dashboard/app.py`) — Streamlit

Các tab:
- **Overview:** tổng alpha, pass rate, avg Sharpe, biểu đồ phân phối.
- **Explorer:** bảng alpha có filter/sort, xem chi tiết expression + metrics.
- **GA Progress:** line chart best/avg score theo generation.
- **Submissions:** trạng thái alpha đã nộp.
- **Correlation:** heatmap correlation giữa top alpha (plotly).

```bash
streamlit run dashboard/app.py
```

### ✅ Acceptance Phase 4

- [ ] `python main.py submit --dry-run` liệt kê đúng alpha sẽ nộp (không trùng correlation).
- [ ] Submit thật ghi nhận trạng thái vào DB.
- [ ] Dashboard chạy, hiển thị data thật từ DB.

---

## Thứ tự ưu tiên tuyệt đối

```
Phase 1 (login + fetch + simulate)   ← LÀM TRƯỚC, TEST KỸ
        ↓
Phase 2 (template + GA + scoring)    ← Trọng tâm thuật toán
        ↓
Phase 3 (DeepSeek LLM)               ← Tăng chất lượng seed
        ↓
Phase 4 (submission + dashboard)     ← Vận hành & theo dõi
```

**Không bắt đầu Phase 2 khi Phase 1 chưa simulate được một alpha thật và lưu được metrics vào DB.** Toàn bộ phần GA và LLM đều phụ thuộc vào việc simulate hoạt động ổn định.

---

## Checklist xử lý lỗi cần lưu ý

- **Session hết hạn giữa chừng** → tự re-authenticate, retry request.
- **Simulation timeout** → đánh dấu status=error, không crash pipeline.
- **Rate limit (429)** → backoff theo `Retry-After`, không retry vô hạn.
- **Expression lỗi từ WQ** → parse message lỗi, lưu để debug, bỏ qua alpha đó.
- **DeepSeek trả JSON sai format** → wrap parse trong try/except, retry.
- **DB lock (SQLite)** → dùng WAL mode, hoặc chuyển PostgreSQL khi chạy song song nhiều.

---

*Bắt đầu từ Setup → Phase 1. Mỗi task hoàn thành thì commit. Kết thúc mỗi Phase phải pass toàn bộ acceptance criteria trước khi đi tiếp.*
