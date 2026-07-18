MiniBrain - Local WorldQuant Research Platform
1. Mục tiêu

Xây dựng hệ thống nghiên cứu alpha cục bộ có khả năng:

Đồng bộ DataFields từ WorldQuant Brain
Đồng bộ Operators từ WorldQuant Brain
Parse Fast Expression
Backtest cục bộ
Tính Sharpe, Turnover, Fitness
Genetic Programming tự sinh alpha
Lọc alpha trùng lặp
Xếp hạng alpha
Chọn alpha tiềm năng trước khi submit lên Brain

Lưu ý:

Mục tiêu KHÔNG phải clone hoàn toàn WorldQuant Brain.

Mục tiêu là:

Giảm số lần submit
Loại bỏ alpha kém
Tăng tốc nghiên cứu
2. Kiến trúc tổng thể
                    WorldQuant Brain
                            |
        -----------------------------------------
        |                                       |
 DataField Fetcher                    Operator Fetcher
        |                                       |
        ----------------Cache-------------------
                            |
                    Expression Parser
                            |
                    Evaluation Engine
                            |
                      Backtest Engine
                            |
                 Sharpe/Fitness Calculator
                            |
                  Alpha Research Database
                            |
                  Genetic Programming Engine
                            |
                    Candidate Selection
                            |
                    Submit To Brain
3. Công nghệ đề xuất
Ngôn ngữ

Python 3.12

Thư viện
Core
numpy
pandas
scipy
Performance
numba
bottleneck
pyarrow
Parallel
ray

hoặc

joblib
Genetic Programming
deap
Parser
lark
Database
sqlite
4. Cấu trúc dự án
MiniBrain/

├── data/
│   ├── parquet/
│   ├── universe/
│   └── cache/
│
├── operators/
│   ├── cross_sectional.py
│   ├── timeseries.py
│   └── transforms.py
│
├── parser/
│   ├── lexer.py
│   ├── grammar.lark
│   └── evaluator.py
│
├── engine/
│   ├── alpha_engine.py
│   ├── portfolio.py
│   └── backtester.py
│
├── gp/
│   ├── generator.py
│   ├── mutation.py
│   ├── crossover.py
│   └── evolution.py
│
├── database/
│   └── alpha.db
│
├── api/
│   ├── datafield_fetcher.py
│   └── operator_fetcher.py
│
└── main.py
5. DataField Fetcher
Mục tiêu

Đồng bộ DataFields về local.

Ví dụ:

close
open
high
low
volume
vwap
returns
market_cap

Lưu:

{
  "id":"close",
  "type":"MATRIX",
  "category":"price"
}
6. Operator Fetcher

Lấy toàn bộ operator khả dụng.

Ví dụ:

rank
zscore
scale

ts_rank
ts_mean
ts_std_dev
ts_corr

group_rank
group_neutralize

Sinh mapping tự động.

operator_map["rank"] = rank
operator_map["ts_mean"] = ts_mean
7. Data Storage

Khuyến nghị dùng Parquet.

Ví dụ:

date
ticker
close
volume
vwap

Không dùng CSV.

Lý do:

nhanh hơn
ít RAM hơn
dễ phân vùng
8. Alpha Parser

Ví dụ expression:

-rank(ts_mean((close-open)/open,5))

AST:

UnaryMinus
└── Rank
    └── TsMean
        └── Divide

Parser:

tree = parser.parse(expression)

Evaluator:

result = evaluate(tree)
9. Operator Engine

Ví dụ:

rank
rank(x)

Cross-sectional ranking.

ts_mean
ts_mean(x,20)

Rolling mean.

ts_rank
ts_rank(x,20)

Ranking trong cửa sổ.

ts_corr
ts_corr(x,y,20)

Rolling correlation.

10. Portfolio Construction
Bước 1

Tạo alpha signal.

alpha
Bước 2

Rank.

weight = alpha.rank()
Bước 3

Neutralize.

weight -= weight.mean()
Bước 4

Normalize.

weight /= abs(weight).sum()
11. Backtest Engine

Daily return:

daily_pnl =
(weight.shift(1) * returns).sum(axis=1)

Output:

equity_curve
daily_return
12. Metrics
Sharpe
sharpe =
mean(ret) /
std(ret) *
sqrt(252)
Turnover
turnover =
abs(weight-weight.shift(1)).sum()
Drawdown
max_drawdown
CAGR
cagr
13. Fitness Approximation

Brain không công khai công thức chính xác.

Gần đúng:

fitness =
sharpe *
sqrt(
    abs(cagr) /
    max(turnover,0.125)
)

Dùng để so sánh tương đối.

14. Genetic Programming
Terminal Set
close
open
high
low
volume
vwap
returns
Function Set
rank
zscore
scale

ts_mean
ts_std_dev
ts_rank
ts_corr
ts_delta
15. Alpha Generator

Ví dụ alpha ngẫu nhiên:

rank(
    ts_mean(
        volume,
        20
    )
)

Hoặc:

ts_rank(
    ts_corr(
        close,
        volume,
        10
    ),
    20
)
16. Mutation

Ví dụ:

close

↓

vwap

Hoặc:

20

↓

60
17. Crossover

Parent A

rank(ts_mean(close,20))

Parent B

ts_rank(volume,60)

Child

rank(ts_rank(volume,60))
18. Evolution Strategy

Population:

1000 alpha

Mỗi generation:

Top 10%

được giữ lại.

Các alpha còn lại:

mutation
crossover

để sinh thế hệ mới.

19. Similarity Filter

Hash AST.

Ví dụ:

rank(ts_mean(close,20))

và

rank(ts_mean(close,21))

được xem là gần giống.

Giữ alpha tốt nhất.

20. Database

Schema:

CREATE TABLE alpha
(
    id INTEGER PRIMARY KEY,
    expression TEXT,
    sharpe REAL,
    turnover REAL,
    fitness REAL,
    created_at DATETIME
);
21. Parallel Evaluation

Sử dụng Ray.

@ray.remote
def evaluate_alpha(expr):
    ...

Có thể chạy:

10,000+

alpha/ngày trên CPU.

22. Roadmap

Phase 1

DataField Fetcher
Operator Fetcher

Phase 2

Parser
Operator Engine

Phase 3

Backtester
Metrics

Phase 4

GP Engine

Phase 5

Dashboard

Phase 6

Auto Submit Assistant
Kỳ vọng thực tế

Hệ thống này không thể tái tạo đúng Sharpe của WorldQuant Brain.

Tuy nhiên nó có thể:

Loại bỏ 80-95% alpha kém
Giảm số lần submit
Tìm alpha tiềm năng nhanh hơn nhiều
Chạy hàng nghìn alpha mỗi ngày
Tạo nền tảng nghiên cứu định lượng lâu dài