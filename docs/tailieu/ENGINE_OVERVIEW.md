# Engine sinh Alpha — Tổng quan kiến trúc

> Sau đợt hợp nhất (`tailieu/CONSOLIDATE_TO_REFINEMENT_LOOP.md`): chỉ còn **một**
> engine sinh alpha là `RefinementLoop`. HybridEngine + GA (`src/optimization/{hybrid,
> evolution}.py`, `SynergyScorer`, CLI `auto`/`start`) đã được gỡ. Diversity của
> engine cũ (`NOVEL_ALPHAS` làm sàn seed + re-seed định kỳ) **đã hợp nhất vào
> RefinementLoop** (Task 1).

## 1. Engine duy nhất: RefinementLoop

| Engine | File | Chiến lược | CLI |
|---|---|---|---|
| **RefinementLoop** | `src/llm/loop.py` | LLM giả thuyết → dịch → sim → tinh chỉnh **tham lam** chiều yếu (hoặc **MCTS** qua `--mcts`) | `research` |

Pipeline điều phối thuần `src/pipeline/auto.py` (`AutoPipeline`) vẫn còn (test-only,
ngoài phạm vi đợt hợp nhất).

## 2. Luồng dữ liệu

```
HẠ TẦNG DÙNG CHUNG: WQBrainClient → Field/OperatorRepository → cache DB
                    PreFilter (cú pháp/type/arity, local) → Simulator (pre_sim_validator)

research <direction>
  → seed_candidates(direction):
        seed LLM (HypothesisGenerator → AlphaTranslator)  ← ưu tiên
        + NOVEL_ALPHAS (lọc qua PreFilter)                ← sàn đa dạng / fallback
  → _evaluate() cho seed đầu tiên hợp lệ:
        prefilter → originality(AST zoo) → cache? → alignment → SIM THẬT
        → score_vector (6 chiều) → [corr / OOS / regime gate]
  → vòng refine tham lam: refiner.refine(chiều yếu nhất đang chặn)
        + re-seed định kỳ (--reseed-every): nhánh kẹt N vòng → idea_generator
          sinh direction mới (LLM re-seed, KHÔNG GA)
  → lặp đến trần sim / hết patience
  → save_alpha / save_simulation → DB → submit (SubmissionManager + CorrelationChecker)
```

## 3. Chấm điểm — `ScoreVector` 6 chiều (`src/scoring/vector.py`)

| sharpe | fitness | pool_fit | regime_fit | drawdown | turnover |
|---|---|---|---|---|---|
| 0.25 | 0.20 | 0.25 | 0.15 | 0.075 | 0.075 |

`pool_fit` (trực giao với pool) và `regime_fit` (ổn định theo năm) là hạng nhất.
`scoring/filter.py` = hard gate + `blocking_dimensions` (chiều đang chặn → refiner nhắm).

> Lưu ý: `SynergyScorer` (objective pool-aware cho GA) đã bị gỡ cùng GA. `scoring/scorer.py`
> (`default_score`) còn lại phục vụ `optimization/bayesian.py` (tune template, độc lập).

## 4. Các cổng (gate) trong `_evaluate` — đều thuộc RefinementLoop

```
prefilter → originality (AST, ngưỡng 0.35) → cache? → alignment
   → SIM → hard filter → corr-với-pool (crowded ≥0.70) → OOS → regime
```

`pool_corr_fn` luôn được `main` lắp (corr là first-class); corr/OOS/regime bật/tắt
qua cờ CLI (`--oos-ratio`, `--min-annual-sharpe`, `--align-soft`...).

## 5. Cờ điều khiển `research`

`--mcts/--greedy` · `--align/--no-align` · `--align-soft` · `--regularize` + `--lambda`
· `--oos-ratio` · `--deflate` · `--min-annual-sharpe` · `--improve-margin`
· **`--reseed-every`** (re-seed diversity, mặc định 0 = tắt).

## 6. Diversity (kế thừa từ engine cũ)

- `NOVEL_ALPHAS` (`src/generation/novel_ideas.py`) — seed dataset thay thế, lọc qua
  PreFilter, trộn vào `seed_candidates` để loop không sụp về một seed LLM.
- Re-seed định kỳ (`--reseed-every`) thay cho cơ chế "inject mỗi K thế hệ" của GA.
- `ReferenceZoo` (AST-similarity, dùng chung) + tự-học vùng chết (`InvalidFieldRepository`).
