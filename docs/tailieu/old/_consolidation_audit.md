# Audit phụ thuộc — Hợp nhất engine về RefinementLoop (Task 0)

Khảo sát mọi tham chiếu trước khi gỡ. `grep` chạy trên `src/ tests/ scripts/ main.py`
(bỏ qua `docs/` vì chỉ là spec/plan lịch sử, không phải code chạy).

## Bảng quyết định symbol

| Symbol | File định nghĩa | Được import / dùng bởi | Quyết định |
|---|---|---|---|
| `HybridEngine` | `src/optimization/hybrid.py` | `main.py:27,1100`; `tests/test_hybrid.py` | **XÓA** (Engine B) |
| `GeneticOptimizer` | `src/optimization/evolution.py` | `hybrid.py:20,115,136,138`; `tests/test_evolution.py` | **XÓA** (lõi GA, chỉ B dùng) |
| `SynergyScorer` | `src/scoring/synergy.py` | `main.py:1107` (`_run_auto` = B); `tests/test_synergy.py` | **XÓA** (chỉ B dùng — A dùng `score_vector` thẳng, không đụng `SynergyScorer`) |
| `score`/`default_score` | `src/scoring/scorer.py` | `evolution.py`, `bayesian.py`, `hybrid.py` | **GIỮ** (`bayesian.py` còn dùng sau khi gỡ GA) |
| `tune_template` | `src/optimization/bayesian.py` | `tests/test_bayesian.py` | **GIỮ** (độc lập optuna, KHÔNG import GeneticOptimizer, KHÔNG phục vụ B) |
| `NOVEL_ALPHAS` | `src/generation/novel_ideas.py` | `hybrid.py:17,73`; `scripts/generate_novel.py`; `tests/test_novel_ideas.py`, `test_hybrid.py` | **GIỮ + SALVAGE** vào `loop.py` (Task 1a) |
| `ReferenceZoo` | `src/decorrelation/zoo.py` | A (`loop.py`/`main._make_research_loop`) **và** B | **GIỮ** (dùng chung) |
| `AutoPipeline`/`PrepareInfo` | `src/pipeline/auto.py` | `main.py:32` (`PrepareInfo` qua `_auto_prepare`); `tests/test_auto_pipeline.py` | **GIỮ** (không phải GA/Hybrid — ngoài phạm vi spec; `_auto_prepare` đã là dead code từ trước, không do refactor này tạo ra) |

## CLI / helper trong `main.py`

| Đối tượng | Vị trí | Quyết định |
|---|---|---|
| lệnh `auto` | `main.py:1129` | **XÓA** |
| lệnh `start` + menu (`_MenuState`, `_menu_login/_menu_fields/_menu_operators`, `_print_menu`, `_menu_ask_sim_settings`, `_menu_ask_neutralization`, `_NEUTRALIZATION_MENU`) | `main.py:1151-1304` | **XÓA** (chỉ phục vụ menu của B) |
| `_run_auto` | `main.py:1055` | **XÓA** (chỉ dựng HybridEngine) |
| `_run_hybrid_with_progress` | `main.py:1031` | **XÓA** |
| `_make_refiner` | `main.py:1018` | **XÓA** (chỉ `_run_auto` gọi; A dùng `AlphaRefiner` thẳng trong `_make_research_loop`) |
| cờ `--no-llm-seed` / `no_llm_seed` | `main.py:1139,1108,1058` | **XÓA** |
| lệnh `research` + `_make_research_loop`, `_run_research_with_progress`, `_render_research_result` | `main.py:640-837` | **GIỮ** (Engine A — lệnh chính) |
| `import HybridEngine` | `main.py:27` | **XÓA** |

## Test cần xử lý

| File test | Quyết định |
|---|---|
| `tests/test_hybrid.py` | **XÓA** (chỉ B) |
| `tests/test_evolution.py` | **XÓA** (GA) |
| `tests/test_synergy.py` | **XÓA** (SynergyScorer chỉ B) |
| `tests/test_bayesian.py` | **GIỮ** (bayesian giữ) |
| `tests/test_novel_ideas.py` | **GIỮ** |
| `tests/test_auto_pipeline.py` | **GIỮ** (AutoPipeline giữ) |
| `tests/test_auto_command.py` | **TÁCH**: giữ `test_simulate_command_*` + `test_research_truyen_*`; xóa `test_run_auto_*`, `test_start_menu_*`, `test_lenh_auto_*`, `test_menu_neutralization_*` (4) — vì lệnh auto/start/menu bị gỡ |

## Module dùng chung (KHÔNG xóa theo spec)

`scoring/` (trừ `synergy.py`), `decorrelation/zoo.py`, `generation/novel_ideas.py`,
`generation/families.py`, `local_select.py`, `template.py`, `simulation/*`, `submission/*`.

## Tóm tắt việc xóa

- File xóa: `src/optimization/hybrid.py`, `src/optimization/evolution.py`, `src/scoring/synergy.py`,
  `tests/test_hybrid.py`, `tests/test_evolution.py`, `tests/test_synergy.py`.
- File giữ trong `optimization/`: `bayesian.py` (+ `__init__.py` rỗng).
- `main.py`: gỡ block auto/start/menu + helper B; `research` thành lệnh chính.
