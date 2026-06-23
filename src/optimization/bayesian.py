"""Tinh chỉnh tham số số của một template tốt bằng optuna (optional)."""

from __future__ import annotations

import optuna
from loguru import logger

from src.scoring.scorer import score as default_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_template(
    template: str,
    simulator,
    n_trials: int = 30,
    scorer=default_score,
    simulation_settings: dict | None = None,
) -> dict:
    """Tối ưu d1/d2 trong template dạng `...{d1}...{d2}...` để max score."""

    def objective(trial: optuna.Trial) -> float:
        d1 = trial.suggest_int("d1", 5, 60)
        d2 = trial.suggest_int("d2", d1 + 5, 120)
        expr = template.format(d1=d1, d2=d2)
        if simulation_settings is None:
            result = simulator.simulate(expr)
        else:
            result = simulator.simulate(expr, settings=simulation_settings)
        return scorer(result)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best params: {} (value={:.4f})", study.best_params, study.best_value)
    return study.best_params
