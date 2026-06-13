"""Test MCTSSearch: cây tìm kiếm UCB + backprop, thay greedy (GĐ6: T6.1).

Thuần thuật toán — inject expand_fn/evaluate_fn/weakest_fn nên không cần LLM/sim thật.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.mcts import MCTSSearch


@dataclass
class _Cand:
    expression: str


@dataclass
class _Vec:
    total: float

    def dimensions(self):
        return {"sharpe": self.total}


@dataclass
class _Eval:
    effective_total: float
    vector: _Vec
    metrics: dict
    alpha_id: str | None = None


def _eval(total):
    return _Eval(total, _Vec(total), {"sharpe": total}, alpha_id="a" + str(total))


def _weakest(_vec):
    return "sharpe"


class _Expander:
    """expand_fn giả: mỗi lần sinh một expr duy nhất với reward lấy từ map theo bậc gọi."""

    def __init__(self, rewards):
        self._rewards = list(rewards)
        self.i = 0
        self.expand_calls = 0

    def expand(self, candidate, metrics, weak):
        self.expand_calls += 1
        return _Cand(f"e{self.i}")

    def evaluate(self, candidate, parent_id):
        if self.i >= len(self._rewards):
            return None  # hết "ngân sách"
        r = self._rewards[self.i]
        self.i += 1
        return _eval(r) if r is not None else None


def test_search_tra_best_reward_cao_nhat():
    exp = _Expander([1.0, 2.0, 1.5])
    search = MCTSSearch(exp.expand, exp.evaluate, _weakest, max_iterations=3)
    res = search.search(_Cand("seed"), _eval(0.5))
    assert res.best_eval.effective_total == 2.0
    assert res.best_candidate.expression == "e1"  # node có reward 2.0


def test_search_ton_trong_max_iterations():
    exp = _Expander([1.0] * 100)
    search = MCTSSearch(exp.expand, exp.evaluate, _weakest, max_iterations=5)
    search.search(_Cand("seed"), _eval(0.5))
    assert exp.i == 5  # đúng 5 lần evaluate thật


def test_search_dung_khi_het_ngan_sach_evaluate():
    # evaluate trả None liên tục (hết ngân sách) -> dừng sớm nhờ patience.
    exp = _Expander([None, None, None])
    search = MCTSSearch(exp.expand, exp.evaluate, _weakest, max_iterations=50, none_patience=3)
    res = search.search(_Cand("seed"), _eval(0.5))
    assert res.best_candidate.expression == "seed"  # không có cải thiện -> giữ seed


def test_backprop_cap_nhat_visits_len_to_tien():
    exp = _Expander([1.0, 2.0])
    search = MCTSSearch(exp.expand, exp.evaluate, _weakest, max_iterations=2)
    res = search.search(_Cand("seed"), _eval(0.5))
    # root được thăm lại mỗi vòng backprop -> visits >= số vòng + 1 (khởi tạo).
    assert res.root.visits >= 3


def test_ucb_uu_tien_node_chua_tham():
    exp = _Expander([1.0, 2.0, 3.0])
    search = MCTSSearch(exp.expand, exp.evaluate, _weakest, max_iterations=3, max_children=5)
    res = search.search(_Cand("seed"), _eval(0.5))
    # với max_children cao, cả 3 con đều mở từ root (khám phá rộng, không kẹt 1 nhánh).
    assert len(res.root.children) == 3


def test_search_giu_history_cac_vong():
    exp = _Expander([1.0, 2.0, 1.5])
    search = MCTSSearch(exp.expand, exp.evaluate, _weakest, max_iterations=3)
    res = search.search(_Cand("seed"), _eval(0.5))
    assert len(res.history) == 3
    assert all("total" in h and "expression" in h for h in res.history)
