"""MCTS thay vòng greedy: giữ nhiều nhánh ứng viên, UCB + lan ngược điểm (T6.1).

Greedy ở GĐ2 chỉ bám một alpha tốt nhất; MCTS giữ cây nhiều nhánh, cân bằng khám
phá/khai thác bằng UCB và lan ngược (backprop) điểm qua tổ tiên. Thuần thuật toán:
nhận `expand_fn`/`evaluate_fn`/`weakest_fn` để tách khỏi LLM/simulation (dễ test).

- expand_fn(candidate, metrics, weak) -> candidate mới (đề xuất cải tiến).
- evaluate_fn(candidate, parent_id) -> _Eval (có .effective_total/.vector/.metrics/
  .alpha_id) hoặc None khi hết ngân sách/parse lỗi.
- weakest_fn(vector) -> tên chiều yếu nhất để nhắm cải thiện.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from loguru import logger

EXPLORATION = 1.4


class MCTSNode:
    def __init__(self, candidate, evaluation, parent=None):
        self.candidate = candidate
        self.eval = evaluation
        self.parent = parent
        self.children: list = []
        self.visits = 1
        self.total_reward = float(evaluation.effective_total)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits


@dataclass
class MCTSResult:
    best_candidate: object
    best_eval: object
    root: MCTSNode
    history: list = field(default_factory=list)


class MCTSSearch:
    def __init__(
        self,
        expand_fn,
        evaluate_fn,
        weakest_fn,
        max_iterations: int = 20,
        max_children: int = 3,
        exploration: float = EXPLORATION,
        none_patience: int | None = None,
    ):
        self.expand_fn = expand_fn
        self.evaluate_fn = evaluate_fn
        self.weakest_fn = weakest_fn
        self.max_iterations = max_iterations
        self.max_children = max_children
        self.exploration = exploration
        self.none_patience = none_patience

    # ----------------------------------------------------------- selection
    def _ucb(self, node: MCTSNode, parent_visits: int) -> float:
        explore = self.exploration * math.sqrt(math.log(parent_visits) / node.visits)
        return node.mean_reward + explore

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Đi xuống theo UCB tới node còn chỗ mở rộng (số con < max_children)."""
        node = root
        while len(node.children) >= self.max_children and node.children:
            node = max(node.children, key=lambda c: self._ucb(c, node.visits))
        return node

    @staticmethod
    def _backprop(node: MCTSNode, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.total_reward += reward
            cur = cur.parent

    # --------------------------------------------------------------- search
    def search(self, seed_candidate, seed_eval) -> MCTSResult:
        root = MCTSNode(seed_candidate, seed_eval)
        best_candidate, best_eval = seed_candidate, seed_eval
        history: list = []
        none_streak = 0

        for _ in range(self.max_iterations):
            node = self._select(root)
            weak = self.weakest_fn(node.eval.vector)
            cand = self.expand_fn(node.candidate, node.eval.metrics, weak)
            child_eval = self.evaluate_fn(cand, node.eval.alpha_id)
            if child_eval is None:
                none_streak += 1
                if self.none_patience is not None and none_streak >= self.none_patience:
                    break
                continue
            none_streak = 0

            child = MCTSNode(cand, child_eval, parent=node)
            node.children.append(child)
            self._backprop(node, child_eval.effective_total)
            history.append(
                {"total": child_eval.vector.total, "expression": cand.expression, "dimension": weak}
            )
            if child_eval.effective_total > best_eval.effective_total + 1e-9:
                best_candidate, best_eval = cand, child_eval

        logger.info(
            "MCTS xong: {} vòng, best total={:.3f}",
            len(history), best_eval.effective_total,
        )
        return MCTSResult(best_candidate, best_eval, root, history)
