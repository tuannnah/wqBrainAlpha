"""Tiện ích test: fake HTTP response/client để khỏi gọi WQ Brain thật."""

from __future__ import annotations


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    """Trả response theo hàng đợi cho từng (method, path-prefix)."""

    def __init__(self):
        self.calls = []
        self._get_queue = []
        self._post_queue = []

    def queue_get(self, response):
        self._get_queue.append(response)

    def queue_post(self, response):
        self._post_queue.append(response)

    def authenticate(self):
        self._authenticated = True

    def get(self, path, **kwargs):
        self.calls.append(("GET", path, kwargs))
        return self._get_queue.pop(0)

    def post(self, path, **kwargs):
        self.calls.append(("POST", path, kwargs))
        return self._post_queue.pop(0)


class FakeDeepSeek:
    """LLM giả: trả lần lượt nội dung trong hàng đợi cho mỗi complete()."""

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []  # [(system, user)]

    def queue(self, content):
        self._responses.append(content)

    def complete(self, system, user, json_mode=True):
        self.calls.append((system, user))
        if not self._responses:
            return "{}"
        return self._responses.pop(0)


class FakeSimulator:
    """Simulator giả: map biểu thức -> SimulationResult, đếm số lần gọi."""

    def __init__(self, results=None, default=None):
        # results: dict expr -> SimulationResult, hoặc callable(expr)->result
        self._results = results or {}
        self._default = default
        self.calls = []

    def simulate(self, expression, settings=None):
        self.calls.append(expression)
        if callable(self._results):
            return self._results(expression)
        if expression in self._results:
            return self._results[expression]
        if self._default is not None:
            return self._default(expression) if callable(self._default) else self._default
        from src.simulation.simulator import SimulationResult

        return SimulationResult(expression=expression, status="error")
