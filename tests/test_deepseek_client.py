import unittest
from unittest.mock import Mock

from deepseek_client import (
    DeepSeekClient,
    DeepSeekConfigurationError,
    DeepSeekResponseError,
)
from research_config import ResearchConfig


def fake_response(status, data, headers=None):
    response = Mock(status_code=status, headers=headers or {})
    response.json.return_value = data
    return response


class DeepSeekClientTest(unittest.TestCase):
    def setUp(self):
        self.config = ResearchConfig()

    def _client(self, session, environ=None, sleep_func=None):
        return DeepSeekClient(
            self.config,
            session=session,
            environ=environ if environ is not None else {"DEEPSEEK_API_KEY": "secret-key"},
            sleep_func=sleep_func or Mock(),
        )

    def test_reads_environment_key_and_returns_usage(self):
        session = Mock()
        session.post.return_value = fake_response(200, {
            "choices": [{"message": {"content": '{"alphas": []}'}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "prompt_cache_hit_tokens": 80,
            },
        })
        client = self._client(session)

        result = client.generate_json(
            "ROOT_ALPHA",
            "Return JSON with an alphas key.",
            {"idea": "reversal"},
        )

        self.assertEqual(result.data, {"alphas": []})
        self.assertEqual(result.usage.prompt_tokens, 120)
        self.assertEqual(result.usage.cache_hit_tokens, 80)
        headers = session.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer secret-key")
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(session.post.call_args.kwargs["timeout"], self.config.deepseek_timeout_seconds)

    def test_missing_key_raises_configuration_error(self):
        with self.assertRaises(DeepSeekConfigurationError):
            self._client(Mock(), environ={})

    def test_empty_and_invalid_json_retry_then_fail(self):
        session = Mock()
        session.post.side_effect = [
            fake_response(200, {"choices": [{"message": {"content": ""}}], "usage": {}}),
            fake_response(200, {"choices": [{"message": {"content": "not json"}}], "usage": {}}),
            fake_response(200, {"choices": [{"message": {"content": "still bad"}}], "usage": {}}),
        ]
        client = self._client(session)

        with self.assertRaises(DeepSeekResponseError) as context:
            client.generate_json("ROOT_ALPHA", "sys", {"idea": "x"})

        self.assertEqual(session.post.call_count, self.config.deepseek_max_retries + 1)
        self.assertNotIn("secret-key", str(context.exception))

    def test_rate_limit_respects_retry_after(self):
        session = Mock()
        session.post.side_effect = [
            fake_response(429, {}, headers={"Retry-After": "3"}),
            fake_response(200, {
                "choices": [{"message": {"content": '{"alphas": []}'}}],
                "usage": {},
            }),
        ]
        sleep_mock = Mock()
        client = self._client(session, sleep_func=sleep_mock)

        result = client.generate_json("ROOT_ALPHA", "sys", {"idea": "x"})

        self.assertEqual(result.data, {"alphas": []})
        sleep_mock.assert_any_call(3.0)


if __name__ == "__main__":
    unittest.main()
