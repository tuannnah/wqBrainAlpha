import unittest
from unittest.mock import Mock

from research_models import Scope
from worldquant_client import (
    WorldQuantApiError,
    WorldQuantClient,
    WorldQuantRateLimitError,
)


def response(status, data=None, headers=None):
    item = Mock(status_code=status, headers=headers or {})
    item.json.return_value = data if data is not None else {}
    return item


def authed_session():
    """Mock session đã trả về xác thực thành công ở constructor."""
    session = Mock()
    session.post.return_value = response(201, {})
    return session


class WorldQuantMetadataClientTest(unittest.TestCase):
    def test_iterates_all_pages_with_scope_parameters(self):
        session = authed_session()
        session.get.side_effect = [
            response(200, {"count": 3, "results": [{"id": "a"}, {"id": "b"}]}),
            response(200, {"count": 3, "results": [{"id": "c"}]}),
        ]
        client = WorldQuantClient("user@example.com", "secret", session=session)

        rows = list(client.iter_data_fields(
            Scope("EQUITY", "USA", 1, "TOP3000"),
            limit=2,
        ))

        self.assertEqual([row["id"] for row in rows], ["a", "b", "c"])
        self.assertEqual(session.get.call_args_list[0].kwargs["params"]["offset"], 0)
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["offset"], 2)
        self.assertEqual(session.get.call_args_list[0].kwargs["params"]["region"], "USA")
        self.assertEqual(session.get.call_args_list[0].kwargs["timeout"], 30)

    def test_requests_configuration_categories_and_operators(self):
        session = authed_session()
        session.get.return_value = response(200, [{"id": "cat1"}])
        client = WorldQuantClient("user@example.com", "secret", session=session)

        client.get_categories()
        session.get.assert_called_with(
            "https://api.worldquantbrain.com/data-categories",
            params=None,
            timeout=30,
        )
        client.get_operators()
        session.get.assert_called_with(
            "https://api.worldquantbrain.com/operators",
            params=None,
            timeout=30,
        )
        client.get_configuration()
        session.get.assert_called_with(
            "https://api.worldquantbrain.com/configuration",
            params=None,
            timeout=30,
        )

    def test_rate_limit_raises_with_retry_after(self):
        session = authed_session()
        session.get.return_value = response(429, {}, headers={"Retry-After": "7"})
        client = WorldQuantClient("user@example.com", "secret", session=session)

        with self.assertRaises(WorldQuantRateLimitError) as context:
            client.get_configuration()
        self.assertEqual(context.exception.retry_after_seconds, 7.0)

    def test_api_error_for_other_failures(self):
        session = authed_session()
        session.get.return_value = response(500, {})
        client = WorldQuantClient("user@example.com", "secret", session=session)

        with self.assertRaises(WorldQuantApiError):
            client.get_operators()

    def test_extract_scopes_from_configuration(self):
        configuration = {
            "actions": {"POST": {"settings": {"children": [
                {"name": "instrumentType", "options": [
                    {"value": "EQUITY", "children": [
                        {"name": "region", "options": [
                            {"value": "USA", "children": [
                                {"name": "universe", "options": [
                                    {"value": "TOP3000"},
                                    {"value": "TOP1000"},
                                ]},
                                {"name": "delay", "options": [
                                    {"value": 1},
                                    {"value": 0},
                                ]},
                            ]},
                        ]},
                    ]},
                ]},
            ]}}}
        }
        session = authed_session()
        client = WorldQuantClient("user@example.com", "secret", session=session)

        scopes = client.extract_scopes(configuration)

        self.assertIn(Scope("EQUITY", "USA", 1, "TOP3000"), scopes)
        self.assertIn(Scope("EQUITY", "USA", 0, "TOP1000"), scopes)
        self.assertEqual(len(scopes), 4)


class WorldQuantSimulationTest(unittest.TestCase):
    def test_completed_simulation_returns_metrics(self):
        session = authed_session()
        session.post.side_effect = [
            response(201, {}),  # auth in constructor
            response(201, {}, headers={
                "Location": "https://api.worldquantbrain.com/simulations/sim1"
            }),
        ]
        session.get.side_effect = [
            response(200, {"alpha": "WQ1"}, headers={"Retry-After": "0"}),
            response(200, {"is": {
                "sharpe": 1.6,
                "fitness": 1.1,
                "turnover": 0.4,
                "checks": [{"name": "X", "result": "PASS"}],
            }}),
        ]
        client = WorldQuantClient(
            "user@example.com", "secret", session=session, sleep_func=Mock()
        )

        result = client.simulate_alpha({"type": "REGULAR"})

        self.assertEqual(result.worldquant_alpha_id, "WQ1")
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.metrics["sharpe"], 1.6)
        self.assertEqual(result.checks[0]["name"], "X")

    def test_compile_error_is_classified(self):
        session = authed_session()
        session.post.side_effect = [
            response(201, {}),
            response(400, {"message": "compile failed: syntax error"}),
        ]
        client = WorldQuantClient(
            "user@example.com", "secret", session=session, sleep_func=Mock()
        )

        result = client.simulate_alpha({"type": "REGULAR"})

        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.error_code, "COMPILE_ERROR")

    def test_authorization_error_is_classified(self):
        session = authed_session()
        session.post.side_effect = [
            response(201, {}),
            response(403, {"message": "not authorized for this dataset"}),
        ]
        client = WorldQuantClient(
            "user@example.com", "secret", session=session, sleep_func=Mock()
        )

        result = client.simulate_alpha({"type": "REGULAR"})

        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.error_code, "DATASET_AUTHORIZATION_ERROR")


if __name__ == "__main__":
    unittest.main()
