import unittest

from expression_parser import ExpressionSyntaxError, fingerprint, parse_expression


class ExpressionParserTest(unittest.TestCase):
    def test_parses_calls_keywords_and_arithmetic(self):
        parsed = parse_expression(
            "group_neutralize(rank(ts_delta(close, 20)), subindustry)"
        )
        self.assertEqual(
            parsed.operator_names,
            {"group_neutralize", "rank", "ts_delta"},
        )
        self.assertEqual(parsed.identifiers, {"close", "subindustry"})

    def test_keyword_argument_name_is_not_an_identifier(self):
        parsed = parse_expression("ts_decay_linear(close, dense=true)")
        self.assertIn("close", parsed.identifiers)
        self.assertNotIn("dense", parsed.identifiers)
        self.assertIn("ts_decay_linear", parsed.operator_names)

    def test_fingerprint_normalizes_numeric_parameters_and_fields(self):
        first = fingerprint("rank(ts_delta(close, 20))", {"close"})
        second = fingerprint("rank(ts_delta(open, 10))", {"open"})
        self.assertEqual(first, second)

    def test_hash_is_whitespace_insensitive(self):
        a = parse_expression("rank(ts_delta(close, 20))")
        b = parse_expression("rank( ts_delta( close , 20 ) )")
        self.assertEqual(a.expression_hash, b.expression_hash)

    def test_invalid_expression_raises(self):
        with self.assertRaises(ExpressionSyntaxError):
            parse_expression("rank(close")


if __name__ == "__main__":
    unittest.main()
