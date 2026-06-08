import unittest

from console_encoding import configure_utf8_console


class FakeStream:
    def __init__(self, encoding):
        self.encoding = encoding
        self.calls = []

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)


class ConsoleEncodingTest(unittest.TestCase):
    def test_reconfigures_non_utf8_streams(self):
        stdout = FakeStream("cp1252")
        stderr = FakeStream("cp1252")

        configure_utf8_console(stdout=stdout, stderr=stderr)

        self.assertEqual(stdout.calls, [{"encoding": "utf-8"}])
        self.assertEqual(stderr.calls, [{"encoding": "utf-8"}])

    def test_leaves_utf8_streams_unchanged(self):
        stdout = FakeStream("utf-8")
        stderr = FakeStream("UTF-8")

        configure_utf8_console(stdout=stdout, stderr=stderr)

        self.assertEqual(stdout.calls, [])
        self.assertEqual(stderr.calls, [])


if __name__ == "__main__":
    unittest.main()
