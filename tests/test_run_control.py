import unittest
from unittest.mock import Mock

from run_control import RunControl


class RunControlTest(unittest.TestCase):
    def test_quit_sets_stop_flag_and_ignores_other_input(self):
        input_func = Mock(side_effect=["status", "quit"])
        control = RunControl(input_func=input_func)

        control.start()
        control.join(timeout=1)

        self.assertTrue(control.stop_requested())

    def test_request_stop_sets_flag(self):
        control = RunControl(input_func=Mock(side_effect=["quit"]))
        self.assertFalse(control.stop_requested())
        control.request_stop()
        self.assertTrue(control.stop_requested())

    def test_eof_is_treated_as_quit(self):
        control = RunControl(input_func=Mock(side_effect=EOFError()))
        control.start()
        control.join(timeout=1)
        self.assertTrue(control.stop_requested())


if __name__ == "__main__":
    unittest.main()
