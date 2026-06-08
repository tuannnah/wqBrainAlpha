"""Console encoding helpers for Windows builds."""

import sys


def _is_utf8(encoding):
    if not encoding:
        return False
    return encoding.replace("-", "").replace("_", "").lower() == "utf8"


def _configure_stream(stream):
    if stream is None or _is_utf8(getattr(stream, "encoding", None)):
        return

    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return

    try:
        reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        return


def configure_utf8_console(stdout=None, stderr=None):
    """Use UTF-8 for console streams when Python defaulted to a legacy code page."""

    _configure_stream(sys.stdout if stdout is None else stdout)
    _configure_stream(sys.stderr if stderr is None else stderr)
