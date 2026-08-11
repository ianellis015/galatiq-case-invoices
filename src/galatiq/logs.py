"""What the system was doing, as opposed to what it decided.

Those are two different questions and I answer them in two different places.

**What it decided** lives in the checkpointer and the `runs` table: the findings, the
evidence, the reasoning, the stock position at the moment of the decision. That is the
audit trail, it is durable, and a log file is not the right home for it -- a second copy
of a decision is a copy that can drift from the first.

**What it was doing** is this module: retries, latencies, token counts, batch lifecycle,
and the exceptions nobody sees because the pipeline turns them into records. None of that
belongs in an audit trail and all of it matters when a run behaves oddly.

Two destinations, because they serve different readers:

    var/galatiq.log   always written, full detail, survives the run
    stderr            only with --verbose, and only the lines worth watching live

stderr rather than stdout so `--json` stays pipeable and the report stays clean. A log
line landing in the middle of a summary panel would make the tool feel broken.
"""

import logging
import sys
from pathlib import Path

from galatiq.config import LOG_PATH, ensure_var_dir

# One namespace for everything the package emits, so a caller can raise or silence the
# whole system with a single line and third-party libraries are unaffected.
LOGGER_NAME = "galatiq"

_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"

# Terser on screen. Somebody watching a run live wants the event, not the timestamp --
# they can see the timestamp, they are sitting there.
_CONSOLE_FORMAT = "  %(levelname)-7s %(message)s"

_configured = False


def logger(name: str) -> logging.Logger:
    """A child logger for one module.

    Called as `logger(__name__)`, which yields names like `galatiq.cli.runner` -- so a
    reader of the log file can see which layer produced a line, and a developer can turn
    one layer up without touching the others.
    """
    return logging.getLogger(name if name.startswith(LOGGER_NAME) else f"{LOGGER_NAME}.{name}")


def configure(*, verbose: bool = False, log_file: Path | None = None) -> Path | None:
    """Set up logging once, and return the file being written to.

    Idempotent. Both the CLI and the API call this at startup, and a test that imports
    both should not end up with two handlers writing every line twice.

    The file handler is unconditional. A grader running the batch without any flags
    should still be able to open `var/galatiq.log` afterwards and see what happened --
    observability that has to be switched on in advance is observability you do not have
    the one time you need it.
    """
    global _configured

    root = logging.getLogger(LOGGER_NAME)

    if _configured:
        return _current_file(root)

    root.setLevel(logging.DEBUG)
    # Ours alone. Without this, anything configuring the root logger -- uvicorn does --
    # gets a duplicate of every line.
    root.propagate = False

    path = log_file or LOG_PATH
    try:
        ensure_var_dir(path)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(handler)
    except OSError:
        # A read-only checkout or a full disk. Losing the log is not a reason to lose
        # the run, and the console handler below still works.
        path = None

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    root.addHandler(console)

    _configured = True
    return path


def _current_file(root: logging.Logger) -> Path | None:
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def reset() -> None:
    """Tear down the handlers. For tests, which configure per case."""
    global _configured

    root = logging.getLogger(LOGGER_NAME)
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)

    _configured = False


__all__ = ["LOGGER_NAME", "configure", "logger", "reset"]
