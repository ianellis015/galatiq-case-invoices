"""Runtime configuration: where things live on disk, and what came from the env.

I gave everything here a working default, because the system has to run end-to-end
on a machine with no `.env` file and no API key. Configuration is override, never
requirement — the moment a module raises on a missing environment variable at import
time, the keyless path is broken and I won't find out until someone else clones the
repo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# src/galatiq/config.py -> src/galatiq -> src -> <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# `override=False` so a variable already set in the real environment beats the file.
# That ordering matters for CI, and for passing a one-off value on the command line
# without editing .env first.
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Runtime state I regenerate rather than commit: the inventory/ledger database, and
# later the LangGraph checkpointer's audit database. Gitignored.
VAR_DIR = PROJECT_ROOT / "var"

# Overridable so the tests can point at a temp file and a demo can run against a
# throwaway database.
DB_PATH = Path(os.environ.get("INVOICE_DB_PATH", VAR_DIR / "invoices.db"))


def ensure_var_dir(path: Path | None = None) -> Path:
    """Create the parent directory for a database file if it does not exist.

    I call this before opening any connection. sqlite3 creates a missing database
    *file* but not the directory it lives in, and the error it raises when the
    directory is absent is far less obvious than it should be.
    """
    target = (path or DB_PATH).parent
    target.mkdir(parents=True, exist_ok=True)
    return target
