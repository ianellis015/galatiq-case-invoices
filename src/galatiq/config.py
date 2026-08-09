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


# --- LLM ---------------------------------------------------------------------
#
# Base URL and model are adapter constants rather than deployment config -- they only
# live here so a one-off override does not require editing code. The minimum .env is
# still a single line: the key.
#
# The brief's `from xai import Grok` against https://grok.x.ai does not describe a
# real SDK or a real endpoint. The working integration is the OpenAI client pointed
# at the address below.
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_BASE_URL = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4.5")

# Zero by default. Extraction and validation are transcription tasks, not creative
# ones -- the same document should produce the same reading twice, and a system that
# moves money has no use for variety.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

# Transport-level retries only: a dropped connection or a rate limit. Distinct from
# the extraction retry loop, which handles a model that misread a document.
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))


def ensure_var_dir(path: Path | None = None) -> Path:
    """Create the parent directory for a database file if it does not exist.

    I call this before opening any connection. sqlite3 creates a missing database
    *file* but not the directory it lives in, and the error it raises when the
    directory is absent is far less obvious than it should be.
    """
    target = (path or DB_PATH).parent
    target.mkdir(parents=True, exist_ok=True)
    return target
