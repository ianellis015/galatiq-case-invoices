"""Database connections and schema creation.

Thin on purpose. This module knows how to open a connection and apply the schema; it
knows nothing about invoices. Business reads and writes live in `repository.py`, so
there is exactly one file to open when asking what SQL this system actually runs.
"""

import sqlite3
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator

from galatiq.config import DB_PATH, ensure_var_dir


def _schema_sql() -> str:
    """Read schema.sql, which ships alongside this module inside the package.

    I load it through importlib.resources rather than a path relative to __file__ so
    it resolves whether the package is running from the source tree or an installed
    wheel.
    """
    return resources.files("galatiq.store").joinpath("schema.sql").read_text()


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection to the inventory/ledger database.

    Defaults to the configured path and creates the containing directory first.
    sqlite3 creates missing database *files* but not missing directories, and the
    error when the directory is absent is unhelpfully opaque.

    Tests pass an explicit path so they never touch the real database.
    """
    path = Path(db_path) if db_path is not None else DB_PATH
    ensure_var_dir(path)

    conn = sqlite3.connect(path)

    # Rows addressable by column name rather than position. Positional access
    # silently returns the wrong value the moment a column is added to the middle of
    # a SELECT, and I would rather not debug that later.
    conn.row_factory = sqlite3.Row

    # Off by default in SQLite for backwards compatibility. No foreign keys exist in
    # the schema yet; turning it on now means any I add later are enforced from the
    # start instead of being quietly decorative.
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the tables if they do not already exist.

    Safe on every startup: every statement in schema.sql is written with
    IF NOT EXISTS, so this is a no-op against an existing database and never
    destroys data.
    """
    conn.executescript(_schema_sql())
    conn.commit()


@contextmanager
def connection(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection, ensure the schema exists, and close it afterwards.

    The entry point for callers that just want to do one thing:

        with connection() as conn:
            stock = get_stock(conn, "WidgetA")

    Closing sits in a finally block so a raised exception still releases the file
    handle. Note I am not using sqlite3's own context manager support here — its
    __exit__ commits or rolls back the *transaction* and leaves the connection open,
    which would leak the handle while looking like it was handled.

    This deliberately does not commit. Writes are committed by the functions that
    make them, because a half-written payment should not be committed by a context
    manager tidying up on the way out.
    """
    conn = connect(db_path)
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()
