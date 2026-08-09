"""Persistence layer: the inventory catalog and the payment ledger.

Two guarantees live in this package, and I put both in the database rather than in
calling code, so no future caller can get them wrong:

  * Inventory seeds to a known state and re-seeding is safe, which is what makes
    batch runs reproducible.
  * An invoice number appears in the ledger at most once, which is what stops a
    re-run from paying the same invoice twice.

No audit table here, on purpose. LangGraph's SQLite checkpointer owns its own tables
in a separate database file, and a hand-rolled audit store alongside it would give me
two sources of truth that can disagree.
"""

from galatiq.store.db import connect, connection, init_db
from galatiq.store.repository import (
    get_all_stock,
    get_catalog_price,
    get_payment,
    get_stock,
    is_paid,
    record_payment,
)

# `seed_inventory` stays out of this list deliberately. It's a script entry point run
# as `python -m galatiq.store.seed`, and importing it eagerly here makes that command
# load the module twice — which Python warns about, and which would run any
# module-level code twice.
__all__ = [
    "connect",
    "connection",
    "init_db",
    "get_stock",
    "get_all_stock",
    "get_catalog_price",
    "is_paid",
    "record_payment",
    "get_payment",
]
