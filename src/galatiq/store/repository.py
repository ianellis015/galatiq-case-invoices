"""Business reads and writes against the inventory and ledger tables.

Everything the rest of the system needs from the database goes through here, so no
other module builds SQL. Two reasons I structured it that way: the queries are all
reviewable in one file, and the idempotency rule cannot be got wrong by a caller,
because callers never write to `ledger` directly.
"""

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal

from galatiq.money import from_cents, to_cents


# ---------------------------------------------------------------------------
# inventory -- read-only by default
#
# Validation is a pure function of the invoice plus this seed data. Nothing in this
# section writes, which is what makes batch runs order-independent: process the 16
# invoices in any order and get the same findings. The optional --reserve mode adds a
# transactional decrement later; it is an add-on, not the default.
# ---------------------------------------------------------------------------


def get_stock(conn: sqlite3.Connection, item: str) -> int | None:
    """Stock on hand for a catalog item, or None if the item is unknown.

    The None case is meaningful rather than an error condition — it is how I detect
    an unknown item. "SuperGizmo" (invoice 1008) and "WidgetC" (invoice 1016) return
    None here, which becomes an UNKNOWN_ITEM finding. Returning 0 instead would
    conflate "we have none of these" with "this product does not exist", and those
    are two different rejection reasons a human reviewer needs told apart.
    """
    row = conn.execute(
        "SELECT stock FROM inventory WHERE item = ?", (item,)
    ).fetchone()

    return row["stock"] if row is not None else None


def get_all_stock(conn: sqlite3.Connection) -> dict[str, int]:
    """The full stock position as {item: stock}.

    The stock check reads the whole catalog once and then compares *aggregate*
    quantity per item across all line items, rather than querying per line. Invoice
    1013 is why: each of its eight lines passes on its own, and the totals bust all
    three products (22 > 15, 18 > 10, 9 > 5). A per-line check approves it.
    """
    return {
        row["item"]: row["stock"]
        for row in conn.execute("SELECT item, stock FROM inventory")
    }


def get_catalog_price(conn: sqlite3.Connection, item: str) -> Decimal | None:
    """Catalog unit price for an item, or None if unknown or unpriced.

    Line prices reconcile against this. Invoice 1014 quotes WidgetA at 225 against a
    catalog price of 250 — a discrepancy only visible because I store the catalog
    price rather than assuming it.
    """
    row = conn.execute(
        "SELECT unit_price_cents FROM inventory WHERE item = ?", (item,)
    ).fetchone()

    if row is None or row["unit_price_cents"] is None:
        return None
    return from_cents(row["unit_price_cents"])


# ---------------------------------------------------------------------------
# ledger -- payment records, written once per invoice
# ---------------------------------------------------------------------------


def is_paid(conn: sqlite3.Connection, invoice_number: str) -> bool:
    """Has this invoice number already been paid?

    For reporting, and for the supersede rule — a revision of an already-paid invoice
    is held for human review rather than auto-paid.

    Not a guard to put in front of `record_payment`. Checking and then writing leaves
    a gap between the two operations; `record_payment` is already safe on its own,
    which is the whole point of how it is written.
    """
    row = conn.execute(
        "SELECT 1 FROM ledger WHERE invoice_number = ?", (invoice_number,)
    ).fetchone()

    return row is not None


def record_payment(
    conn: sqlite3.Connection,
    *,
    invoice_number: str,
    amount: Decimal,
    currency: str,
    provider: str,
    model: str,
    vendor: str | None = None,
    revision: str | None = None,
    paid_at: str | None = None,
) -> bool:
    """Record a payment. Returns True if written, False if already paid.

    This is where idempotency actually happens. I attempt the insert and let the
    UNIQUE constraint on `invoice_number` reject a duplicate, rather than reading
    first and writing second — the read-then-write version has a window between the
    two steps where a crash or a re-run produces a second payment.

    Returning False rather than raising is deliberate. Paying an invoice twice is a
    bug, but *attempting* to is an ordinary event: it happens on any batch re-run
    after a failure, and it is what invoice 1004_revised does to invoice 1004. The
    caller wants "already handled, moving on", not an exception to catch.

    Keyword-only because the call site has several same-typed strings — vendor,
    provider, model, currency — and positional order would be easy to get silently
    wrong.
    """
    # UTC, ISO-8601, generated here rather than by the caller so every ledger row is
    # timestamped by the same clock in the same format.
    timestamp = paid_at or datetime.now(timezone.utc).isoformat()

    try:
        conn.execute(
            """
            INSERT INTO ledger (
                invoice_number, revision, amount_cents, currency,
                vendor, provider, model, paid_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invoice_number,
                revision,
                to_cents(amount),
                currency,
                vendor,
                provider,
                model,
                timestamp,
            ),
        )
        conn.commit()
        return True

    except sqlite3.IntegrityError:
        # Roll back before inspecting, so the failed statement leaves no partial
        # transaction open.
        conn.rollback()

        # Distinguish the expected collision from any other constraint failure. A
        # duplicate invoice number is the idempotent no-op this function exists to
        # provide; a NOT NULL violation is a real bug and must not be swallowed.
        if is_paid(conn, invoice_number):
            return False
        raise


def get_payment(conn: sqlite3.Connection, invoice_number: str) -> dict | None:
    """The ledger row for an invoice, with the amount converted back to Decimal.

    Reading through this rather than raw SQL keeps the cents-to-Decimal conversion in
    one place, so no caller ever handles a bare integer and mistakes it for dollars.
    """
    row = conn.execute(
        "SELECT * FROM ledger WHERE invoice_number = ?", (invoice_number,)
    ).fetchone()

    if row is None:
        return None

    record = dict(row)
    record["amount"] = from_cents(record["amount_cents"])
    return record
