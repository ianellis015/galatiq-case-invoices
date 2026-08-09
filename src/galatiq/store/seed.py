"""Seed the inventory catalog to a known state.

Built on the starter code supplied with the assessment:

    import sqlite3
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS inventory (item TEXT PRIMARY KEY, stock INTEGER)')
    cursor.execute(\"\"\"
        INSERT INTO inventory VALUES
        ('WidgetA', 15), ('WidgetB', 10), ('GadgetX', 5), ('FakeItem', 0)
    \"\"\")
    conn.commit()

Same four items, same quantities, same connect/cursor/execute/commit shape. I made
four changes:

  1. INSERT OR REPLACE instead of INSERT. The original raises IntegrityError on a
     second run, because the PRIMARY KEY on `item` rejects the duplicate. Validation
     is only reproducible if stock always starts from the same position, so
     re-seeding has to be something I can do freely.
  2. Parameterised values via executemany rather than literals inlined in SQL.
     Nothing here is user input today, but I would rather not form the habit of
     building SQL by string concatenation in a system that will later handle
     vendor-supplied text.
  3. Catalog unit prices alongside stock. The brief invites extending the seed to
     support richer validation, and line-item prices need something to reconcile
     against.
  4. CREATE TABLE moved to schema.sql, so the catalog and the ledger are defined
     together in one readable place.
"""

import sqlite3
from decimal import Decimal

from galatiq.money import to_cents
from galatiq.store.db import connect, init_db

# The catalog: item, stock on hand, catalog unit price.
#
# Stock figures are the assessment's, unchanged. Prices are the catalog prices the
# invoices are written against. FakeItem has stock 0 and no price — it exists to
# exercise the zero-stock rejection path (invoice 1003 orders 100 of them), not to be
# a real product.
SEED_INVENTORY: list[tuple[str, int, Decimal | None]] = [
    ("WidgetA", 15, Decimal("250.00")),
    ("WidgetB", 10, Decimal("500.00")),
    ("GadgetX", 5, Decimal("750.00")),
    ("FakeItem", 0, None),
]


def seed_inventory(conn: sqlite3.Connection) -> int:
    """Reset the inventory table to the known seed state. Returns rows written.

    INSERT OR REPLACE gives *reset* semantics rather than *preserve* semantics: an
    item already present is overwritten back to its seed values. That is what the
    test corpus needs — every batch run starts from WidgetA 15 / WidgetB 10 /
    GadgetX 5 / FakeItem 0, or comparing two runs means nothing.

    I deliberately leave `ledger` alone. Resetting stock and erasing payment history
    are different operations, and quietly doing the second while asked for the first
    would destroy the idempotency record.
    """
    cursor = conn.cursor()

    # Prices convert to integer cents at the boundary: Decimal in the seed data
    # above, integers on disk. A NULL price stays NULL rather than becoming 0, since
    # "no listed price" and "free" are different facts.
    rows = [
        (item, stock, to_cents(price) if price is not None else None)
        for item, stock, price in SEED_INVENTORY
    ]

    cursor.executemany(
        """
        INSERT OR REPLACE INTO inventory (item, stock, unit_price_cents)
        VALUES (?, ?, ?)
        """,
        rows,
    )
    conn.commit()

    return len(rows)


def main() -> None:
    """Create the schema and seed it:

        uv run python -m galatiq.store.seed

    Idempotent end to end — run it as many times as you like.
    """
    conn = connect()
    try:
        init_db(conn)
        count = seed_inventory(conn)

        print(f"Seeded {count} inventory items:")
        for row in conn.execute(
            "SELECT item, stock, unit_price_cents FROM inventory ORDER BY item"
        ):
            price = (
                f"${row['unit_price_cents'] / 100:,.2f}"
                if row["unit_price_cents"] is not None
                else "-"
            )
            print(f"  {row['item']:<10} stock={row['stock']:<4} price={price}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
