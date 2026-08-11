-- ===========================================================================
-- galatiq schema
--
-- Two tables. `inventory` is the product catalog and stock position validation
-- checks against; `ledger` records money actually paid.
--
-- No audit table here on purpose: LangGraph's SQLite checkpointer creates and
-- owns its own tables in a separate database file.
-- ===========================================================================


-- ---------------------------------------------------------------------------
-- inventory -- the catalog. I treat it as truth for both item identity and price.
--
-- An item absent from this table is *unknown*, not new. That is what makes
-- invoice 1008's "SuperGizmo" and 1016's "WidgetC" rejectable rather than
-- something to be quietly accepted.
--
-- Grown from the assessment's starter DDL:
--     CREATE TABLE IF NOT EXISTS inventory (item TEXT PRIMARY KEY, stock INTEGER)
-- with two additions, both of which the brief explicitly invites:
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inventory (
    item             TEXT PRIMARY KEY,

    -- Stock on hand. CHECK (stock >= 0) means the database refuses an oversell on
    -- its own once the optional --reserve mode starts decrementing this column.
    -- Until then it's a cheap assertion that my seed data is sane.
    stock            INTEGER NOT NULL CHECK (stock >= 0),

    -- Catalog unit price in integer cents, never a float. Nullable because
    -- FakeItem exists to exercise the zero-stock path and has no listed price.
    --
    -- I added this so line prices have something to reconcile against: invoice
    -- 1014 quotes WidgetA at 225 against a catalog price of 250, and that is only
    -- detectable if the catalog price is stored somewhere.
    unit_price_cents INTEGER CHECK (unit_price_cents IS NULL OR unit_price_cents >= 0)
);


-- ---------------------------------------------------------------------------
-- ledger -- one row per invoice actually paid.
--
-- This table is my idempotency mechanism. The UNIQUE constraint on
-- invoice_number is what stops a batch re-run, or the 1004 / 1004_revised pair,
-- from paying twice. I let the check and the record be a single operation:
-- attempt the insert, and let the database reject the duplicate. Doing it as a
-- separate "have we paid this?" read followed by a write leaves a gap between the
-- two where a crash or a concurrent run can slip a second payment through.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ledger (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The idempotency key. One payment per invoice number, enforced here.
    invoice_number TEXT    NOT NULL UNIQUE,

    -- Revision marker (e.g. "R1" on invoice_1004_revised). Unused so far; I added
    -- the column now because altering a table that already has rows and test
    -- fixtures asserting against them is the annoying case. The supersede rule --
    -- a revision replaces its original if the original is unpaid, and is held for
    -- human review if it is already paid -- lands with the payment node.
    revision       TEXT,

    -- Amount in integer minor units, and the currency those units are in.
    -- Currency is stored explicitly so a EUR invoice (1014) can never be read back
    -- as though it were USD.
    amount_cents   INTEGER NOT NULL,
    currency       TEXT    NOT NULL,

    vendor         TEXT,

    -- Which model produced the decision behind this payment. An auto-selecting
    -- provider is a convenience for a demo and a hazard for an audit trail, so I
    -- make "what decided to pay this" a stored fact rather than something inferred
    -- from whose machine the batch happened to run on.
    provider       TEXT    NOT NULL,
    model          TEXT    NOT NULL,

    -- ISO-8601 UTC. SQLite has no native date type, and text in this format sorts
    -- chronologically, which is all I need from it.
    paid_at        TEXT    NOT NULL
);


-- Payment history is queried by invoice number on every run. The UNIQUE constraint
-- already creates an index that serves those lookups, so I add no other index.


-- ---------------------------------------------------------------------------
-- runs -- one row per document processed, for the web interface.
--
-- The CLI holds these in memory for the length of a batch, which is fine for a
-- terminal that prints and exits. A browser refresh would lose everything, so the
-- API writes them here as each document completes.
--
-- Deliberately additive: nothing in the pipeline reads this table, and the CLI does
-- not write to it. It is a record of what happened, not part of how anything is
-- decided.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Groups the documents of one batch, so a page can show the latest run.
    batch_id       TEXT    NOT NULL,

    source_path    TEXT    NOT NULL,
    invoice_number TEXT,
    outcome        TEXT,

    -- The whole RunRecord as JSON. Findings, the decision, the extracted invoice --
    -- shapes that already have pydantic models and would gain nothing from being
    -- shredded into columns nobody queries.
    record         TEXT    NOT NULL,

    created_at     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_batch ON runs (batch_id);
