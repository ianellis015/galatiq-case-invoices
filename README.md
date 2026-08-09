# Galatiq — Invoice Processing Pipeline

A multi-agent system that ingests invoices in mixed formats, validates them against
inventory, reasons through approval, and records payment — designed so that every
decision can be reconstructed after the fact.

**Governing principle:** the LLM handles ambiguity, deterministic code handles
correctness. The model reads messy documents into structure and writes human-legible
reasoning. It never does arithmetic, never queries stock, never evaluates a
threshold, and never decides to release a payment.

> **Status:** early. The persistence layer is built and tested. Invoice loading,
> extraction, validation, approval, and payment are not yet implemented. See
> [Current status](#current-status).

---

## Setup

### Requirements

- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)** — manages the virtual environment and dependencies

Nothing needs to be running. There is no server, no daemon, and no database to
install — the database is a local SQLite file created by the seed script below.

No API key is required at this stage; nothing implemented so far calls an LLM.

### Install

```bash
uv sync
```

Creates `.venv/` in the project root, installs dependencies, and installs the
`galatiq` package in editable mode. Run once after cloning, and again whenever
dependencies change.

You do **not** need to activate the virtual environment — every command below uses
`uv run`, which executes inside it automatically. If you prefer activating it,
`source .venv/bin/activate` works and lets you drop the `uv run` prefix.

---

## Running

### Seed the database

```bash
uv run python -m galatiq.store.seed
```

Creates `var/invoices.db` and fills the inventory table:

```
Seeded 4 inventory items:
  FakeItem   stock=0    price=-
  GadgetX    stock=5    price=$750.00
  WidgetA    stock=15   price=$250.00
  WidgetB    stock=10   price=$500.00
```

Safe to run repeatedly — it resets stock to these starting values rather than
erroring on the second run. Reproducible batch results depend on being able to
return to a known stock position.

### Run the tests

```bash
uv run pytest
```

Expected: `84 passed`. The suite uses temporary databases and never touches
`var/invoices.db`.

### Inspect the database

```bash
sqlite3 var/invoices.db "SELECT * FROM inventory;"
sqlite3 var/invoices.db "SELECT * FROM ledger;"
```

### Reset

```bash
rm var/invoices.db && uv run python -m galatiq.store.seed
```

`var/` is gitignored and regenerated, so deleting it costs nothing.

---

## Project layout

```
src/galatiq/
├── config.py          paths and environment resolution
├── money.py           exact money handling (Decimal ↔ integer cents)
├── models.py          the shapes passed between stages
└── store/
    ├── schema.sql     inventory + ledger tables
    ├── db.py          connections and schema creation
    ├── seed.py        inventory seed data
    └── repository.py  all reads and writes
tests/                 pytest suite
data/invoices/         the provided test corpus (16 invoices, 20 files)
var/                   runtime database (gitignored, regenerated)
```

---

## Current status

**Implemented**

| Component | Notes |
|---|---|
| `inventory` table | Stock on hand plus catalog unit prices, seeded from the brief's starter data |
| `ledger` table | One row per paid invoice, with the provider and model that produced the decision |
| Re-runnable seed | `INSERT OR REPLACE`, so stock always returns to a known position |
| Payment idempotency | `UNIQUE` on `invoice_number` — re-running a batch cannot pay the same invoice twice |
| Exact money handling | `Decimal` in memory, integer cents on disk, `float` rejected at the boundary |
| Data shapes | `Invoice`, `LineItem`, `Finding`, `ApprovalDecision`, `PaymentResult` — the objects passed between stages, and the schema handed to the LLM for structured output |

**Not yet implemented**

Invoice loading (txt/pdf/json/xml/csv), extraction, validation checks, the policy
engine, approval, and payment execution.

---

## Design notes

**Money is never a float.** Floats are binary fractions, so ordinary decimal
amounts have no exact representation and arithmetic accumulates error. That matters
because validation compares computed sums against stated totals within a $0.01
tolerance, and policy thresholds ($10,000) are exact boundary comparisons. Amounts
are `Decimal` in memory and integer cents on disk, converting only through
`galatiq.money`.

**Idempotency is enforced by the database, not by calling code.** `record_payment`
attempts the insert and lets the `UNIQUE` constraint reject a duplicate, rather than
reading first and writing second — a read-then-write leaves a window where a crash or
a re-run can produce a second payment.

**Inventory is read-only by default.** Validation is a pure function of the invoice
plus seed data, so batch runs are order-independent and repeatable.

**No hand-rolled audit table.** LangGraph's SQLite checkpointer will own its own
tables in a separate database file; a second audit store alongside it would give two
sources of truth that can disagree.

**The models validate shape, not truth.** An `Invoice` has to be able to hold a
*broken* invoice faithfully — a negative quantity, an empty vendor, a due date of
`"yesterday"`, a stated total that disagrees with its own line items. The corpus
contains all of those. If the schema rejected them, extraction would crash and the
reviewer would learn nothing; the product is a reasoned rejection, not a stack trace.
Judging whether an invoice is correct is the checks' job.
