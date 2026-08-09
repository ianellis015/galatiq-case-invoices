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

### API key

Create a `.env` in the project root:

```bash
XAI_API_KEY=xai-...
```

That is the whole file. The endpoint (`https://api.x.ai/v1`) and model (`grok-4.5`)
are adapter defaults, overridable via `XAI_BASE_URL` and `XAI_MODEL` if needed.

The test suite needs no key and makes no network calls — it injects a test double in
place of the API client.

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

Expected: `238 passed`. The suite uses temporary databases and never touches
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
├── loaders/           reading any document off disk, in any format
├── llm/               the typed interface every agent calls through
└── store/
    ├── schema.sql     inventory + ledger tables
    ├── db.py          connections and schema creation
    ├── seed.py        inventory seed data
    └── repository.py  all reads and writes
tests/                 pytest suite
data/invoices/         the provided test corpus (16 invoices, 20 files)
data/adversarial/      my own fixtures, in formats the corpus does not contain
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
| Format loaders | Any document that decodes to text, whatever its extension. Dedicated structural parsers for txt, pdf, json, xml and csv; everything else falls back to text. Directory and glob discovery for batch mode |
| LLM provider layer | One typed interface every agent calls through — a pydantic model in, a validated instance out. Grok via the OpenAI-compatible endpoint, with strict structured output and bounded transport retries |

**Not yet implemented**

Extraction, validation checks, the policy engine, approval, and payment execution.

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

**No input crashes the pipeline. Every input produces a decision with reasoning.**
This is the invariant the ingestion layer is built to. Nothing in `loaders/` raises for
a content problem — an unreadable file, an unknown extension, malformed JSON — because
a stack trace tells the person running a batch of twenty invoices nothing, while a
rejection naming the file and the reason tells them everything. Load-time problems come
back as `Finding`s on the document and travel the same reporting channel as every
validation finding.

**Text is the universal interface.** Every document produces `raw_text`, whatever its
format, and the extractor reads text. That is what makes the system indifferent to
layout: an invoice in a shape nobody anticipated is still text. It costs nothing to
guarantee, because JSON, XML and CSV files already *are* text.

**Structural parsing is a cross-check, not a bypass.** When a parser recognises a shape
it produces a `structural_hint` — a second, independent reading of the same document.
The extractor gets it as context and the critic gets something to disagree with: when a
deterministic parse says the subtotal is 21040.00 and the model says 21400.00, that
disagreement is a caught misparse. When no parser recognises the shape, the hint is
simply absent and extraction proceeds from text — so degradation needs no detection
logic and no branch that can be wrong.

**Loaders report, they do not repair.** OCR damage (`2O26`, `$3,500.O0`), typo'd field
labels (`INVOCE`, `Vndr`), and inconsistent invoice numbers (`INV 1012` against
`INV-1012` everywhere else) all pass through untouched. Deciding what those were meant
to say needs the whole document in view — that is the extractor's judgement, and one
the critic can question. A loader that quietly fixed them would destroy the evidence
and silently rewrite a document the system moves money against.

**Model output is typed or it is an error.** Every agent calls one method: a
conversation plus a pydantic model in, a validated instance of that model out. No
agent anywhere parses free-form text out of a response. The schema is sent with
`strict: true`, so the provider enforces the shape rather than suggesting it, and
anything that still fails validation raises an error carrying the specific field
message — which is what makes a retry a correction rather than a re-roll.

**The LLM layer takes no LangChain or LangGraph dependency.** Agents are plain
functions that can be called directly or from a graph node. Keeping the raw client
means the structured-output path and its validation errors stay ours to shape, and
those errors are what the self-correction loops attach to.

**Amounts are transcribed, not interpreted.** Money fields are strings in the schema
handed to the model, so `$3,500.O0` comes back as those characters and the OCR damage
stays visible. A numeric field would invite the model to decide what the number
probably was, and would reintroduce float error at the one boundary the rest of the
system is built to keep it out of.

**A parser that knows two shapes says so rather than guessing at a third.** A CSV whose
columns are all named differently parses mechanically into a tidy dict containing no
invoice number and no line items. Passing that on as a hint would be worse than passing
nothing — it looks like a reading of the document, and the downstream mismatch gets
misdiagnosed as an arithmetic problem rather than a parse failure. A hint has to clear a
bar to be offered at all.

---

## Assumptions

Recorded here rather than buried in code.

**The structural parsers are fitted to the provided corpus.** The XML parser assumes
INV-1014's schema; the CSV parser knows two layouts. A UBL invoice or an unfamiliar CSV
dialect produces no hint and takes the text route with an INFO note. Generalising the
parsers would be building for invoices that do not exist — the model is the
generalisation mechanism.

**Inventory is read-only.** Validation is a pure function of the invoice plus seed data,
so batch runs are order-independent and repeatable. Stock is never decremented, so
processing the same batch twice gives the same findings both times.

**A directory of the corpus yields 20 documents, not 16.** INV-1011, 1012 and 1013 each
exist as a pair whose contents genuinely differ, so both members are real documents and
both are processed. The ledger's `UNIQUE` constraint means a pair can only ever produce
one payment.

**Partial approval is out of scope.** INV-1016 has two valid lines and one unknown item;
the system rejects the whole invoice with reasoning rather than paying a subset.

**A revision supersedes its original** if the original is unpaid. If it is already paid,
the revision is held for human review rather than auto-paid.

**The catalog defines truth for item identity.** An item absent from inventory is
unknown, not new.
