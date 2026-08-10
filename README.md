# Galatiq — Invoice Processing Pipeline

A multi-agent system that ingests invoices in mixed formats, validates them against
inventory, reasons through approval, and records payment — designed so that every
decision can be reconstructed after the fact.

**Governing principle:** the LLM handles ambiguity, deterministic code handles
correctness. The model reads messy documents into structure and writes human-legible
reasoning. It never does arithmetic, never queries stock, never evaluates a
threshold, and never decides to release a payment.

> **Status:** working end to end. Any document format in; a payment, a reasoned
> rejection, or an invoice held for a human out. Over the provided corpus: 20 documents
> in 2m24s, $203,758 of bad payments prevented. See
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

### Process invoices

```bash
uv run python main.py --invoice_path=data/invoices/invoice_1001.txt
```

One invoice: what was extracted, every finding with its evidence, and the decision with
its reasoning.

(With the virtual environment activated, `python main.py --invoice_path=…` works exactly
as the brief writes it. The `uv run` prefix just saves activating it.)

```bash
uv run python main.py --invoice_path=data/invoices/ --as-of 2026-02-01
```

A directory or a glob: processed concurrently, one line per document, then a summary.
Invoices needing a human are collected during the run and reviewed at the end — approve,
deny, or skip. Skipping leaves the run suspended, so a later session picks it back up.

**`--as-of` matters.** The provided corpus is dated January 2026, so without it every
invoice reports as months past due and the signal is worthless.

| Flag | |
|---|---|
| `--as-of DATE` | Reference date for due-date checks |
| `--no-interactive` | Skip the review queue; held invoices are recorded and the run ends |
| `--concurrency N` / `-j N` | Documents at once (default 8) |
| `--json` | Run records as JSON instead of a report |

Exit codes: `0` all approved, `1` something needs attention, `2` a failure to run.

Worth [resetting](#reset) before a demo, or invoices paid on an earlier run come back as
duplicates — which is correct, and confusing.

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

Expected: `576 passed, 27 deselected`. The suite uses temporary databases and never touches
`var/invoices.db`.

The deselected tests hit the real API. They cost money, need a key, and fail when the
network is down — none of which belongs in a suite you run on every save. Run them
deliberately:

```bash
uv run pytest -m live -v
```

### Inspect the database

```bash
sqlite3 var/invoices.db "SELECT * FROM inventory;"
sqlite3 var/invoices.db "SELECT * FROM ledger;"
```

### Reset

```bash
rm -f var/invoices.db var/audit.db && uv run python -m galatiq.store.seed
```

`var/` is gitignored and regenerated, so deleting it costs nothing.

To keep the seeded inventory but clear the payment history — the usual thing before a
demo, since re-running a batch otherwise reports already-paid invoices as duplicates:

```bash
uv run python -c "
from galatiq.store.db import connection
with connection() as c:
    c.execute('DELETE FROM ledger'); c.commit()
"
```

---

## Project layout

```
src/galatiq/
├── config.py          paths and environment resolution
├── money.py           exact money handling (Decimal ↔ integer cents)
├── models.py          the shapes passed between stages
├── dates.py           parsing what a document called a date
├── amounts.py         parsing what a document called an amount
├── mapping.py         structural hints, and cross-checking two readings
├── fx.py              currency conversion, from a static table
├── payment.py         releasing money, write-ahead and at most once
├── state.py           the shared document every graph node writes to
├── graph.py           the pipeline as a state graph
├── cli/               the command line: batching, review queue, rendering
├── loaders/           reading any document off disk, in any format
├── llm/               the typed interface every agent calls through
├── agents/            the nodes that call a model
├── checks/            the seven deterministic validation checks
├── policy/            the approval rules, as configuration
└── store/
    ├── schema.sql     inventory + ledger tables
    ├── db.py          connections and schema creation
    ├── seed.py        inventory seed data
    └── repository.py  all reads and writes
main.py                entry point — the command the brief names
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
| Extraction phase | Extractor and extraction critic, running in a LangGraph state graph with two bounded self-correction loops and durable checkpointing |
| Validation phase | Item normalisation against the catalog, then seven checks fanned out in parallel — stock, arithmetic, integrity, duplicates, dates, currency, fraud |
| Approval phase | Rules as YAML configuration, an approver that writes the reasoning, and an adversarial critic behind a second bounded reflection loop |
| Payment | Idempotent ledger write ahead of the payment call, so a crash can delay money but never duplicate it |
| CLI | `python main.py --invoice_path=…` for a file, a directory or a glob. Concurrent batching, a held-invoice review queue, and a summary in the brief's terms |

**Deliberately not built**

A web dashboard — named as the cut line from the start and cut. Partial approval: INV-1016 has two valid lines and one unknown item, and the whole invoice is rejected with reasoning rather than paying a subset.

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

**Amounts and dates are carried twice: as written, and as parsed.** A document does not
contain money or dates — it contains text that may or may not denote them. INV-1012
states a total of `$3,500.O0`, with a letter O where a zero belongs, and INV-1003 states
a due date of `"yesterday"`. Both are perfectly clear statements of values that are not
values. So `total_raw` holds what the document said, `total` holds what it turned out to
mean, and the gap between them is what a finding reports. Collapsing the two would mean
either crashing on those documents or silently rewriting `$3,500.O0` as `3500.00` — and
the second is worse, because a corrected amount is indistinguishable from one that was
always right and the correction leaves no trace.

**Quantities aggregate per item, not per line.** INV-1013 lists eight lines with items
repeating — WidgetA appears as 15, 5 and 2. Each passes on its own against a stock of 15.
Together they are 22, and all three products bust. A per-line check approves that invoice
and pays for stock that does not exist, without erroring.

**Item names are matched conservatively, and never fuzzily.**
`difflib.SequenceMatcher("WidgetC", "WidgetA")` scores about 0.857 — any cutoff loose
enough to accept `Widget A` also accepts `WidgetC`, which would turn INV-1016's unknown
item into an in-stock one and approve the payment. So differences in *formatting* are
resolved and differences in *content* never are: strip qualifiers and punctuation,
lowercase, and require an exact match on what remains. A missed match means a human
resolves it in a minute; a wrong match means money leaves against a product nobody sells,
and nothing downstream would catch it.

**Checks report; they do not decide.** A CRITICAL finding is a statement about the
invoice, not a verdict on it. Whether one means rejection is the policy engine's
question, and separating them is what lets the rules change without touching the code
that detects the facts.

**The check context is a snapshot, taken once.** Seven concurrent database reads become
one, checks become pure functions of explicit data that a test can construct in three
lines, and the snapshot travels in the checkpointed state — so the audit trail answers
*"what stock did we see when we rejected this?"* rather than *"what does stock say
now?"*. `today` is injected for the same reason: a past-due check that calls
`date.today()` passes today and fails next month.

**The model has veto power, not approval power.** The rules engine is authoritative. The
approver may reject something the rules would have paid, and may escalate something they
would have approved. It can never approve something they rejected. The whole guarantee is
one line — take the more conservative of the two outcomes — and it is tested as an
exhaustive 3×3 matrix, because it is the single thing standing between a persuasive
document and a payment.

**Size and correctness are independent axes.** Something wrong with the invoice decides
approve-versus-reject; the amount decides automatic-versus-human. A clean $100,000 invoice
is held for a VP, not refused. A $500 invoice with a stock breach is refused. The business
story is one sentence: automate the small clean invoices, escalate the large ones,
reject the broken ones with a reason.

**Rules are configuration, within a fixed vocabulary.** `policy/rules.yaml` holds the
thresholds, bands and effects, and each rule names a condition from a short menu of tested
Python predicates. A finance lead can retune the threshold or move a signal from "hold" to
"reject" without a developer; they cannot invent a new *kind* of condition, which would
mean a config file containing untested logic in the path of every payment.

**The ledger row is written before the payment call.** Pay-then-record loses the record on
a crash and double-pays on the next run. Record-then-pay leaves a record of a payment that
did not happen — recoverable by a human reconciling against the bank. Given a choice
between losing money and delaying it, delay it.

**Payment is not an agent.** It is a tool call behind a deterministic edge, reached only
when the rules and the model concur. No model is given discretion over releasing funds.

**Document text is data, never instruction.** Invoices are written by vendors, and a
vendor is not a trusted party. Instructions live in the system message and document
content in the user message — never concatenated, which is how most injection defenses
leak. Content is fenced by sentinels the document cannot forge, since occurrences inside
it are neutralised before wrapping. INV-1003's *"URGENT — Pay immediately… wire transfer
preferred"* is a fact about the invoice that belongs in `notes` for the fraud check to
score, and never an instruction to the system reading it.

**The critic has three verdicts, not two.** It answers *"did I misread the document?"*
— not *"is the document wrong?"*, which is the checks' question. INV-1009 states a
subtotal of 1000.00 while its line items sum to −250.00: the transcription is faithful
and the document is contradictory. A critic that can only say "sound" or "misread" sends
the extractor back to re-read a document it already read correctly, gets the same values
because they are the right values, and burns its budget escalating something it
understood perfectly. `DOCUMENT_INCONSISTENT` is what makes the loop terminate.

**Retry budgets live in code, not in prompts.** A prompt saying "only retry twice" is a
suggestion — untestable without spending API calls, and exactly the kind of instruction a
hostile document tries to talk past. An integer compared in a routing function is none of
those things, and the routing functions are tested by calling them with plain dicts.

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
