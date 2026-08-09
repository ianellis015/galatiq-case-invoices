"""The shapes that move through the pipeline.

Five models: an `Invoice` made of `LineItem`s, the `Finding`s the checks raise
against it, the `ApprovalDecision` the policy engine reaches, and the
`PaymentResult` that comes back from the ledger.

These are not documentation. They do three jobs:

  1. `Invoice` is the schema I hand to the LLM for structured output, so the model's
     response is constrained to this shape rather than being free-form JSON I have to
     defend against.
  2. When a response still doesn't fit, pydantic raises — and that failure is the
     trigger for the bounded extraction retry loop.
  3. Every check has the signature `(Invoice) -> list[Finding]`. That uniformity is
     what lets the six of them fan out in parallel and have their results merged by a
     single reducer.

**The rule I follow throughout: these models validate shape, not truth.**

The instinct is to add constraints — quantities must be positive, subtotal must equal
the sum of the lines, due dates must parse. Every one of those would be a mistake
here, because the corpus is full of invoices that violate them and the whole product
is a *reasoned rejection* rather than a crash:

  * INV-1009 has quantity -5, an empty vendor, a null due date, and a stated subtotal
    of 1000.00 against line items summing to -250.00
  * INV-1003 has a due date of "yesterday" and no subtotal or tax lines at all
  * INV-1013's stated total is $50 more than its subtotal plus tax

If the models reject those, extraction dies and the reviewer learns nothing. So I
accept them, record them exactly as they arrived, and let the checks decide what is
wrong. A stack trace is not a decision.
"""

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from galatiq.money import parse_money


# ---------------------------------------------------------------------------
# Money on the model boundary
# ---------------------------------------------------------------------------
#
# Pydantic will accept a float for a Decimal field and convert it without
# complaint, which would reintroduce exactly the representation error `money.py`
# exists to prevent. Routing every money field through `parse_money` closes that
# door: it rejects floats outright and handles the messy string forms ("$9,975.00")
# that turn up in documents.


def _to_money(value: Any) -> Decimal:
    """Coerce to Decimal via `parse_money`, re-raising as ValueError.

    Pydantic converts ValueError and AssertionError from a validator into a
    ValidationError, but lets other exception types propagate untouched. Translating
    here means every bad field on a model surfaces the same way — as a
    ValidationError naming the field — instead of a TypeError from one field and a
    ValidationError from the next.
    """
    try:
        return parse_money(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _to_money_or_none(value: Any) -> Decimal | None:
    """As above, but treats absence as absence.

    None and the empty string both mean "this document did not state an amount",
    which is a real case: INV-1003 has no subtotal or tax lines whatsoever. Absent is
    not the same as zero, so it stays None rather than becoming Decimal("0").
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _to_money(value)


# Required and optional money fields. Same guarantee either way: Decimal or nothing,
# never a float.
Money = Annotated[Decimal, BeforeValidator(_to_money)]
OptionalMoney = Annotated[Decimal | None, BeforeValidator(_to_money_or_none)]

# A tax rate is not money, but it has the same float problem — 0.07 as a float is not
# 0.07 — so it goes through the same conversion.
OptionalRate = Annotated[Decimal | None, BeforeValidator(_to_money_or_none)]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(StrEnum):
    """How much a finding matters. Only CRITICAL forces a rejection on its own."""

    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class Outcome(StrEnum):
    """What the approval stage concluded.

    HELD_FOR_REVIEW is a real third state, not a softer rejection: it is the invoice
    routed to a human, and it is what the durable interrupt suspends on.
    """

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    HELD_FOR_REVIEW = "HELD_FOR_REVIEW"


class FindingCode(StrEnum):
    """Every problem the system can report.

    An enum rather than a free string. A typo'd code in one check would produce a
    finding the policy engine silently never matches — a failure with no symptom.
    This turns that into an error at import time.
    """

    # Inventory
    STOCK_EXCEEDED = "STOCK_EXCEEDED"          # aggregate qty across lines > stock
    UNKNOWN_ITEM = "UNKNOWN_ITEM"              # no catalog match after normalization

    # Arithmetic
    MATH_MISMATCH = "MATH_MISMATCH"            # lines vs subtotal, or subtotal+tax vs total
    PRICE_MISMATCH = "PRICE_MISMATCH"          # line price differs from catalog price

    # Document integrity
    DATA_INTEGRITY = "DATA_INTEGRITY"          # negative qty, empty vendor, missing field
    DOC_INCONSISTENT = "DOC_INCONSISTENT"      # read correctly, but internally contradictory
    DATE_UNPARSEABLE = "DATE_UNPARSEABLE"      # "yesterday"
    DATE_PAST_DUE = "DATE_PAST_DUE"
    TERMS_MISMATCH = "TERMS_MISMATCH"          # "Net 30" but due date == issue date

    # Duplicates
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"
    REVISION_SUPERSEDES = "REVISION_SUPERSEDES"

    # Money and thresholds
    THRESHOLD_EXCEEDED = "THRESHOLD_EXCEEDED"  # >= $10,000, needs VP scrutiny
    STRUCTURING = "STRUCTURING"                # sits just under the threshold
    NON_USD_CURRENCY = "NON_USD_CURRENCY"

    # Fraud
    FRAUD_SIGNAL = "FRAUD_SIGNAL"              # urgency language, wire request
    PROMPT_INJECTION = "PROMPT_INJECTION"      # document text attempting to instruct

    # Pipeline health
    CHECK_FAILED = "CHECK_FAILED"              # a check raised; pipeline continues
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"  # retry budget exhausted


class PaymentStatus(StrEnum):
    """What actually happened at the ledger.

    ALREADY_PAID is a success, not a failure — it is the idempotent no-op that stops
    a batch re-run, or INV-1004_revised, from paying twice.
    """

    PAID = "PAID"
    ALREADY_PAID = "ALREADY_PAID"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# The invoice
# ---------------------------------------------------------------------------


class LineItem(BaseModel):
    """One row on an invoice, as it appeared.

    `raw_name` and `item` are separate on purpose. `raw_name` is whatever the
    document actually said -- "WidgetA (rush order)", "Widget A", "SuperGizmo" -- and
    I never overwrite it, because a human tracing a rejection needs to see the bytes
    that caused it. `item` is the canonical catalog name the normalizer resolves it
    to, and stays None when nothing matches, which is what becomes an UNKNOWN_ITEM
    finding.
    """

    model_config = ConfigDict(extra="forbid")

    raw_name: str
    item: str | None = None

    # No ge=0 constraint. INV-1009 has quantity -5, and I need to store that in order
    # to report it.
    quantity: int

    unit_price: Money

    # Some documents state a per-line amount as well as a unit price (INV-1013 does).
    # I keep it as stated rather than recomputing, so check_math has both numbers to
    # compare.
    stated_amount: OptionalMoney = None

    # INV-1013 annotates lines with "Volume discount", "Expedited", "Sample".
    note: str | None = None


class Invoice(BaseModel):
    """A single invoice, exactly as the document presented it.

    Deliberately permissive. Nearly everything below the invoice number is optional,
    because the corpus contains documents missing each of these in turn, and a
    missing field has to become a finding rather than a parse error.

    Nothing here is computed. Stated amounts are stored as stated -- the arithmetic
    comparison is check_math's job, and it can only run if both the stated and the
    computable values survive extraction intact.
    """

    model_config = ConfigDict(extra="forbid")

    # --- identity ---
    invoice_number: str

    # "R1" on invoice_1004_revised. The supersede rule -- a revision replaces its
    # original if unpaid, and is held for review if already paid -- lives in the
    # payment node, not here.
    revision: str | None = None

    # --- vendor ---
    # Flat rather than nested: some documents give a vendor object with a name and
    # address (INV-1009), others a bare string (INV-1014). One shape downstream.
    # Empty string is valid and meaningful -- INV-1009's vendor name is "".
    vendor: str = ""
    vendor_address: str | None = None

    # --- dates ---
    # Raw text and parsed value both, for every date. INV-1003's due date is
    # "yesterday" and INV-1009's is null; typing these as `date` would make both
    # documents unreadable. The raw form is what check_dates reports as evidence.
    #
    # I do not parse raw into parsed here. That is the extractor's job, and leaving
    # it explicit means the model has no hidden behaviour.
    issue_date_raw: str | None = None
    issue_date: date | None = None
    due_date_raw: str | None = None
    due_date: date | None = None

    # --- content ---
    line_items: list[LineItem] = Field(default_factory=list)

    # --- money ---
    # All optional: INV-1003 states only a total, with no subtotal or tax lines.
    subtotal: OptionalMoney = None
    tax_rate: OptionalRate = None
    tax_amount: OptionalMoney = None
    total: OptionalMoney = None

    # No default currency. Assuming USD when a document is silent is a judgement
    # about the world, and judgements belong in the checks where they can be
    # reported. INV-1014 is in EUR, and a silent USD default is exactly how that
    # slips past the $10,000 threshold rule.
    currency: str | None = None

    # --- free text ---
    payment_terms: str | None = None

    # Where INV-1003's "URGENT - Pay immediately... Wire transfer preferred" lands.
    # The fraud check cannot score language the model never stored, so this field is
    # load-bearing rather than incidental.
    notes: str | None = None

    # --- provenance ---
    # Which file this came from, so a finding traces back to bytes on disk. INV-1011,
    # 1012 and 1013 each exist in two formats whose contents differ, so "which
    # invoice" is not a specific enough answer.
    source_path: str | None = None
    source_format: str | None = None


# ---------------------------------------------------------------------------
# Findings and decisions
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """One problem, with the evidence that produced it.

    `evidence` is not optional in spirit even though it has a default. A finding
    without it says "this invoice is bad"; a finding with it says "here is the line
    that is bad" -- and only the second is useful to the person deciding what to do
    about it.
    """

    model_config = ConfigDict(extra="forbid")

    code: FindingCode
    severity: Severity
    message: str
    evidence: str = ""


class ApprovalDecision(BaseModel):
    """The outcome of the approval stage, with its reasoning.

    `policy_refs` records which rules fired, by id (R1, R3, ...). Between that and
    the findings' evidence, any decision can be reconstructed after the fact without
    re-running the model -- which is the entire point of the design.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: Outcome
    rationale: str
    policy_refs: list[str] = Field(default_factory=list)

    # Bounded because a score outside 0-100 is a malformed response, not a strong
    # opinion. This is a shape constraint, not a judgement about the invoice.
    risk_score: int = Field(default=0, ge=0, le=100)


class PaymentResult(BaseModel):
    """What happened when payment was attempted.

    Carries vendor and amount because that is what the mock payment call takes, and
    provider and model because "what decided to release this money" has to be a
    stored fact rather than something inferred later.

    A NOT_ATTEMPTED result is the normal case for a rejected invoice, not an error.
    """

    model_config = ConfigDict(extra="forbid")

    invoice_number: str
    status: PaymentStatus

    amount: OptionalMoney = None
    currency: str | None = None
    vendor: str | None = None

    provider: str | None = None
    model: str | None = None

    paid_at: str | None = None
    message: str = ""
