"""The deterministic validation checks.

Nothing in this package calls a model. That is the load-bearing half of the governing
principle: the LLM reads a messy document into structure, and code decides whether the
structure is acceptable. Arithmetic, stock levels and thresholds are things with right
answers, and a right answer deserves a test rather than a prompt.

Every check has the same shape:

    (Invoice, CheckContext) -> list[Finding]

Uniformity is what makes the fan-out possible -- seven nodes with one signature, wired
by a loop and merged by one reducer. Seven functions each taking something different
would need seven pieces of glue.

Checks report; they do not decide. A CRITICAL finding is a statement about the invoice,
not a verdict on it. Whether one means rejection is the policy engine's question, and
keeping that separate is what lets the rules change without touching the code that
detects the facts.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Iterable

from galatiq.fx import RATES
from galatiq.models import Finding, FindingCode, Invoice, Severity


@dataclass(frozen=True)
class CheckContext:
    """Everything the checks are allowed to know, captured once.

    A snapshot rather than a live database handle, for three reasons. Checks become
    pure functions of explicit data, so a test constructs one in three lines instead of
    seeding a database. Seven concurrent reads become one. And the snapshot travels in
    the checkpointed state, so the audit trail can answer "what stock did we see when we
    rejected this?" rather than "what does stock say now, months later".

    `today` is injected for the same reason. A check calling `date.today()` passes today
    and fails next month, and a past-due rule that cannot be tested is a past-due rule
    nobody trusts.
    """

    stock: dict[str, int]
    catalog_prices: dict[str, Decimal]
    paid_invoices: dict[str, str | None]   # invoice number -> revision, if any
    today: date
    fx_rates: dict[str, Decimal]

    def to_snapshot(self) -> dict[str, Any]:
        """Plain data, for the state bus.

        Serialised as primitives rather than as a dataclass because the checkpointer
        writes state to SQLite, and depending on how it handles custom types is a
        dependency I would rather not have.
        """
        return {
            "stock": dict(self.stock),
            "catalog_prices": {k: str(v) for k, v in self.catalog_prices.items()},
            "paid_invoices": dict(self.paid_invoices),
            "today": self.today.isoformat(),
            "fx_rates": {k: str(v) for k, v in self.fx_rates.items()},
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "CheckContext":
        return cls(
            stock=dict(data.get("stock", {})),
            catalog_prices={
                k: Decimal(v) for k, v in data.get("catalog_prices", {}).items()
            },
            paid_invoices=dict(data.get("paid_invoices", {})),
            today=date.fromisoformat(data["today"]),
            fx_rates={k: Decimal(v) for k, v in data.get("fx_rates", {}).items()},
        )


def snapshot(conn: Any, *, today: date | None = None) -> CheckContext:
    """Read everything the checks need from the database, once.

    Deliberately read-only. Validation stays a pure function of the invoice plus this
    snapshot, which is what makes batch runs order-independent: process the corpus in
    any order and get the same findings.
    """
    from galatiq.store.repository import get_all_stock

    stock = get_all_stock(conn)

    prices: dict[str, Decimal] = {}
    for row in conn.execute(
        "SELECT item, unit_price_cents FROM inventory WHERE unit_price_cents IS NOT NULL"
    ):
        prices[row["item"]] = Decimal(row["unit_price_cents"]) / 100

    paid = {
        row["invoice_number"]: row["revision"]
        for row in conn.execute("SELECT invoice_number, revision FROM ledger")
    }

    return CheckContext(
        stock=stock,
        catalog_prices=prices,
        paid_invoices=paid,
        today=today or date.today(),
        fx_rates=dict(RATES),
    )


Check = Callable[[Invoice, CheckContext], list[Finding]]


def run_check(name: str, check: Check, invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Run one check behind an exception boundary.

    A bug in one check must not take down the other six or fail the batch. The failure
    becomes a finding like any other, so it reaches the same reviewer through the same
    channel -- and an invoice whose stock check crashed is one a human should look at,
    not one that quietly skipped a step.
    """
    try:
        return check(invoice, context)
    except Exception as exc:  # noqa: BLE001 - the boundary is the point
        return [
            Finding(
                code=FindingCode.CHECK_FAILED,
                severity=Severity.CRITICAL,
                message=f"The {name} check failed to run.",
                evidence=f"{type(exc).__name__}: {exc}",
            )
        ]


def all_checks() -> dict[str, Check]:
    """The registry the graph fans out over.

    Imported lazily because each check module imports `CheckContext` from here, and a
    top-level import in both directions is a cycle. The registry is also the single
    place a new check is added -- the graph wiring is a loop over this dict, so nothing
    else has to change.
    """
    from galatiq.checks.arithmetic import check_arithmetic
    from galatiq.checks.currency import check_currency
    from galatiq.checks.dates import check_dates
    from galatiq.checks.duplicates import check_duplicates
    from galatiq.checks.fraud import check_fraud
    from galatiq.checks.integrity import check_integrity
    from galatiq.checks.stock import check_stock

    return {
        "check_stock": check_stock,
        "check_arithmetic": check_arithmetic,
        "check_integrity": check_integrity,
        "check_duplicates": check_duplicates,
        "check_dates": check_dates,
        "check_currency": check_currency,
        "check_fraud": check_fraud,
    }


def run_all(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Run every check sequentially. For tests and for callers without a graph."""
    findings: list[Finding] = []
    for name, check in all_checks().items():
        findings.extend(run_check(name, check, invoice, context))
    return merge_findings(findings)


def merge_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Deduplicate and order by severity.

    Two checks can legitimately notice the same thing -- normalisation and the stock
    check both care about an unrecognised item -- and an audit trail that says the same
    thing twice reads as two problems.

    Sorted CRITICAL first because that is the order a human reads them in, and stable
    within a severity so the output of two identical runs is byte-identical.
    """
    order = {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.INFO: 2}
    seen: set[tuple[str, str, str]] = set()
    unique: list[Finding] = []

    for finding in findings:
        key = (finding.code, finding.message, finding.evidence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)

    return sorted(unique, key=lambda f: order[f.severity])
