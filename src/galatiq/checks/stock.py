"""Inventory: is this item real, and do we have enough of it?

The check the whole corpus is built around, and the one a naive implementation gets
wrong in a way that produces no error.

**Quantities aggregate per item across every line.** INV-1013 lists eight lines with
items repeating: WidgetA appears as 15, 5 and 2. Each of those passes on its own against
a stock of 15. Together they are 22, and all three products bust. A per-line check
approves that invoice and pays for stock that does not exist.

**Unknown and out-of-stock are different findings.** INV-1008's "SuperGizmo" does not
exist; INV-1003's "FakeItem" exists with zero on hand. Both block payment, and they
block it for different reasons -- one is a catalog problem to raise with the vendor, the
other is a supply problem to raise with the warehouse. Collapsing them tells the reviewer
neither.
"""

from collections import Counter, defaultdict

from galatiq.checks import CheckContext
from galatiq.models import Finding, FindingCode, Invoice, Severity


def check_stock(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Unknown items, then aggregate quantities against stock on hand."""
    findings: list[Finding] = []

    findings.extend(_unknown_items(invoice, context))
    findings.extend(_aggregate_quantities(invoice, context))

    return findings


def _unknown_items(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Items the normalizer could not resolve to a catalog entry.

    The catalog defines truth for item identity: an item absent from inventory is
    unknown, not new. A system that invented catalog entries from invoice text would let
    a vendor add products to it by billing for them.

    Reported once per distinct name, however many lines carry it.
    """
    unresolved: dict[str, int] = defaultdict(int)

    for line in invoice.line_items:
        if line.item is None:
            unresolved[line.raw_name] += 1

    return [
        Finding(
            code=FindingCode.UNKNOWN_ITEM,
            severity=Severity.CRITICAL,
            message=f"{name!r} does not match any catalog item.",
            evidence=(
                f"{name!r} on {count} line(s); "
                f"catalog: {', '.join(sorted(context.stock))}"
            ),
        )
        for name, count in unresolved.items()
    ]


def _aggregate_quantities(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Total quantity per canonical item, compared against stock.

    Lines with no usable quantity are skipped rather than counted as zero -- the
    integrity check reports those, and silently treating an unreadable quantity as
    nothing would understate the order.
    """
    totals: Counter[str] = Counter()
    line_counts: Counter[str] = Counter()

    for line in invoice.line_items:
        if line.item is None or line.quantity is None:
            continue
        totals[line.item] += line.quantity
        line_counts[line.item] += 1

    findings: list[Finding] = []

    for item, requested in sorted(totals.items()):
        available = context.stock.get(item)

        if available is None:
            # Resolved to a name the snapshot does not know. Should not happen -- the
            # normalizer only assigns catalog names -- but a stock check that silently
            # skipped an item would be the wrong thing to be right about.
            findings.append(
                Finding(
                    code=FindingCode.UNKNOWN_ITEM,
                    severity=Severity.CRITICAL,
                    message=f"{item!r} is not present in inventory.",
                    evidence=f"requested {requested}",
                )
            )
            continue

        # Equal is fine. INV-1005 orders exactly 10 WidgetB against a stock of 10 --
        # a boundary, not a breach, and an off-by-one here rejects a valid invoice.
        if requested <= available:
            continue

        across = (
            f" across {line_counts[item]} lines" if line_counts[item] > 1 else ""
        )
        findings.append(
            Finding(
                code=FindingCode.STOCK_EXCEEDED,
                severity=Severity.CRITICAL,
                message=(
                    f"{item}: {requested} requested{across}, {available} in stock."
                ),
                evidence=f"{item} requested={requested} available={available}",
            )
        )

    return findings
