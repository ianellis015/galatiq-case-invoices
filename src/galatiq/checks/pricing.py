"""Unit prices: is the vendor billing what we agreed to pay?

The check that closes the last way money leaves incorrectly. Stock catches an order for
goods we do not have; arithmetic catches an invoice that does not add up. Neither notices
an invoice that is internally perfect and simply charges the wrong price.

INV-1010 is the case. It orders eight WidgetA at the catalog price of $250, and then a
fourth line -- "WidgetA (rush order)" -- at $300. Every quantity is within stock, every
line multiplies out, subtotal and tax and shipping all reconcile to the stated total. The
invoice is flawless except that $200 of it should not be there.

**An overcharge escalates rather than rejects.** That line says "rush order", and a rush
surcharge is a thing that genuinely exists. Refusing to pay would be as wrong as paying
in full; the honest answer is that a person needs to say whether it was agreed. R8 is what
turns this finding into a hold.

**Undercharges are recorded, not escalated.** A vendor billing below catalog is either
honouring a discount or has made a mistake in our favour, and neither is a reason to hold
up payment. It goes in the record because a reviewer looking at a decision should see the
whole comparison, not only the half that costs us.
"""

from decimal import Decimal

from galatiq.checks import CheckContext
from galatiq.fx import to_usd
from galatiq.models import Finding, FindingCode, Invoice, Severity
from galatiq.money import round_money, within_tolerance

# How far a converted price may drift before it counts as a different price.
#
# This band exists for exactly one reason: exchange rates. The rate we hold is a dated
# snapshot and the vendor priced on theirs, so a few percent between the two is noise
# rather than disagreement. INV-1014 bills WidgetB at EUR 475, which converts to $517.75
# against a catalog price of $500 -- 3.6% apart, and not evidence of anything.
#
# It deliberately does *not* apply to invoices already denominated in USD. There is no
# conversion there, so there is nothing to absorb, and INV-1010's rush line at 20% over
# would clear any band wide enough to be worth having.
FX_TOLERANCE = Decimal("0.05")


def check_pricing(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Compare each line's unit price against the catalog."""
    findings: list[Finding] = []

    for index, line in enumerate(invoice.line_items, start=1):
        # An unresolved item has no catalog price to compare against, and an unreadable
        # price is the integrity check's business. Both are already reported; adding a
        # third voice saying the same thing helps nobody.
        if line.item is None or line.unit_price is None:
            continue

        catalog = context.catalog_prices.get(line.item)
        if catalog is None:
            continue

        finding = _compare(
            index=index,
            line_item=line.item,
            raw_name=line.raw_name,
            quoted=line.unit_price,
            quantity=line.quantity,
            catalog=catalog,
            currency=invoice.currency,
            context=context,
        )

        if finding is not None:
            findings.append(finding)

    return findings


def _compare(
    *,
    index: int,
    line_item: str,
    raw_name: str,
    quoted: Decimal,
    quantity: int | None,
    catalog: Decimal,
    currency: str | None,
    context: CheckContext,
) -> Finding | None:
    """One line against one catalog price, in USD."""
    converted = to_usd(quoted, currency, rates=context.fx_rates)
    converted_currency = bool(currency) and currency.upper() != "USD"

    if _within_band(converted, catalog, converted_currency):
        return None

    # Rounded for display only -- the comparison above used the full precision. A
    # document writing "240.0000" should not produce a message reading "$240.0000".
    shown = round_money(converted)
    catalog_shown = round_money(catalog)
    difference = round_money(converted - catalog)
    # Named separately from the catalog name because they are often different -- the
    # normalizer resolved "WidgetA (rush order)" to WidgetA, and a reviewer checking
    # this against the paper document needs the words that are actually printed on it.
    described = f"{raw_name!r}" if raw_name != line_item else line_item

    rate_note = (
        f" ({quoted} {currency} at {context.fx_rates.get((currency or '').upper(), 1)})"
        if converted_currency
        else ""
    )

    if difference > 0:
        # The dollar impact, not just the per-unit gap. "$50 over" on a line of four is
        # $200, and the second number is the one a reviewer is deciding about.
        impact = (
            f", ${round_money(difference * quantity)} over the line" if quantity else ""
        )

        return Finding(
            code=FindingCode.PRICE_MISMATCH,
            severity=Severity.WARN,
            message=(
                f"{described} is billed at ${shown} against a catalog price of "
                f"${catalog_shown}{impact}."
            ),
            evidence=(
                f"line {index}: {line_item} quoted={shown}{rate_note} "
                f"catalog={catalog_shown} difference=+{difference} quantity={quantity}"
            ),
        )

    return Finding(
        code=FindingCode.PRICE_MISMATCH,
        severity=Severity.INFO,
        message=(
            f"{described} is billed at ${shown}, below the catalog price of "
            f"${catalog_shown}."
        ),
        evidence=(
            f"line {index}: {line_item} quoted={shown}{rate_note} "
            f"catalog={catalog_shown} difference={difference} quantity={quantity}"
        ),
    )


def _within_band(quoted: Decimal, catalog: Decimal, converted: bool) -> bool:
    """Is this close enough to be the same price?

    A cent either way on a USD invoice, and a few percent on one we had to convert.
    """
    if not converted:
        return within_tolerance(quoted, catalog)

    if catalog == 0:
        return quoted == 0

    return abs(quoted - catalog) / catalog <= FX_TOLERANCE
