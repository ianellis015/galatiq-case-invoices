"""Does the invoice add up?

Two equations, checked separately:

    1.  Sum of (quantity x unit price)  ==  stated subtotal
    2.  Stated subtotal + stated tax    ==  stated total

Separately, because INV-1013 satisfies the first and fails the second. Its eight lines
sum to exactly its stated subtotal of 21,040.00, and 7% tax on that is exactly its stated
1,472.80 -- then the stated total is 22,562.80 against a computed 22,512.80. Fifty
dollars, appearing only at the last line.

One combined check reports "this invoice does not add up". Two report "the line items and
the subtotal agree, and the total is fifty dollars more than it should be" -- which tells
a human where to look.

Everything is `Decimal`, compared within a one-cent tolerance. The tolerance exists to
forgive the vendor rounding tax differently than we would, not to cover our own
arithmetic, which is exact.
"""

from decimal import Decimal

from galatiq.checks import CheckContext
from galatiq.models import Finding, FindingCode, Invoice, LineItem, Severity
from galatiq.money import within_tolerance


def check_arithmetic(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Reconcile line items against subtotal, and subtotal + tax against total."""
    findings: list[Finding] = []

    findings.extend(_check_line_extensions(invoice))
    findings.extend(_check_subtotal(invoice))
    findings.extend(_check_total(invoice))

    return findings


def _check_line_extensions(invoice: Invoice) -> list[Finding]:
    """Where a line states its own amount, does it match quantity x price?

    INV-1013 and INV-1016 state both. A line that disagrees with itself is a more
    specific finding than a subtotal that is off by the same amount, and it points at
    the row rather than at the invoice.
    """
    findings: list[Finding] = []

    for index, line in enumerate(invoice.line_items, start=1):
        expected = _extension(line)
        if expected is None or line.stated_amount is None:
            continue
        if within_tolerance(expected, line.stated_amount):
            continue

        findings.append(
            Finding(
                code=FindingCode.MATH_MISMATCH,
                severity=Severity.CRITICAL,
                message=(
                    f"Line {index} ({line.raw_name}): {line.quantity} x "
                    f"{line.unit_price} is {expected}, but the line states "
                    f"{line.stated_amount}."
                ),
                evidence=f"computed {expected}, stated {line.stated_amount}",
            )
        )

    return findings


def _check_subtotal(invoice: Invoice) -> list[Finding]:
    """Sum of the lines against the stated subtotal.

    Requires every line to carry a quantity and a price. A partial sum compared against
    a full subtotal would report a discrepancy caused by the missing data rather than by
    the arithmetic -- the integrity check reports the missing data, and this check stays
    quiet rather than blaming the wrong thing.
    """
    if invoice.subtotal is None or not invoice.line_items:
        return []

    extensions = [_extension(line) for line in invoice.line_items]
    if any(value is None for value in extensions):
        return []

    computed = sum(extensions, Decimal("0"))
    if within_tolerance(computed, invoice.subtotal):
        return []

    difference = invoice.subtotal - computed

    return [
        Finding(
            code=FindingCode.MATH_MISMATCH,
            severity=Severity.CRITICAL,
            message=(
                f"Line items sum to {computed}, but the invoice states a subtotal of "
                f"{invoice.subtotal} (a difference of {difference})."
            ),
            evidence=(
                f"computed {computed}, stated {invoice.subtotal}, "
                f"difference {difference}"
            ),
        )
    ]


def _check_total(invoice: Invoice) -> list[Finding]:
    """Subtotal plus tax against the stated total.

    Tax is treated as zero when absent, which is the ordinary reading of an invoice with
    no tax line. An absent subtotal is different -- there is nothing to add to, so
    nothing to check, and INV-1003 states only a total.
    """
    if invoice.total is None or invoice.subtotal is None:
        return []

    tax = invoice.tax_amount if invoice.tax_amount is not None else Decimal("0")

    # Shipping counts toward the total but not toward the subtotal, which is the whole
    # reason it has its own field. INV-1010 states 6,700 + 335 tax + 150 shipping =
    # 7,185, and without this term the check reports a $150 discrepancy on an invoice
    # that adds up perfectly -- arithmetically correct and completely wrong.
    shipping = invoice.shipping if invoice.shipping is not None else Decimal("0")

    computed = invoice.subtotal + tax + shipping

    if within_tolerance(computed, invoice.total):
        return []

    difference = invoice.total - computed
    parts = f"{invoice.subtotal} + {tax} tax"
    if shipping:
        parts += f" + {shipping} shipping"

    return [
        Finding(
            code=FindingCode.MATH_MISMATCH,
            severity=Severity.CRITICAL,
            message=(
                f"{parts} is {computed}, but the invoice states a total of "
                f"{invoice.total} (unexplained {difference})."
            ),
            evidence=(
                f"{parts} = {computed}, stated {invoice.total}, "
                f"difference {difference}"
            ),
        )
    ]


def _extension(line: LineItem) -> Decimal | None:
    """Quantity x unit price, or None when either is missing."""
    if line.quantity is None or line.unit_price is None:
        return None
    return line.unit_price * line.quantity
