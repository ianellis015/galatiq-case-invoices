"""Turning the amounts a document stated into numbers, or reporting that they aren't.

The money counterpart of `dates.py`, and it exists for the same reason. A document
does not contain money; it contains text that may or may not denote money. INV-1012
states a total of "$3,500.O0" -- a letter O where a zero belongs -- and that is a
perfectly clear statement of an amount that is not a number.

So every amount is carried twice: `total_raw` holds what the document said, `total`
holds what it turned out to mean, and the gap between them is what a finding reports.
Collapsing the two would mean either crashing on INV-1012 or silently rewriting it as
3500.00, and the second is worse -- a corrected amount is indistinguishable from one
that was always right, and the correction leaves no trace.

Used by both readings of a document. The extractor's transcription and a structural
parser's output go through the same function, which is what makes comparing them
meaningful.
"""

from typing import Callable

from galatiq.models import Finding, FindingCode, Invoice, LineItem, Severity
from galatiq.money import try_parse_money, try_parse_rate

Parser = Callable[[object], object]

# (raw field, parsed field, parser) for the invoice itself.
INVOICE_AMOUNTS: tuple[tuple[str, str, Parser], ...] = (
    ("subtotal_raw", "subtotal", try_parse_money),
    ("tax_amount_raw", "tax_amount", try_parse_money),
    ("shipping_raw", "shipping", try_parse_money),
    ("total_raw", "total", try_parse_money),
    ("tax_rate_raw", "tax_rate", try_parse_rate),
)

# ... and for each line item.
LINE_AMOUNTS: tuple[tuple[str, str, Parser], ...] = (
    ("unit_price_raw", "unit_price", try_parse_money),
    ("stated_amount_raw", "stated_amount", try_parse_money),
)


def parse_quantities(invoice: Invoice) -> Invoice:
    """Fill `quantity` from `quantity_raw` where the model left it null.

    The same raw/parsed split as amounts and dates, and it was missing. A model told
    repeatedly to put values in `*_raw` fields will sometimes generalise and do it for
    quantity too -- reasonably, since that is the pattern it was shown -- and without
    this the quantity is simply lost. The invoice then reports "no readable quantity"
    for a line that stated one perfectly clearly.

    Found by running the corpus: INV-1006, a documented-clean invoice, was rejected for
    a missing quantity that was present in the document and present in `quantity_raw`.

    A raw value that is not an integer stays unparsed, which is the point -- "a dozen"
    should reach a DATA_INTEGRITY finding that can quote it, not be guessed at.
    """
    lines = [_parse_quantity(line) for line in invoice.line_items]

    if lines == invoice.line_items:
        return invoice
    return invoice.model_copy(update={"line_items": lines})


def _parse_quantity(line: LineItem) -> LineItem:
    if line.quantity is not None or not line.quantity_raw:
        return line

    try:
        return line.model_copy(update={"quantity": int(line.quantity_raw.strip())})
    except ValueError:
        return line


def parse_amounts(invoice: Invoice) -> Invoice:
    """Fill the parsed money fields from the raw ones.

    An already-populated parsed field is left alone. Re-deriving a value that is
    already a Decimal would be work with no answer to add, and it lets a caller supply
    a parsed amount directly when it has one.
    """
    updates: dict[str, object] = {}

    for raw_field, parsed_field, parse in INVOICE_AMOUNTS:
        raw = getattr(invoice, raw_field)
        if raw is not None and getattr(invoice, parsed_field) is None:
            updates[parsed_field] = parse(raw)

    lines = [_parse_line(line) for line in invoice.line_items]
    if lines != invoice.line_items:
        updates["line_items"] = lines

    return invoice.model_copy(update=updates) if updates else invoice


def _parse_line(line: LineItem) -> LineItem:
    updates = {
        parsed_field: parse(getattr(line, raw_field))
        for raw_field, parsed_field, parse in LINE_AMOUNTS
        if getattr(line, raw_field) is not None
        and getattr(line, parsed_field) is None
    }
    return line.model_copy(update=updates) if updates else line


def unparsed_amount_findings(invoice: Invoice) -> list[Finding]:
    """Report amounts the document stated that are not numbers.

    Where INV-1012's OCR damage becomes visible. The model transcribed "$3,500.O0"
    faithfully, the parser could not turn it into a number, and this says so with the
    original text attached.

    **Severity depends on which amount it is, and the distinction is the point.**

    An unreadable *total* is CRITICAL: you cannot pay an amount you cannot read, and
    letting a retry quietly rewrite it as 3500.00 would produce a payment nobody could
    trace back to the document.

    Everything else is a WARN. INV-1012 is the case that taught me the difference -- its
    total reads perfectly at $9,975.00, and it is line 2's extended amount that has the
    letter O. Rejecting an entire invoice because one line's arithmetic is mistyped,
    when the quantity, the unit price and the total are all legible, is out of
    proportion to the problem. It is worth a human's glance, not a refusal.
    """
    findings: list[Finding] = []

    for raw_field, parsed_field, _ in INVOICE_AMOUNTS:
        raw = getattr(invoice, raw_field)
        if raw is None or getattr(invoice, parsed_field) is not None:
            continue

        findings.append(
            Finding(
                code=FindingCode.DATA_INTEGRITY,
                # Only the total blocks payment on its own. A subtotal or tax line that
                # cannot be read silences the arithmetic check, which is a loss of
                # verification rather than evidence of a problem.
                severity=(
                    Severity.CRITICAL if parsed_field == "total" else Severity.WARN
                ),
                message=f"Stated {parsed_field.replace('_', ' ')} is not a number.",
                evidence=f"{parsed_field}: {raw!r}",
            )
        )

    for index, line in enumerate(invoice.line_items, start=1):
        for raw_field, parsed_field, _ in LINE_AMOUNTS:
            raw = getattr(line, raw_field)
            if raw is None or getattr(line, parsed_field) is not None:
                continue

            findings.append(
                Finding(
                    code=FindingCode.DATA_INTEGRITY,
                    severity=Severity.WARN,
                    message=(
                        f"Line {index} states a "
                        f"{parsed_field.replace('_', ' ')} that is not a number."
                    ),
                    evidence=f"{line.raw_name}: {raw!r}",
                )
            )

    return findings
