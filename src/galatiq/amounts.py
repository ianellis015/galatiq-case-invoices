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
    ("total_raw", "total", try_parse_money),
    ("tax_rate_raw", "tax_rate", try_parse_rate),
)

# ... and for each line item.
LINE_AMOUNTS: tuple[tuple[str, str, Parser], ...] = (
    ("unit_price_raw", "unit_price", try_parse_money),
    ("stated_amount_raw", "stated_amount", try_parse_money),
)


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

    CRITICAL, because an invoice whose total cannot be read is not one to pay. The
    alternative -- letting a retry quietly rewrite it as 3500.00 -- produces a payment
    nobody can trace back to what the document actually said.
    """
    findings: list[Finding] = []

    for raw_field, parsed_field, _ in INVOICE_AMOUNTS:
        raw = getattr(invoice, raw_field)
        if raw is not None and getattr(invoice, parsed_field) is None:
            findings.append(
                Finding(
                    code=FindingCode.DATA_INTEGRITY,
                    severity=Severity.CRITICAL,
                    message=f"Stated {parsed_field.replace('_', ' ')} is not a number.",
                    evidence=f"{parsed_field}: {raw!r}",
                )
            )

    for index, line in enumerate(invoice.line_items, start=1):
        for raw_field, parsed_field, _ in LINE_AMOUNTS:
            raw = getattr(line, raw_field)
            if raw is not None and getattr(line, parsed_field) is None:
                findings.append(
                    Finding(
                        code=FindingCode.DATA_INTEGRITY,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Line {index} states a "
                            f"{parsed_field.replace('_', ' ')} that is not a number."
                        ),
                        evidence=f"{line.raw_name}: {raw!r}",
                    )
                )

    return findings
