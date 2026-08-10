"""Structural hints: building an Invoice from one, and checking one against another.

Two jobs, both deterministic, both about the *hint* -- the independent reading a
structural parser produced for documents whose shape it recognised.

`hint_to_invoice` is the cheap path: a hint already has the fields named, so turning it
into an `Invoice` needs no judgement.

`compare_to_hint` is why the hint exists at all. A parser and a language model read the
same document and have to agree. When they don't -- the parse says the subtotal is
21040.00 and the model says 21400.00 -- that disagreement is a caught misread, and
neither reading alone would have revealed it. It is a stronger signal than a critic
re-examining its own work, because the two readings share no failure mode.
"""

from decimal import Decimal, InvalidOperation
from typing import Any

from galatiq.amounts import parse_amounts
from galatiq.dates import parse_invoice_dates
from galatiq.models import (
    Finding,
    FindingCode,
    Invoice,
    LineItem,
    Severity,
)
from galatiq.money import parse_money, within_tolerance

# Hint keys that map straight onto Invoice fields of the same name.
_DIRECT_FIELDS = (
    "invoice_number",
    "revision",
    "vendor",
    "vendor_address",
    "currency",
    "payment_terms",
    "notes",
)

# Money the parser read. It lands in the raw fields, exactly as the extractor's output
# does, so both readings travel the same path and the same parser turns both into
# numbers. A CSV stating "Tax (6%)" and a model transcribing "6%" produce the same
# result, which is what makes comparing the two readings meaningful.
_AMOUNT_FIELDS = ("subtotal", "tax_rate", "tax_amount", "total")

# Hint keys consumed by the mapping, so they do not also land in `extra`.
_CONSUMED = (
    frozenset(_DIRECT_FIELDS)
    | frozenset(_AMOUNT_FIELDS)
    | {"date", "due_date", "line_items"}
)


def hint_to_invoice(
    hint: dict[str, Any],
    *,
    source_path: str | None = None,
    source_format: str | None = None,
) -> Invoice:
    """Build an Invoice from a structural hint.

    Nothing is computed and nothing is corrected. Values that cannot be coerced are
    preserved in their raw form rather than dropped, because a quantity of "a dozen"
    has to reach a DATA_INTEGRITY finding that can quote it.

    Unrecognised keys land in `extra`. A vendor's PO number or department code does
    not map onto a modelled field, and discarding it would mean a human reading the
    audit trail cannot see what the system saw and ignored.
    """
    fields: dict[str, Any] = {
        name: hint[name] for name in _DIRECT_FIELDS if hint.get(name) is not None
    }

    for name in _AMOUNT_FIELDS:
        if hint.get(name) is not None:
            fields[f"{name}_raw"] = str(hint[name])

    # Dates keep the document's text; the parsed value is filled in afterwards by the
    # extractor. Keeping them separate is what lets "no due date" and "an unparseable
    # due date" stay different findings.
    if hint.get("date") is not None:
        fields["issue_date_raw"] = str(hint["date"])
    if hint.get("due_date") is not None:
        fields["due_date_raw"] = str(hint["due_date"])

    fields["line_items"] = [
        _line_item(raw) for raw in hint.get("line_items", []) if isinstance(raw, dict)
    ]

    extra = {key: value for key, value in hint.items() if key not in _CONSUMED}
    if extra:
        fields["extra"] = extra

    # Parsed immediately: this path involves no model, so there is nothing to wait
    # for and no reason to hand a caller half-converted data.
    return parse_amounts(
        parse_invoice_dates(
            Invoice(source_path=source_path, source_format=source_format, **fields)
        )
    )


def _line_item(raw: dict[str, Any]) -> LineItem:
    """One hint line item, coerced as far as it will go and no further."""
    quantity, quantity_raw = _coerce_quantity(raw.get("quantity"))

    return LineItem(
        raw_name=str(raw.get("item") or ""),
        quantity=quantity,
        quantity_raw=quantity_raw,
        unit_price_raw=_as_text(raw.get("unit_price")),
        stated_amount_raw=_as_text(raw.get("amount")),
        note=raw.get("note"),
    )


def _as_text(value: Any) -> str | None:
    """Whatever the parser read, as the text it was."""
    return None if value is None else str(value)


def _coerce_quantity(value: Any) -> tuple[int | None, str | None]:
    """Return (quantity, quantity_raw).

    The raw form is kept whenever coercion fails, so the finding that reports it can
    quote what the document actually said instead of "missing".
    """
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, str(value)
    if isinstance(value, int):
        return value, str(value)

    text = str(value).strip()
    try:
        return int(text), text
    except ValueError:
        return None, text


# ---------------------------------------------------------------------------
# Cross-checking the two readings
# ---------------------------------------------------------------------------


def compare_to_hint(invoice: Invoice, hint: dict[str, Any] | None) -> list[Finding]:
    """Findings for material disagreements between the extraction and the hint.

    Only material fields are compared. A vendor address formatted differently is not
    worth a finding; a subtotal that differs by fifty dollars is. The point is to
    catch a misread, not to demand two readings phrase themselves identically.

    WARN rather than CRITICAL: a disagreement means one of the two is wrong, and the
    system does not know which. That is information for a reviewer and a signal for
    the critic, not grounds to reject an invoice on its own.
    """
    if not hint:
        return []

    findings: list[Finding] = []

    findings.extend(_compare_invoice_number(invoice, hint))
    findings.extend(_compare_amounts(invoice, hint))
    findings.extend(_compare_line_items(invoice, hint))

    return findings


def _compare_invoice_number(invoice: Invoice, hint: dict[str, Any]) -> list[Finding]:
    stated = hint.get("invoice_number")
    if not stated or not invoice.invoice_number:
        return []

    if str(stated).strip() == invoice.invoice_number.strip():
        return []

    return [
        Finding(
            code=FindingCode.HINT_DISAGREEMENT,
            severity=Severity.WARN,
            message="Structural parse and extraction disagree on the invoice number.",
            evidence=f"parsed {stated!r}, extracted {invoice.invoice_number!r}",
        )
    ]


def _compare_amounts(invoice: Invoice, hint: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []

    for field in ("subtotal", "tax_amount", "total"):
        parsed = _as_decimal(hint.get(field))
        extracted = getattr(invoice, field) or _as_decimal(
            getattr(invoice, f"{field}_raw")
        )

        if parsed is None or extracted is None:
            continue
        if within_tolerance(parsed, extracted):
            continue

        findings.append(
            Finding(
                code=FindingCode.HINT_DISAGREEMENT,
                severity=Severity.WARN,
                message=f"Structural parse and extraction disagree on {field}.",
                evidence=f"parsed {parsed}, extracted {extracted}",
            )
        )

    return findings


def _compare_line_items(invoice: Invoice, hint: dict[str, Any]) -> list[Finding]:
    """Line count first, then quantities.

    Count is the one that matters most. INV-1006 repeats the `item` key, and a reader
    that collapses duplicates produces an invoice with one line instead of two --
    plausible, self-consistent, and wrong. The parser walks the file positionally and
    would not make that mistake, so a count mismatch is exactly the signal that
    something dropped a line.
    """
    parsed_lines = hint.get("line_items")
    if not isinstance(parsed_lines, list):
        return []

    if len(parsed_lines) != len(invoice.line_items):
        return [
            Finding(
                code=FindingCode.HINT_DISAGREEMENT,
                severity=Severity.WARN,
                message="Structural parse and extraction disagree on the number of line items.",
                evidence=(
                    f"parsed {len(parsed_lines)} line(s), "
                    f"extracted {len(invoice.line_items)}"
                ),
            )
        ]

    findings: list[Finding] = []

    for index, (parsed, extracted) in enumerate(zip(parsed_lines, invoice.line_items)):
        parsed_qty, _ = _coerce_quantity(parsed.get("quantity"))

        if parsed_qty is None or extracted.quantity is None:
            continue
        if parsed_qty == extracted.quantity:
            continue

        findings.append(
            Finding(
                code=FindingCode.HINT_DISAGREEMENT,
                severity=Severity.WARN,
                message=f"Disagreement on quantity for line {index + 1}.",
                evidence=(
                    f"{extracted.raw_name}: parsed {parsed_qty}, "
                    f"extracted {extracted.quantity}"
                ),
            )
        )

    return findings


def _as_decimal(value: Any) -> Decimal | None:
    """Best-effort Decimal, or None. Never raises -- a hint is advisory."""
    if value is None:
        return None
    try:
        return parse_money(value)
    except (TypeError, ValueError, InvalidOperation):
        return None
