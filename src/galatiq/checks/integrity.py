"""Is this document a complete invoice at all?

The check that runs before the interesting ones matter. An invoice with no number cannot
be paid, tracked, or deduplicated; one with an empty vendor has nobody to pay; one with a
quantity of -5 is describing something other than a purchase.

INV-1009 has all three problems at once, which is what makes it the test point here. It
is also the case that proves why the models are permissive: every one of these fields had
to be *storable* in order to be reportable, and a schema that rejected them would have
turned this check into a stack trace.

Nothing here is about arithmetic or stock. It asks only whether the fields a payment
needs are present and sane.
"""

from galatiq.checks import CheckContext
from galatiq.models import Finding, FindingCode, Invoice, Severity


def check_integrity(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Required fields, and quantities that make sense."""
    findings: list[Finding] = []

    findings.extend(_required_fields(invoice))
    findings.extend(_quantities(invoice))

    return findings


def _required_fields(invoice: Invoice) -> list[Finding]:
    findings: list[Finding] = []

    if not invoice.invoice_number:
        findings.append(
            Finding(
                code=FindingCode.DATA_INTEGRITY,
                severity=Severity.CRITICAL,
                message="No invoice number.",
                evidence=(
                    f"source: {invoice.source_path or 'unknown'}"
                ),
            )
        )

    # Empty string, not just absent. INV-1009's vendor object has a name of "", which is
    # a vendor field that was filled in with nothing -- and paying it would mean sending
    # money to a party the document does not name.
    if not (invoice.vendor or "").strip():
        findings.append(
            Finding(
                code=FindingCode.DATA_INTEGRITY,
                severity=Severity.CRITICAL,
                message="No vendor named.",
                evidence=f"vendor: {invoice.vendor!r}",
            )
        )

    if not invoice.line_items:
        findings.append(
            Finding(
                code=FindingCode.DATA_INTEGRITY,
                severity=Severity.CRITICAL,
                message="No line items.",
                evidence=f"source: {invoice.source_path or 'unknown'}",
            )
        )

    if invoice.total is None and invoice.total_raw is None:
        findings.append(
            Finding(
                code=FindingCode.DATA_INTEGRITY,
                severity=Severity.CRITICAL,
                message="No total stated.",
                evidence=f"invoice: {invoice.invoice_number or 'unnumbered'}",
            )
        )

    return findings


def _quantities(invoice: Invoice) -> list[Finding]:
    """Quantities that are absent, unreadable, or not a purchase.

    Zero and negative are both reported, and separately. A negative quantity is usually a
    credit note that arrived in the wrong pipeline; a zero is usually a formatting
    accident. Neither is something to pay, and telling them apart saves the reviewer a
    guess.
    """
    findings: list[Finding] = []

    for index, line in enumerate(invoice.line_items, start=1):
        if line.quantity is None:
            findings.append(
                Finding(
                    code=FindingCode.DATA_INTEGRITY,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Line {index} ({line.raw_name}) has no readable quantity."
                    ),
                    evidence=(
                        f"stated: {line.quantity_raw!r}"
                        if line.quantity_raw
                        else "no quantity stated"
                    ),
                )
            )
        elif line.quantity < 0:
            findings.append(
                Finding(
                    code=FindingCode.DATA_INTEGRITY,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Line {index} ({line.raw_name}) has a negative quantity "
                        f"({line.quantity}). This may be a credit note."
                    ),
                    evidence=f"{line.raw_name}: quantity={line.quantity}",
                )
            )
        elif line.quantity == 0:
            findings.append(
                Finding(
                    code=FindingCode.DATA_INTEGRITY,
                    severity=Severity.WARN,
                    message=f"Line {index} ({line.raw_name}) has a quantity of zero.",
                    evidence=f"{line.raw_name}: quantity=0",
                )
            )

    return findings
