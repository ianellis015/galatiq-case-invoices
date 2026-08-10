"""Have we paid this before?

The ledger's UNIQUE constraint already makes double payment impossible -- the insert
fails and the second attempt is a no-op. This check exists so the *reason* reaches a
human before payment is attempted, rather than showing up as a silently skipped invoice
at the end of a batch.

Two situations that look identical if you only compare invoice numbers:

**A straight duplicate.** The same invoice submitted twice. Nothing to do, and worth
saying so.

**A revision.** INV-1004_revised carries the same number as INV-1004 with a revision
marker and a different total. If the original has not been paid, the revision supersedes
it. If it has, the revision is held for a human -- because superseding an invoice whose
money already left is a decision with consequences, and not one to make automatically.
"""

from galatiq.checks import CheckContext
from galatiq.models import Finding, FindingCode, Invoice, Severity


def check_duplicates(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Compare this invoice number against the payment ledger."""
    number = invoice.invoice_number
    if not number:
        # The integrity check reports a missing number. Nothing to look up.
        return []

    if number not in context.paid_invoices:
        return []

    paid_revision = context.paid_invoices[number]

    if invoice.revision:
        return [
            Finding(
                code=FindingCode.REVISION_SUPERSEDES,
                severity=Severity.WARN,
                message=(
                    f"{number} revision {invoice.revision} supersedes an invoice that "
                    "has already been paid. Superseding a completed payment needs a "
                    "human decision."
                ),
                evidence=(
                    f"{number}: paid revision {paid_revision or 'original'}, "
                    f"presented revision {invoice.revision}"
                ),
            )
        ]

    return [
        Finding(
            code=FindingCode.DUPLICATE_INVOICE,
            severity=Severity.CRITICAL,
            message=f"{number} has already been paid.",
            evidence=(
                f"{number}: ledger holds "
                f"{'revision ' + paid_revision if paid_revision else 'the original'}"
            ),
        )
    ]
