"""Releasing money, and the ordering that makes it safe to retry.

Deliberately not an agent. No model is given discretion over payment -- this is a tool
call behind a deterministic edge, reached only when both the rules engine and the
approver concurred.

**The ledger row is written before the payment is made.** That ordering is the entire
safety property, and it is worth being explicit about why, because the natural instinct
is the opposite:

    pay first, then record   ->  a crash between the two loses the record, and the next
                                 run pays again
    record first, then pay   ->  a crash between the two leaves a record of a payment
                                 that did not happen

The second failure is recoverable: a human reconciles against the bank and re-issues one
payment. The first quietly double-pays a vendor and nobody notices until they do. Given
a choice between losing money and delaying it, delay it.

The `UNIQUE` constraint on `ledger.invoice_number` does the rest. A second attempt at the
same invoice number fails the insert and never reaches the payment call, which is what
stops INV-1004 and INV-1004_revised from paying twice.
"""

import sqlite3
from decimal import Decimal
from typing import Any

from galatiq.models import ApprovalDecision, Invoice, Outcome, PaymentResult, PaymentStatus
from galatiq.store.repository import record_payment


def mock_payment(vendor: str, amount: Decimal) -> dict[str, Any]:
    """The banking integration, as supplied with the assessment.

    Stands in for a real payments API. Everything around it -- the idempotency key, the
    write-ahead ordering, the audit record -- is built as though this were real, because
    replacing it with a real client should be a one-file change rather than a redesign.
    """
    print(f"Paid {amount} to {vendor}")
    return {"status": "success"}


def execute_payment(
    conn: sqlite3.Connection,
    invoice: Invoice,
    decision: ApprovalDecision,
    *,
    provider: str,
    model: str,
) -> PaymentResult:
    """Pay an approved invoice, at most once.

    Refuses anything not approved rather than trusting the caller. The routing already
    ensures only approved invoices arrive here, and a payment function that would release
    money for a rejected invoice if called wrongly is one bad refactor away from doing so.
    """
    if decision.outcome is not Outcome.APPROVED:
        return PaymentResult(
            invoice_number=invoice.invoice_number or "",
            status=PaymentStatus.NOT_ATTEMPTED,
            message=f"Not approved ({decision.outcome}).",
        )

    if not invoice.invoice_number:
        # Without a number there is no idempotency key, and without that a retry pays
        # again. The integrity check has already reported this; refusing here means it
        # cannot be bypassed by a caller who did not read the findings.
        return PaymentResult(
            invoice_number="",
            status=PaymentStatus.FAILED,
            message="Cannot pay an invoice with no number: there is no idempotency key.",
        )

    if invoice.total is None:
        return PaymentResult(
            invoice_number=invoice.invoice_number,
            status=PaymentStatus.FAILED,
            message="Cannot pay an invoice with no readable total.",
        )

    vendor = invoice.vendor or "unknown vendor"

    # Write-ahead. If this returns False the invoice is already in the ledger, and the
    # payment call below is never reached.
    written = record_payment(
        conn,
        invoice_number=invoice.invoice_number,
        amount=invoice.total,
        currency=invoice.currency or "USD",
        vendor=vendor,
        revision=invoice.revision,
        provider=provider,
        model=model,
    )

    if not written:
        return PaymentResult(
            invoice_number=invoice.invoice_number,
            status=PaymentStatus.ALREADY_PAID,
            amount=invoice.total,
            currency=invoice.currency,
            vendor=vendor,
            provider=provider,
            model=model,
            message="Already in the ledger; no payment made.",
        )

    response = mock_payment(vendor, invoice.total)

    return PaymentResult(
        invoice_number=invoice.invoice_number,
        status=(
            PaymentStatus.PAID
            if response.get("status") == "success"
            else PaymentStatus.FAILED
        ),
        amount=invoice.total,
        currency=invoice.currency,
        vendor=vendor,
        provider=provider,
        model=model,
        message=str(response),
    )
