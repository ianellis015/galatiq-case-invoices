"""Dates, and whether they agree with the payment terms.

Three states, and keeping them apart is the whole reason amounts and dates are carried
twice:

    due_date_raw   due_date   meaning
    -----------------------------------------------------------------
    None           None       the document stated no due date
    "yesterday"    None       it stated one, and it is not a date
    "2026-02-01"   a date     it stated one and we understood it

A single field collapses the first two, and the system reports "missing due date" on
INV-1003, which states one loudly. Wrong in a way a reviewer cannot correct, because the
evidence is gone.

The terms check is the subtle one. INV-1002 says "Net 30" and gives a due date equal to
its issue date -- an invoice demanding immediate payment while claiming thirty-day terms.
Neither field is wrong on its own; they contradict each other.
"""

import re
from datetime import timedelta

from galatiq.checks import CheckContext
from galatiq.models import Finding, FindingCode, Invoice, Severity

# "Net 30", "NET30", "net 60 days". Deliberately narrow: matching loosely here would
# invent terms from prose and then report a contradiction with something the document
# never said.
_NET_TERMS = re.compile(r"\bnet\s*(\d{1,3})\b", re.IGNORECASE)

# Terms that mean "pay now", where a due date equal to the issue date is correct rather
# than contradictory.
_IMMEDIATE_TERMS = ("immediate", "due on receipt", "upon receipt", "cod", "prepaid")

# How far a due date may drift from its terms before it is a contradiction.
#
# Vendors count differently -- from the invoice date or the day after, sometimes to a
# month end -- and most of the corpus is a day either side of exact. Demanding exactness
# fires on nearly every invoice, and a check that flags everything flags nothing.
#
# Three days is wide enough to absorb counting conventions and narrow enough to catch
# INV-1002, whose due date equals its issue date under "Net 30" -- thirty days out.
_TERMS_TOLERANCE_DAYS = 3


def check_dates(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Parseability, past-due, and agreement with the stated terms."""
    findings: list[Finding] = []

    findings.extend(_due_date_present_and_readable(invoice))
    findings.extend(_past_due(invoice, context))
    findings.extend(_terms_agree(invoice))

    return findings


def _due_date_present_and_readable(invoice: Invoice) -> list[Finding]:
    if invoice.due_date is not None:
        return []

    if invoice.due_date_raw:
        return [
            Finding(
                code=FindingCode.DATE_UNPARSEABLE,
                severity=Severity.CRITICAL,
                message=(
                    f"The due date {invoice.due_date_raw!r} is not a date."
                ),
                evidence=f"due_date_raw: {invoice.due_date_raw!r}",
            )
        ]

    return [
        Finding(
            code=FindingCode.DATA_INTEGRITY,
            severity=Severity.WARN,
            message="No due date stated.",
            evidence=f"invoice: {invoice.invoice_number or 'unnumbered'}",
        )
    ]


def _past_due(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """A due date already in the past.

    WARN rather than CRITICAL. Invoices arrive late for ordinary reasons -- postal
    delays, a backlog in the inbox -- and a past-due date is a prompt to pay promptly,
    not evidence of fraud. It becomes interesting in combination with other signals,
    which is the risk score's job rather than this check's.
    """
    if invoice.due_date is None or invoice.due_date >= context.today:
        return []

    days = (context.today - invoice.due_date).days

    return [
        Finding(
            code=FindingCode.DATE_PAST_DUE,
            severity=Severity.WARN,
            message=f"Due date has passed ({days} day(s) ago).",
            evidence=f"due {invoice.due_date.isoformat()}, today {context.today.isoformat()}",
        )
    ]


def _terms_agree(invoice: Invoice) -> list[Finding]:
    """Does the due date match what the payment terms promise?

    INV-1002 is the case: "Net 30" with a due date equal to the issue date. Read alone,
    each field is unremarkable. Together they are an invoice claiming thirty-day terms
    while demanding payment today -- which is either a mistake worth catching or pressure
    worth noticing.
    """
    terms = invoice.payment_terms or ""
    match = _NET_TERMS.search(terms)

    if not match or invoice.issue_date is None or invoice.due_date is None:
        return []

    if any(phrase in terms.lower() for phrase in _IMMEDIATE_TERMS):
        return []

    promised = int(match.group(1))
    expected = invoice.issue_date + timedelta(days=promised)
    actual = (invoice.due_date - invoice.issue_date).days

    if abs(actual - promised) <= _TERMS_TOLERANCE_DAYS:
        return []

    return [
        Finding(
            code=FindingCode.TERMS_MISMATCH,
            severity=Severity.WARN,
            message=(
                f"Terms say {terms.strip()!r}, which implies a due date of "
                f"{expected.isoformat()}, but the invoice states "
                f"{invoice.due_date.isoformat()} ({actual} days)."
            ),
            evidence=(
                f"terms={terms.strip()!r} issued={invoice.issue_date.isoformat()} "
                f"due={invoice.due_date.isoformat()} implied={expected.isoformat()}"
            ),
        )
    ]
