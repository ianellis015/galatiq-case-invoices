"""Signals that an invoice is trying to get itself paid.

INV-1003 is the planted case: "URGENT - Pay immediately to avoid penalties!!! Wire
transfer preferred." Every element of that is a known pressure tactic -- manufactured
urgency, a threatened consequence, and a request for a payment method that is fast and
irreversible.

This check is deliberately deterministic and deliberately dumb. It matches phrases in
text the extractor preserved. It does not ask a model whether an invoice "seems"
fraudulent, for two reasons: a keyword list can be read, argued with and adjusted by
someone in accounts payable, and asking a model to judge text written by an adversary
puts the adversary and the judge in the same channel.

The one thing it will not do is obey. Manipulation language is scored, quoted, and
attached to the invoice as evidence -- the same way a stock shortfall is.
"""

import re
from decimal import Decimal

from galatiq.checks import CheckContext
from galatiq.models import Finding, FindingCode, Invoice, Severity

# Manufactured urgency and threatened consequences. Individually weak -- plenty of honest
# invoices say "due immediately" -- which is why these are WARN and accumulate rather
# than reject on their own.
_URGENCY = (
    "urgent",
    "immediately",
    "asap",
    "act now",
    "final notice",
    "avoid penalties",
    "avoid penalty",
    "late fee will",
    "legal action",
    "account will be suspended",
    "do not delay",
)

# Requests for fast, irreversible payment methods, or for a change of banking details.
# The second is the single most common invoice fraud there is: a real vendor, a real
# invoice, and a bank account that belongs to somebody else.
_PAYMENT_PRESSURE = (
    "wire transfer",
    "wire only",
    "bank transfer only",
    "new bank details",
    "updated bank details",
    "changed our bank",
    "crypto",
    "bitcoin",
    "gift card",
)

# Text attempting to address the system rather than describe the invoice.
_INJECTION = (
    "ignore previous",
    "ignore all previous",
    "disregard the above",
    "system prompt",
    "you are an ai",
    "as an ai",
    "approve this invoice",
    "do not flag",
    "skip validation",
    "override",
)

# Three or more consecutive exclamation marks. Punctuation as pressure.
_SHOUTING = re.compile(r"!{3,}")


def check_fraud(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Scan the document's free text for pressure and manipulation."""
    haystack = " ".join(
        part
        for part in (invoice.notes, invoice.payment_terms, invoice.vendor)
        if part
    ).lower()

    findings: list[Finding] = []

    findings.extend(_phrase_findings(haystack, _URGENCY, "urgency", Severity.WARN))
    findings.extend(
        _phrase_findings(haystack, _PAYMENT_PRESSURE, "payment-method", Severity.WARN)
    )
    findings.extend(_injection_findings(haystack))
    findings.extend(_shouting(invoice, haystack))
    findings.extend(_round_number(invoice))

    return findings


def _phrase_findings(
    haystack: str, phrases: tuple[str, ...], kind: str, severity: Severity
) -> list[Finding]:
    hits = [phrase for phrase in phrases if phrase in haystack]
    if not hits:
        return []

    return [
        Finding(
            code=FindingCode.FRAUD_SIGNAL,
            severity=severity,
            message=(
                f"The document uses {kind} language: "
                f"{', '.join(repr(h) for h in hits)}."
            ),
            evidence=_quote(haystack, hits[0]),
        )
    ]


def _injection_findings(haystack: str) -> list[Finding]:
    """Text addressed to the system rather than describing the invoice.

    CRITICAL, and not because the instruction might work -- the extractor fences
    document content and the system prompt says it carries no authority. It is critical
    because a document that tries is a document written by someone who expects an
    automated reader and wants to steer it, and that is worth a human's attention
    regardless of whether the attempt succeeded.
    """
    hits = [phrase for phrase in _INJECTION if phrase in haystack]
    if not hits:
        return []

    return [
        Finding(
            code=FindingCode.PROMPT_INJECTION,
            severity=Severity.CRITICAL,
            message=(
                "The document contains text addressed to an automated reader rather "
                f"than describing the invoice: {', '.join(repr(h) for h in hits)}."
            ),
            evidence=_quote(haystack, hits[0]),
        )
    ]


def _shouting(invoice: Invoice, haystack: str) -> list[Finding]:
    if not _SHOUTING.search(haystack):
        return []

    return [
        Finding(
            code=FindingCode.FRAUD_SIGNAL,
            severity=Severity.INFO,
            message="The document uses emphatic punctuation.",
            evidence=_quote(haystack, "!!!"),
        )
    ]


def _round_number(invoice: Invoice) -> list[Finding]:
    """A total that is a suspiciously round number.

    INFO only, and I nearly left it out. INV-1003's $100,000.00 is round -- and so is
    INV-1001's $5,000.00, which is a perfectly ordinary invoice. On its own this signal
    is close to noise.

    It earns its place as a contributor to a risk score rather than as grounds for
    anything: round *and* urgent *and* wire-transfer is a different picture from round
    alone. Kept at INFO so it can never reject an invoice by itself.
    """
    if invoice.total is None or invoice.total <= 0:
        return []

    if invoice.total % Decimal("10000") != 0:
        return []

    return [
        Finding(
            code=FindingCode.FRAUD_SIGNAL,
            severity=Severity.INFO,
            message=f"The total is an exact round number ({invoice.total}).",
            evidence=f"total: {invoice.total}",
        )
    ]


def _quote(haystack: str, phrase: str, window: int = 60) -> str:
    """A snippet of the document around a matched phrase.

    Evidence has to be quotable. "Urgency language detected" tells a reviewer nothing
    they can check; the sentence it came from tells them everything.
    """
    index = haystack.find(phrase)
    if index == -1:
        return phrase

    start = max(0, index - window // 2)
    end = min(len(haystack), index + len(phrase) + window // 2)
    snippet = haystack[start:end].strip()

    return f"...{snippet}..." if start > 0 or end < len(haystack) else snippet
