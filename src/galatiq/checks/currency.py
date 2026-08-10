"""Currency, and getting everything into the units the rules are written in.

The approval threshold is "over $10,000" -- a USD rule. INV-1014 is in EUR. Comparing
EUR 9,500 against a USD 10,000 threshold is comparing different units, and it fails in
the direction that matters: EUR 9,500 is roughly USD 10,300, so an invoice that should
attract extra scrutiny passes under the bar.

This check does not decide anything. It reports the currency, records the converted value
and the rate used, and leaves the threshold to the policy engine -- which then only ever
compares USD to USD.

An unstated currency is reported too. Most documents name theirs; the ones that do not
show a dollar sign, and reading that as USD is an assumption. Assumptions that change what
gets paid belong in the audit trail rather than in a default argument.
"""

from galatiq.checks import CheckContext
from galatiq.fx import AS_OF, DEFAULT_CURRENCY, UnknownCurrencyError, to_usd
from galatiq.models import Finding, FindingCode, Invoice, Severity


def check_currency(invoice: Invoice, context: CheckContext) -> list[Finding]:
    """Report the currency, and convert the total to USD."""
    if invoice.total is None:
        return []

    stated = (invoice.currency or "").strip().upper()

    if not stated:
        return [
            Finding(
                code=FindingCode.NON_USD_CURRENCY,
                severity=Severity.INFO,
                message=(
                    f"No currency stated; assuming {DEFAULT_CURRENCY} for threshold "
                    "evaluation."
                ),
                evidence=f"total {invoice.total} with no currency field",
            )
        ]

    if stated == DEFAULT_CURRENCY:
        return []

    try:
        converted = to_usd(invoice.total, stated, rates=context.fx_rates)
    except UnknownCurrencyError as exc:
        # Refusing to guess. A threshold comparison against an unconverted foreign
        # amount is not a comparison, and silently treating it as USD would make the
        # rule meaningless exactly when it matters.
        return [
            Finding(
                code=FindingCode.NON_USD_CURRENCY,
                severity=Severity.CRITICAL,
                message=(
                    f"Invoice is in {stated}, and no exchange rate is available. "
                    "The amount cannot be evaluated against USD thresholds."
                ),
                evidence=str(exc),
            )
        ]

    rate = context.fx_rates[stated]

    return [
        Finding(
            code=FindingCode.NON_USD_CURRENCY,
            severity=Severity.WARN,
            message=(
                f"Invoice is in {stated}. {invoice.total} {stated} converts to "
                f"{converted.quantize(invoice.total)} USD at {rate} (as of {AS_OF})."
            ),
            evidence=(
                f"{invoice.total} {stated} x {rate} = {converted} USD; rates as of {AS_OF}"
            ),
        )
    ]
