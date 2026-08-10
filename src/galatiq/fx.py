"""Currency conversion, from a static table.

The approval rules are written in USD -- "over $10,000 requires VP scrutiny" -- and
INV-1014 is denominated in EUR. Comparing 9,500 EUR against a 10,000 USD threshold is
comparing different units, and it fails in the dangerous direction: EUR 9,500 is about
USD 10,300, so an invoice that should attract extra scrutiny slips under the line if the
numbers are compared as though they were the same thing.

Rates are hardcoded because the brief says to assume no internet. That is a real
limitation rather than a shortcut: these rates were true on the date below and drift
every day afterwards, so a production system would read them from a rates service and
record which rate it applied to which decision. Recorded in the README's assumptions.

What this module does *not* do is decide anything. It converts, records the rate used,
and leaves the judgement to the policy engine.
"""

from decimal import Decimal

# Approximate mid-market rates, USD per unit of currency, as of 2026-01-31.
#
# Stored as strings and built as Decimals for the same reason every other amount in this
# system is: a float rate multiplied by a Decimal amount reintroduces exactly the
# representation error the rest of the design keeps out.
AS_OF = "2026-01-31"

RATES: dict[str, Decimal] = {
    "USD": Decimal("1.00"),
    "EUR": Decimal("1.09"),
    "GBP": Decimal("1.27"),
    "CAD": Decimal("0.74"),
    "AUD": Decimal("0.66"),
    "JPY": Decimal("0.0067"),
}

DEFAULT_CURRENCY = "USD"


class UnknownCurrencyError(ValueError):
    """No rate for this currency.

    Raised rather than defaulted. Guessing that an unrecognised currency is USD would
    silently convert a threshold comparison into a coin flip, and the whole point of
    normalising is that the comparison means something.
    """


def is_known(currency: str | None) -> bool:
    """True when a rate exists for this currency."""
    return bool(currency) and currency.strip().upper() in RATES


def to_usd(
    amount: Decimal,
    currency: str | None,
    *,
    rates: dict[str, Decimal] | None = None,
) -> Decimal:
    """Convert an amount to USD.

    A missing currency is treated as USD. That is an assumption rather than a fact, and
    the currency check reports it as one -- most documents in the corpus name their
    currency, and the ones that do not show a dollar sign.

    `rates` is injectable so a check can use the snapshot it was given rather than
    reaching for module state, which keeps checks pure functions of their inputs.
    """
    table = RATES if rates is None else rates
    code = (currency or DEFAULT_CURRENCY).strip().upper()

    if code not in table:
        raise UnknownCurrencyError(
            f"no exchange rate for {code!r}; known: {', '.join(sorted(table))}"
        )

    return amount * table[code]
