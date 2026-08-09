"""Money handling.

I never represent money as a float. Floats are binary fractions, so ordinary decimal
amounts like 0.10 have no exact representation and arithmetic accumulates error.
Three places in this system where that would bite:

  1. Validation compares a computed sum against a stated total within a $0.01
     tolerance. Float noise either manufactures false mismatches or eats into the
     tolerance I rely on to catch real ones.
  2. Policy thresholds — $10,000 for VP scrutiny, the $9,500-$10,000 structuring
     band — are exact boundary comparisons. A value that should be 10000.00 but
     computes to 9999.999999 routes to the wrong branch, silently.
  3. A ledger row reading 4999.999999999999 is not a defensible payment record.

So: `Decimal` in memory, integer cents on disk (SQLite has no decimal type), and
conversion only through this module. Confining it to one file is what lets me claim
the rest of the system cannot introduce a rounding bug.
"""

from decimal import Decimal, ROUND_HALF_UP

# Every amount reaching the database or a policy comparison carries this precision.
CENT = Decimal("0.01")

# One rounding policy for the whole system: half-up, the ordinary commercial
# convention. I declare it once here so tax calculation, FX conversion and storage
# can't each quietly pick a different one and disagree by a cent.
ROUNDING = ROUND_HALF_UP

# Reconciliation tolerance. Two amounts differing by no more than this agree.
# This exists to forgive *the vendor's* rounding convention differing from mine —
# they rounded tax per line, I round on the total — not to paper over my own
# arithmetic, which is exact.
TOLERANCE = Decimal("0.01")


def parse_money(value: str | int | Decimal) -> Decimal:
    """Build a Decimal from a string, int, or existing Decimal.

    Handles the messy forms that turn up in invoice documents: currency symbols,
    thousands separators, parenthesised negatives.

    The float rejection is the reason this function exists rather than calling
    `Decimal()` directly at each site. `Decimal(0.1)` returns
    0.1000000000000000055511151231257827 without complaint, inheriting the exact
    error I'm trying to avoid, and the mistake survives all the way to the ledger.
    Raising makes it impossible to introduce by accident.
    """
    # bool subclasses int, so without this check it slips through as 0 or 1.
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(
            f"refusing to build money from {type(value).__name__} "
            f"({value!r}) -- pass a string to preserve exact decimal precision"
        )

    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)

    text = value.strip()
    if not text:
        raise ValueError("cannot parse an empty string as money")

    # "(1,234.56)" is accounting notation for a negative amount.
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    for junk in ("$", "€", "£", ",", "USD", "EUR", "GBP"):
        text = text.replace(junk, "")
    text = text.strip()

    # Deliberately no attempt to repair OCR damage here — invoice 1012 contains
    # "$3,500.O0", a letter O standing in for a zero. Repair belongs upstream in the
    # extractor, where there's document context to justify it. Down here a malformed
    # amount should fail loudly, because guessing wrong changes what gets paid.
    amount = Decimal(text)
    return -amount if negative else amount


def round_money(amount: Decimal) -> Decimal:
    """Round to whole cents under the system-wide half-up policy.

    Apply once, at the end of a calculation — not on each intermediate step.
    Rounding each line item and then summing gives a different answer than summing
    and then rounding, and the difference surfaces as a spurious few-cent
    discrepancy against the vendor's stated total.
    """
    return amount.quantize(CENT, rounding=ROUNDING)


def to_cents(amount: Decimal) -> int:
    """Convert a Decimal amount to the integer minor units I store in SQLite.

    $9,975.00 becomes 997500. Sub-cent precision — 7% tax on an odd subtotal, say —
    is rounded here by `round_money`, which makes this the single boundary where
    rounding happens on the way to storage.
    """
    if isinstance(amount, bool) or isinstance(amount, float):
        raise TypeError(
            f"refusing to convert {type(amount).__name__} to cents -- "
            f"use parse_money() to build a Decimal first"
        )
    if not isinstance(amount, Decimal):
        amount = parse_money(amount)

    return int(round_money(amount) * 100)


def from_cents(cents: int) -> Decimal:
    """Convert integer minor units back to a Decimal amount.

    997500 becomes Decimal('9975.00'). Exact in both directions, since integers have
    no representation error — a round trip through the database cannot lose a cent.
    """
    if isinstance(cents, bool) or not isinstance(cents, int):
        raise TypeError(f"expected int cents, got {type(cents).__name__}")

    return round_money(Decimal(cents) / 100)


def within_tolerance(a: Decimal, b: Decimal, tolerance: Decimal = TOLERANCE) -> bool:
    """True when two amounts agree to within the reconciliation tolerance.

    The arithmetic checks use this twice: line items against the stated subtotal, and
    subtotal + tax against the stated total. Invoice 1013's $50.00 gap is 5,000x the
    tolerance, so it lands as a genuine discrepancy rather than a rounding artifact.
    """
    return abs(a - b) <= tolerance
