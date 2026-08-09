"""Tests for money handling.

Every assertion here is about a value surviving a round trip or a comparison without
drifting. A cent lost in this module is a cent that reappears months later as a
ledger row that will not reconcile, so I test the boundaries hard.
"""

from decimal import Decimal

import pytest

from galatiq.money import (
    from_cents,
    parse_money,
    round_money,
    to_cents,
    within_tolerance,
)


class TestParseMoney:
    """parse_money accepts messy real-world strings and rejects floats."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("250.00", Decimal("250.00")),
            ("$9,975.00", Decimal("9975.00")),
            ("$22,562.80", Decimal("22562.80")),
            ("€1,234.00", Decimal("1234.00")),  # invoice 1014 is in EUR
            ("(250.00)", Decimal("-250.00")),   # accounting notation for negative
            ("0", Decimal("0")),
        ],
    )
    def test_parses_documented_forms(self, text, expected):
        assert parse_money(text) == expected

    def test_rejects_float(self):
        """The guard that makes the float bug impossible to introduce by accident.

        Decimal(0.1) silently yields 0.1000000000000000055511151231257827, which then
        propagates all the way to storage. Raising is the entire reason parse_money
        exists rather than calling Decimal() at each site.
        """
        with pytest.raises(TypeError, match="refusing to build money from float"):
            parse_money(0.1)

    def test_rejects_bool(self):
        """bool subclasses int and would otherwise parse as 0 or 1."""
        with pytest.raises(TypeError):
            parse_money(True)

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            parse_money("   ")

    def test_does_not_repair_ocr_damage(self):
        """Invoice 1012 contains "$3,500.O0" — a letter O standing in for a zero.

        Repairing that belongs upstream in the extractor, where there is document
        context to justify it. Down here I want a malformed amount to fail loudly,
        because guessing wrong changes what gets paid.
        """
        with pytest.raises(Exception):
            parse_money("$3,500.O0")


class TestCentsRoundTrip:
    """Decimal -> integer cents -> Decimal has to be lossless."""

    @pytest.mark.parametrize(
        "amount,cents",
        [
            ("250.00", 25000),
            ("9975.00", 997500),      # invoice 1012, just under the $10k threshold
            ("22562.80", 2256280),    # invoice 1013's stated total
            ("21040.00", 2104000),    # invoice 1013's subtotal
            ("1472.80", 147280),      # invoice 1013's tax
            ("0.01", 1),
            ("0.00", 0),
            ("-250.00", -25000),      # invoice 1009's negative total
        ],
    )
    def test_round_trip(self, amount, cents):
        value = Decimal(amount)
        assert to_cents(value) == cents
        assert from_cents(cents) == value

    def test_to_cents_rejects_float(self):
        with pytest.raises(TypeError):
            to_cents(9975.00)

    def test_from_cents_rejects_non_int(self):
        with pytest.raises(TypeError):
            from_cents(997500.0)


class TestRounding:
    """One rounding policy, applied once, at the end."""

    def test_half_up(self):
        assert round_money(Decimal("1.005")) == Decimal("1.01")
        assert round_money(Decimal("1.004")) == Decimal("1.00")

    def test_sub_cent_tax_is_rounded_at_the_boundary(self):
        """7% of an odd subtotal produces a fraction of a cent that has to round."""
        tax = Decimal("21043.21") * Decimal("0.07")   # 1473.0247
        assert round_money(tax) == Decimal("1473.02")
        assert to_cents(tax) == 147302

    def test_rounding_once_differs_from_rounding_per_line(self):
        """Why the policy says round once, on the total.

        Three lines each ending in a third of a cent: rounding each and summing gives
        a different answer than summing and rounding. Both are defensible; picking
        one and applying it everywhere is what matters, and this test pins which one
        I picked.
        """
        lines = [Decimal("0.334"), Decimal("0.334"), Decimal("0.334")]

        per_line = sum(round_money(line) for line in lines)   # 0.33 * 3 = 0.99
        on_total = round_money(sum(lines))                    # 1.002   -> 1.00

        assert per_line == Decimal("0.99")
        assert on_total == Decimal("1.00")
        assert per_line != on_total


class TestTolerance:
    """The +/-$0.01 reconciliation band."""

    def test_penny_difference_is_forgiven(self):
        """Forgives the vendor rounding differently than I did — not fraud."""
        assert within_tolerance(Decimal("1472.80"), Decimal("1472.81"))

    def test_two_cents_is_not(self):
        assert not within_tolerance(Decimal("1472.80"), Decimal("1472.82"))

    def test_invoice_1013_gap_is_far_outside_tolerance(self):
        """The case this tolerance has to get right.

        1013's line items sum exactly to its stated subtotal, and 7% tax on that
        subtotal matches its stated tax exactly. The $50 appears only at the final
        total — 5,000x the tolerance, so unambiguously a discrepancy rather than a
        rounding artifact.
        """
        subtotal = Decimal("21040.00")
        tax = Decimal("1472.80")
        stated_total = Decimal("22562.80")

        computed_total = subtotal + tax
        assert computed_total == Decimal("22512.80")
        assert not within_tolerance(computed_total, stated_total)
        assert stated_total - computed_total == Decimal("50.00")
