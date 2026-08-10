"""Tests for turning stated amounts into numbers.

The cases that matter are the ones where the text is not a number. A real invoice
states amounts that do not parse — INV-1012's "$3,500.O0" has a letter O where a zero
belongs — and what the system does about that is the difference between a traceable
rejection, a crash, and a silently rewritten payment.

Every one of these was found by running the pipeline against the real model.
"""

from decimal import Decimal

import pytest

from galatiq.amounts import parse_amounts, unparsed_amount_findings
from galatiq.models import FindingCode, Invoice, LineItem, Severity
from galatiq.money import parse_rate, try_parse_money, try_parse_rate


class TestRates:
    """A tax rate is not money, and reusing the money parser for it was a bug.

    Documents write the same rate two ways: the JSON invoices say 0.07, the CSVs label
    a column "Tax (6%)". A faithful transcription of the second is "6%", which the
    money parser rejects — found on the first live run, on four separate invoices.
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("6%", Decimal("0.06")),      # INV-1007's "Tax (6%)"
            ("0%", Decimal("0")),         # INV-1001, INV-1015
            ("7%", Decimal("0.07")),      # INV-1013
            ("0.07", Decimal("0.07")),    # the JSON invoices
            ("0.10", Decimal("0.10")),
            (" 8 % ".replace(" ", ""), Decimal("0.08")),
        ],
    )
    def test_both_notations(self, text, expected):
        assert parse_rate(text) == expected

    def test_percent_and_fraction_agree(self):
        """6% and 0.06 are one value in two notations, not two readings."""
        assert parse_rate("6%") == parse_rate("0.06")

    def test_unparseable_returns_none(self):
        assert try_parse_rate("not a rate") is None
        assert try_parse_rate(None) is None


class TestLenientMoney:
    def test_valid_amounts(self):
        assert try_parse_money("$9,975.00") == Decimal("9975.00")

    def test_ocr_damage_returns_none(self):
        """INV-1012's total. Not an exception — a real invoice states this."""
        assert try_parse_money("$3,500.O0") is None

    def test_floats_still_raise(self):
        """That guard protects our arithmetic, not the document's. A messy invoice is
        no reason to accept binary rounding into a payment."""
        with pytest.raises(TypeError):
            try_parse_money(3500.00)

    def test_invalid_operation_is_reachable_as_value_error(self):
        """decimal.InvalidOperation inherits from ArithmeticError, not ValueError.

        A caller writing `except ValueError` — the obvious thing to write — would not
        catch it, and the exception escaped through pydantic and took down the graph.
        Found exactly that way, on the first live run.
        """
        from galatiq.money import parse_money

        with pytest.raises(ValueError):
            parse_money("$3,500.O0")


class TestParseAmounts:
    def test_raw_becomes_parsed(self):
        invoice = parse_amounts(
            Invoice(
                invoice_number="INV-1001",
                subtotal_raw="$5,000.00",
                tax_rate_raw="0%",
                total_raw="$5,000.00",
            )
        )

        assert invoice.subtotal == Decimal("5000.00")
        assert invoice.tax_rate == Decimal("0")
        assert invoice.total == Decimal("5000.00")

    def test_raw_text_is_never_overwritten(self):
        """The evidence has to survive the conversion."""
        invoice = parse_amounts(Invoice(total_raw="$5,000.00"))
        assert invoice.total_raw == "$5,000.00"

    def test_line_items(self):
        invoice = parse_amounts(
            Invoice(
                line_items=[
                    LineItem(
                        raw_name="WidgetA",
                        quantity=10,
                        unit_price_raw="$250.00",
                        stated_amount_raw="$2,500.00",
                    )
                ]
            )
        )
        line = invoice.line_items[0]

        assert line.unit_price == Decimal("250.00")
        assert line.stated_amount == Decimal("2500.00")

    def test_unparseable_leaves_the_parsed_field_null(self):
        invoice = parse_amounts(Invoice(total_raw="$3,500.O0"))

        assert invoice.total is None
        assert invoice.total_raw == "$3,500.O0"

    def test_an_already_parsed_value_is_left_alone(self):
        """Structural hints arrive with real Decimals. Re-deriving them from text
        would be work with no answer to add."""
        invoice = parse_amounts(Invoice(total_raw="ignored", total=Decimal("42.00")))
        assert invoice.total == Decimal("42.00")

    def test_no_raw_no_change(self):
        invoice = Invoice(invoice_number="INV-1001")
        assert parse_amounts(invoice) is invoice


class TestUnparsedAmountFindings:
    """Where INV-1012's OCR damage becomes visible."""

    def test_ocr_damage_is_reported_with_the_original_text(self):
        invoice = parse_amounts(Invoice(invoice_number="INV-1012", total_raw="$3,500.O0"))

        findings = unparsed_amount_findings(invoice)

        assert len(findings) == 1
        assert findings[0].code == FindingCode.DATA_INTEGRITY
        assert "$3,500.O0" in findings[0].evidence

    def test_severity_is_critical(self):
        """An invoice whose total cannot be read is not one to pay. The alternative —
        letting a retry quietly rewrite it as 3500.00 — produces a payment nobody can
        trace back to what the document said."""
        invoice = parse_amounts(Invoice(total_raw="$3,500.O0"))
        assert unparsed_amount_findings(invoice)[0].severity == Severity.CRITICAL

    def test_line_level_damage_names_the_line(self):
        invoice = parse_amounts(
            Invoice(
                line_items=[
                    LineItem(raw_name="WidgetA", quantity=1, unit_price_raw="$25O.00")
                ]
            )
        )

        findings = unparsed_amount_findings(invoice)

        assert "Line 1" in findings[0].message
        assert "WidgetA" in findings[0].evidence

    def test_clean_amounts_produce_nothing(self):
        invoice = parse_amounts(
            Invoice(
                subtotal_raw="5000.00",
                total_raw="5000.00",
                tax_rate_raw="0%",
                line_items=[
                    LineItem(raw_name="WidgetA", quantity=10, unit_price_raw="250.00")
                ],
            )
        )

        assert unparsed_amount_findings(invoice) == []

    def test_absent_amounts_produce_nothing(self):
        """INV-1003 states no subtotal at all. Absent is not the same as unreadable."""
        assert unparsed_amount_findings(Invoice(invoice_number="INV-1003")) == []
