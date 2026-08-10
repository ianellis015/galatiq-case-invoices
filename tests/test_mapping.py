"""Tests for hint mapping and the two-reading cross-check.

`compare_to_hint` is why the structural parsers still exist. A parser and a model read
the same document independently, and the parser is the only thing that would notice a
model quietly dropping a line item — a mistake that produces a plausible,
self-consistent, wrong invoice rather than an error.
"""

from decimal import Decimal

from galatiq.mapping import compare_to_hint, hint_to_invoice
from galatiq.models import FindingCode, Invoice, LineItem, Severity


class TestHintToInvoice:
    def test_direct_fields(self):
        invoice = hint_to_invoice(
            {
                "invoice_number": "INV-1006",
                "vendor": "Acme Industrial Supplies",
                "subtotal": "2750.00",
                "total": "2750.00",
                "currency": "USD",
                "payment_terms": "Net 15",
            }
        )

        assert invoice.invoice_number == "INV-1006"
        assert invoice.subtotal == Decimal("2750.00")
        assert invoice.payment_terms == "Net 15"

    def test_dates_land_in_the_raw_fields(self):
        """Parsed values are filled in later. Keeping both is what lets the date check
        tell "no due date" apart from "an unparseable due date"."""
        invoice = hint_to_invoice({"date": "2026-01-25", "due_date": "2026-02-10"})

        assert invoice.issue_date_raw == "2026-01-25"
        assert invoice.due_date_raw == "2026-02-10"
        assert invoice.issue_date is None

    def test_line_items(self):
        invoice = hint_to_invoice(
            {
                "line_items": [
                    {"item": "WidgetA", "quantity": "5", "unit_price": "250.00"},
                    {
                        "item": "WidgetB",
                        "quantity": "3",
                        "unit_price": "500.00",
                        "amount": "1500.00",
                        "note": "Volume discount",
                    },
                ]
            }
        )

        assert len(invoice.line_items) == 2
        assert invoice.line_items[0].raw_name == "WidgetA"
        assert invoice.line_items[0].quantity == 5
        assert invoice.line_items[1].stated_amount == Decimal("1500.00")
        assert invoice.line_items[1].note == "Volume discount"

    def test_unparseable_quantity_is_preserved(self):
        """"a dozen" has to reach a DATA_INTEGRITY finding that can quote it."""
        invoice = hint_to_invoice(
            {"line_items": [{"item": "WidgetA", "quantity": "a dozen"}]}
        )

        assert invoice.line_items[0].quantity is None
        assert invoice.line_items[0].quantity_raw == "a dozen"

    def test_negative_quantity_survives(self):
        """INV-1009."""
        invoice = hint_to_invoice({"line_items": [{"item": "WidgetA", "quantity": -5}]})
        assert invoice.line_items[0].quantity == -5

    def test_unmapped_keys_go_to_extra(self):
        """A PO number does not map onto a modelled field, and discarding it means a
        human cannot see what the system saw and ignored."""
        invoice = hint_to_invoice(
            {"invoice_number": "INV-1007", "tax_label": "Tax (6%):"}
        )

        assert invoice.extra == {"tax_label": "Tax (6%):"}

    def test_provenance(self):
        invoice = hint_to_invoice(
            {"invoice_number": "INV-1006"},
            source_path="data/invoices/invoice_1006.csv",
            source_format="csv",
        )

        assert invoice.source_format == "csv"

    def test_empty_hint_yields_an_empty_invoice(self):
        invoice = hint_to_invoice({})

        assert invoice.invoice_number is None
        assert invoice.line_items == []


class TestCompareToHint:
    """Two independent readings that have to agree."""

    def _invoice(self, **overrides):
        base = {
            "invoice_number": "INV-1006",
            "line_items": [
                LineItem(raw_name="WidgetA", quantity=5, unit_price="250.00"),
                LineItem(raw_name="WidgetB", quantity=3, unit_price="500.00"),
            ],
            "subtotal": "2750.00",
            "total": "2750.00",
        }
        return Invoice(**(base | overrides))

    def _hint(self, **overrides):
        base = {
            "invoice_number": "INV-1006",
            "line_items": [
                {"item": "WidgetA", "quantity": "5", "unit_price": "250.00"},
                {"item": "WidgetB", "quantity": "3", "unit_price": "500.00"},
            ],
            "subtotal": "2750.00",
            "total": "2750.00",
        }
        return base | overrides

    def test_agreement_produces_nothing(self):
        assert compare_to_hint(self._invoice(), self._hint()) == []

    def test_no_hint_produces_nothing(self):
        """Most documents have no hint. Absence is not disagreement."""
        assert compare_to_hint(self._invoice(), None) == []
        assert compare_to_hint(self._invoice(), {}) == []

    def test_invoice_number_disagreement(self):
        findings = compare_to_hint(self._invoice(), self._hint(invoice_number="INV-9999"))

        assert findings[0].code == FindingCode.HINT_DISAGREEMENT
        assert "INV-9999" in findings[0].evidence

    def test_total_disagreement(self):
        findings = compare_to_hint(self._invoice(), self._hint(total="21400.00"))

        assert len(findings) == 1
        assert "total" in findings[0].message

    def test_a_penny_is_within_tolerance(self):
        """Two readings need not phrase themselves identically. The point is catching a
        misread, not demanding agreement to the cent on a rounding convention."""
        assert compare_to_hint(self._invoice(), self._hint(total="2750.01")) == []

    def test_dropped_line_item(self):
        """The mistake this cross-check exists for.

        INV-1006 repeats the `item` key; a reader that collapses duplicates produces
        one line instead of two — plausible, self-consistent, and wrong. The parser
        walks the file positionally and would not make that mistake.
        """
        invoice = self._invoice(
            line_items=[LineItem(raw_name="WidgetB", quantity=3, unit_price="500.00")]
        )
        findings = compare_to_hint(invoice, self._hint())

        assert findings[0].code == FindingCode.HINT_DISAGREEMENT
        assert "parsed 2 line(s), extracted 1" in findings[0].evidence

    def test_quantity_disagreement_on_a_matching_line(self):
        invoice = self._invoice(
            line_items=[
                LineItem(raw_name="WidgetA", quantity=50, unit_price="250.00"),
                LineItem(raw_name="WidgetB", quantity=3, unit_price="500.00"),
            ]
        )
        findings = compare_to_hint(invoice, self._hint())

        assert len(findings) == 1
        assert "line 1" in findings[0].message
        assert "parsed 5, extracted 50" in findings[0].evidence

    def test_severity_is_warn_not_critical(self):
        """A disagreement means one of the two is wrong and the system does not know
        which. That is information for a reviewer, not grounds to reject."""
        findings = compare_to_hint(self._invoice(), self._hint(total="9999.00"))
        assert findings[0].severity == Severity.WARN

    def test_missing_values_are_not_disagreements(self):
        invoice = self._invoice(subtotal=None)
        assert compare_to_hint(invoice, self._hint()) == []

    def test_unparseable_hint_values_are_ignored(self):
        """A hint is advisory. Garbage in it must not crash the comparison."""
        assert compare_to_hint(self._invoice(), self._hint(total="not a number")) == []
