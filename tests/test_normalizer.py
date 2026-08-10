"""Tests for item-name normalisation.

The important test here is a negative one. `difflib` scores "WidgetC" against "WidgetA"
at about 0.857 — high enough that any fuzzy cutoff loose enough to accept "Widget A"
also accepts "WidgetC", which would silently turn INV-1016's unknown item into an
in-stock one and approve a payment for a product nobody sells.

So the failure modes are not symmetric, and neither is the design: formatting differences
are resolved, content differences never are.
"""

from galatiq.agents.normalizer import (
    ItemMatch,
    ItemMatches,
    canonical_key,
    normalize_invoice,
    resolve_deterministically,
)
from galatiq.models import Invoice, LineItem

from conftest import FakeLLM

CATALOG = {"WidgetA": 15, "WidgetB": 10, "GadgetX": 5, "FakeItem": 0}


def invoice_with(*names: str) -> Invoice:
    return Invoice(
        invoice_number="INV-TEST",
        line_items=[
            LineItem(raw_name=name, quantity=1, unit_price="100.00") for name in names
        ],
    )


class TestCanonicalKey:
    def test_formatting_differences_collapse(self):
        assert canonical_key("Widget A") == canonical_key("WidgetA") == "widgeta"

    def test_qualifiers_are_stripped(self):
        """INV-1010's "WidgetA (rush order)".

        Qualifiers go first, then punctuation. Strip in the other order and this becomes
        "widgetarushorder", which matches nothing.
        """
        assert canonical_key("WidgetA (rush order)") == "widgeta"
        assert canonical_key("WidgetB [backordered]") == "widgetb"
        assert canonical_key("GadgetX - expedited") == "gadgetx"

    def test_content_differences_survive(self):
        """The whole point. These must not collapse."""
        assert canonical_key("WidgetC") != canonical_key("WidgetA")

    def test_case_and_punctuation(self):
        assert canonical_key("WIDGET_A") == "widgeta"
        assert canonical_key("  widget-a  ") == "widgeta"


class TestDeterministicResolution:
    def _index(self):
        from galatiq.agents.normalizer import build_index

        return build_index(CATALOG)

    def test_exact(self):
        assert resolve_deterministically("WidgetA", self._index()) == "WidgetA"

    def test_spacing_variant(self):
        assert resolve_deterministically("Widget A", self._index()) == "WidgetA"
        assert resolve_deterministically("Gadget X", self._index()) == "GadgetX"

    def test_qualifier_variant(self):
        assert (
            resolve_deterministically("WidgetA (rush order)", self._index()) == "WidgetA"
        )

    def test_widgetc_does_not_become_widgeta(self):
        """The test this design exists for.

        A wrong match produces a payment for a product nobody sells, and nothing
        downstream catches it because the invoice looks entirely ordinary. A missed
        match produces a rejection someone can appeal in a minute.
        """
        assert resolve_deterministically("WidgetC", self._index()) is None

    def test_genuinely_unknown_items(self):
        for name in ("SuperGizmo", "MegaSprocket", "Consulting hours"):
            assert resolve_deterministically(name, self._index()) is None


class TestNormalizeInvoice:
    def test_resolves_without_a_model(self):
        """Deterministic matching handles every name in the corpus. The model is a
        fallback, not the mechanism."""
        invoice = normalize_invoice(
            invoice_with("WidgetA (rush order)", "Widget A", "Gadget X"), CATALOG
        )

        assert [line.item for line in invoice.line_items] == [
            "WidgetA",
            "WidgetA",
            "GadgetX",
        ]

    def test_raw_names_are_preserved(self):
        """The canonical name drives the stock check; the raw name is what a human
        reads when tracing the decision."""
        invoice = normalize_invoice(invoice_with("WidgetA (rush order)"), CATALOG)

        assert invoice.line_items[0].raw_name == "WidgetA (rush order)"
        assert invoice.line_items[0].item == "WidgetA"

    def test_unknown_items_stay_unresolved(self):
        invoice = normalize_invoice(invoice_with("SuperGizmo"), CATALOG)
        assert invoice.line_items[0].item is None

    def test_no_model_call_when_everything_resolves(self):
        client = FakeLLM()

        normalize_invoice(invoice_with("WidgetA", "WidgetB"), CATALOG, client=client)

        assert client.call_count == 0

    def test_one_call_for_all_unresolved_names(self):
        """One call per invoice, not per line."""
        client = FakeLLM(
            ItemMatches(
                matches=[
                    ItemMatch(raw_name="WIDGET_A DELUXE", item=None, reasoning="unsure"),
                    ItemMatch(raw_name="SuperGizmo", item=None, reasoning="not in catalog"),
                ]
            )
        )

        normalize_invoice(
            invoice_with("WIDGET_A DELUXE", "SuperGizmo", "WidgetA"),
            CATALOG,
            client=client,
        )

        assert client.call_count == 1

    def test_a_model_match_is_applied(self):
        client = FakeLLM(
            ItemMatches(
                matches=[
                    ItemMatch(
                        raw_name="Item: WidgetA, blue",
                        item="WidgetA",
                        reasoning="Same product with a colour note.",
                    )
                ]
            )
        )

        invoice = normalize_invoice(
            invoice_with("Item: WidgetA, blue"), CATALOG, client=client
        )

        assert invoice.line_items[0].item == "WidgetA"

    def test_a_model_cannot_invent_a_catalog_entry(self):
        """The catalog is the authority for what exists.

        A model returning "WidgetD" would otherwise add a product to inventory by
        asserting one, which is exactly the thing a vendor would like to be able to do.
        """
        client = FakeLLM(
            ItemMatches(
                matches=[
                    ItemMatch(raw_name="WidgetD", item="WidgetD", reasoning="looks fine")
                ]
            )
        )

        invoice = normalize_invoice(invoice_with("WidgetD"), CATALOG, client=client)

        assert invoice.line_items[0].item is None

    def test_an_unreachable_model_leaves_names_unresolved(self):
        """Degrading toward caution. An unresolved name becomes UNKNOWN_ITEM and the
        invoice reaches a human — the same outcome as a genuine non-match."""
        from galatiq.llm import LLMResponseError

        client = FakeLLM(LLMResponseError("malformed"))

        invoice = normalize_invoice(invoice_with("SuperGizmo"), CATALOG, client=client)

        assert invoice.line_items[0].item is None
