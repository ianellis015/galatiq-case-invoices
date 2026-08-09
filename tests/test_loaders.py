"""Tests for the format loaders.

Run against the real corpus rather than synthetic fixtures. These files are the
specification -- a loader passing on invented CSVs and failing on INV-1006 has tested
nothing worth testing.

Two themes. The traps get the most attention, because all three fail *silently*: a
naive parser returns an invoice, just not the one in the file. And the invariant gets
its own class, because "no input crashes the pipeline" is the claim the whole design
rests on.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from galatiq.config import PROJECT_ROOT
from galatiq.loaders import discover, load, load_many
from galatiq.models import FindingCode, Severity

INVOICES = PROJECT_ROOT / "data" / "invoices"
CORPUS = sorted(INVOICES.iterdir())


class TestInvariant:
    """No input crashes the pipeline. Every input produces a document."""

    @pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.name)
    def test_every_corpus_file_loads(self, path: Path):
        doc = load(path)
        assert doc.source_path == str(path)
        assert doc.source_format == path.suffix.lstrip(".").lower()

    @pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.name)
    def test_every_corpus_file_yields_text(self, path: Path):
        """Text is the universal interface.

        Whatever the format, the extractor has something to read. That is what makes
        the system indifferent to layout -- and it costs nothing, because JSON, XML
        and CSV files already are text.
        """
        assert load(path).is_readable

    def test_only_a_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load(INVOICES / "invoice_9999.txt")

    def test_directory_is_not_a_file(self):
        with pytest.raises(FileNotFoundError):
            load(INVOICES)

    def test_load_many_over_the_directory(self):
        assert len(load_many(INVOICES)) == 20

    def test_clean_corpus_files_report_no_problems(self):
        """The provided corpus is all readable, recognized formats.

        A finding here would mean the loader had started inventing problems.
        """
        for doc in load_many(INVOICES):
            assert doc.findings == [], f"{doc.source_path}: {doc.findings}"


class TestStructuralHints:
    """Structured parsing is a cross-check, not a bypass."""

    @pytest.mark.parametrize(
        "name", ["invoice_1004.json", "invoice_1014.xml", "invoice_1006.csv"]
    )
    def test_recognized_shapes_produce_a_hint(self, name):
        doc = load(INVOICES / name)
        assert doc.has_hint
        assert doc.hint_source == doc.source_format

    @pytest.mark.parametrize("name", ["invoice_1001.txt", "invoice_1011.pdf"])
    def test_unstructured_formats_have_no_hint(self, name):
        doc = load(INVOICES / name)
        assert not doc.has_hint
        assert doc.hint_source is None

    def test_hint_accompanies_text_rather_than_replacing_it(self):
        """Both, always -- the hint is a second reading, not a substitute.

        Without the source text there is nothing for the extraction critic to
        compare a parse against, and `Finding.evidence` cannot quote the document.
        """
        doc = load(INVOICES / "invoice_1004.json")
        assert doc.is_readable
        assert doc.has_hint


class TestDiscovery:
    def test_directory_returns_every_file(self):
        """Twenty files, not sixteen.

        INV-1011, 1012 and 1013 each exist as a pair whose contents genuinely differ,
        so both members are real documents. Recognising that two describe the same
        invoice belongs to the dedupe check.
        """
        assert len(discover(INVOICES)) == 20

    def test_ordering_is_deterministic(self):
        assert discover(INVOICES) == sorted(discover(INVOICES))

    def test_glob(self):
        names = [p.name for p in discover(str(INVOICES / "*.csv"))]
        assert names == ["invoice_1006.csv", "invoice_1007.csv", "invoice_1015.csv"]

    def test_single_file(self):
        path = INVOICES / "invoice_1001.txt"
        assert discover(path) == [path]

    def test_extension_is_not_a_filter(self, tmp_path):
        """Filtering by extension here would reinstate the brittleness the fallback
        removes -- a directory of .yaml invoices would come back empty."""
        (tmp_path / "invoice_A.yaml").write_text("invoice_number: INV-A")
        (tmp_path / "invoice_B.txt").write_text("INVOICE")

        assert len(discover(tmp_path)) == 2

    def test_hidden_files_are_skipped(self, tmp_path):
        (tmp_path / "invoice.txt").write_text("INVOICE")
        (tmp_path / ".DS_Store").write_text("junk")

        assert [p.name for p in discover(tmp_path)] == ["invoice.txt"]

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            discover(INVOICES / "does_not_exist")


class TestTextLoader:
    def test_reads_verbatim(self):
        """No stripping or whitespace normalising -- evidence has to quote exactly."""
        doc = load(INVOICES / "invoice_1003.txt")
        assert doc.raw_text == (INVOICES / "invoice_1003.txt").read_text()

    def test_injection_text_survives(self):
        """INV-1003's manipulation language has to reach the fraud check intact.

        The loader treats it as bytes, not instructions -- the first place the
        untrusted-input boundary is either respected or broken.
        """
        text = load(INVOICES / "invoice_1003.txt").raw_text
        assert "URGENT" in text
        assert "Wire transfer preferred" in text

    def test_typod_labels_are_not_corrected(self):
        """INV-1002 has "INVOCE", "Vndr", "Itms". Repairing those is a judgement the
        extractor makes with the whole document in view."""
        assert "INVOCE" in load(INVOICES / "invoice_1002.txt").raw_text


class TestPdfLoader:
    @pytest.mark.parametrize(
        "stem,number",
        [("invoice_1011", "1011"), ("invoice_1012", "1012"), ("invoice_1013", "1013")],
    )
    def test_pdfs_yield_text(self, stem, number):
        doc = load(INVOICES / f"{stem}.pdf")
        assert doc.is_readable
        assert number in doc.raw_text

    def test_ocr_damage_survives(self):
        """INV-1012 has "2O26" -- a letter O standing in for a zero.

        Passing it through is the point. A loader that quietly fixed it would destroy
        the evidence the extraction critic needs, and would mean the system silently
        rewrote a document it moves money against.
        """
        assert "2O26" in load(INVOICES / "invoice_1012.pdf").raw_text

    def test_invoice_number_formatting_is_not_repaired(self):
        """INV-1012's PDF says "INV NO: INV 1012" -- a space where every other
        document has a hyphen. Same class as INV-1002's missing prefix."""
        text = load(INVOICES / "invoice_1012.pdf").raw_text
        assert "INV 1012" in text
        assert "INV-1012" not in text

    def test_twins_differ(self):
        """INV-1011 exists as PDF and txt, and they are not the same document."""
        assert (
            load(INVOICES / "invoice_1011.pdf").raw_text
            != load(INVOICES / "invoice_1011.txt").raw_text
        )

    def test_unreadable_pdf_is_a_finding_not_a_crash(self, tmp_path):
        """A corrupt or encrypted PDF must not stop a batch of twenty."""
        broken = tmp_path / "invoice_broken.pdf"
        broken.write_bytes(b"%PDF-1.4\nnot actually a pdf")

        doc = load(broken)
        assert not doc.is_readable
        assert doc.findings[0].code == FindingCode.UNREADABLE_DOCUMENT
        assert doc.findings[0].severity == Severity.CRITICAL


class TestJsonLoader:
    def test_numbers_are_decimal_not_float(self):
        """parse_float=Decimal is what makes JSON invoices usable at all.

        Without it json produces floats, floats are rejected at the model boundary,
        and every JSON invoice fails to build an Invoice. Converting afterwards is
        too late -- the precision is already gone.
        """
        hint = load(INVOICES / "invoice_1004.json").structural_hint

        assert isinstance(hint["subtotal"], Decimal)
        assert hint["subtotal"] == Decimal("1750.00")
        assert isinstance(hint["line_items"][0]["unit_price"], Decimal)

    def test_vendor_object_is_flattened(self):
        hint = load(INVOICES / "invoice_1004.json").structural_hint

        assert hint["vendor"] == "Precision Parts Ltd."
        assert "Springfield" in hint["vendor_address"]

    def test_empty_vendor_survives(self):
        """INV-1009's vendor name is "". That becomes a finding later, which it can
        only do if it is recorded faithfully now."""
        hint = load(INVOICES / "invoice_1009.json").structural_hint

        assert hint["vendor"] == ""
        assert hint["vendor_address"] is None
        assert hint["due_date"] is None

    def test_negative_quantity_survives(self):
        hint = load(INVOICES / "invoice_1009.json").structural_hint
        assert hint["line_items"][0]["quantity"] == -5

    def test_revision_marker_is_kept(self):
        original = load(INVOICES / "invoice_1004.json").structural_hint
        revised = load(INVOICES / "invoice_1004_revised.json").structural_hint

        assert original["invoice_number"] == revised["invoice_number"] == "INV-1004"
        assert "revision" not in original
        assert revised["revision"] == "R1"
        assert revised["total"] != original["total"]

    def test_line_item_notes_survive(self):
        hint = load(INVOICES / "invoice_1013.json").structural_hint
        assert "Volume discount" in [li.get("note") for li in hint["line_items"]]

    def test_malformed_json_degrades_to_text(self, tmp_path):
        """A truncated file still reaches the extractor, which can often read what is
        there. No hint, no crash, and an INFO note explaining the missing
        cross-check."""
        path = tmp_path / "invoice_truncated.json"
        path.write_text('{"invoice_number": "INV-9001", "line_items": [')

        doc = load(path)
        assert doc.is_readable
        assert not doc.has_hint
        assert doc.findings[0].severity == Severity.INFO


class TestXmlLoader:
    def test_sections_are_flattened(self):
        """<header> and <totals> are presentation, not content."""
        hint = load(INVOICES / "invoice_1014.xml").structural_hint

        assert hint["invoice_number"] == "INV-1014"
        assert hint["vendor"] == "TechParts International"
        assert hint["subtotal"] == "3750.00"
        assert hint["total"] == "4125.00"
        assert "header" not in hint
        assert "totals" not in hint

    def test_currency_is_preserved(self):
        """The only non-USD invoice. A loader dropping this would hand a EUR total
        to a USD threshold rule."""
        assert load(INVOICES / "invoice_1014.xml").structural_hint["currency"] == "EUR"

    def test_line_item_name_becomes_item(self):
        lines = load(INVOICES / "invoice_1014.xml").structural_hint["line_items"]

        assert len(lines) == 2
        assert lines[0] == {"item": "WidgetA", "quantity": "4", "unit_price": "225.00"}

    def test_unrecognized_schema_degrades_to_text(self, tmp_path):
        """This parser knows INV-1014's schema. A UBL or cXML invoice produces no
        hint and takes the text route -- the model is the generalization mechanism,
        not more parser cases."""
        path = tmp_path / "invoice_ubl.xml"
        path.write_text(
            '<?xml version="1.0"?><Invoice xmlns="urn:oasis:ubl">'
            "<ID>INV-9002</ID><LegalMonetaryTotal><PayableAmount>500.00"
            "</PayableAmount></LegalMonetaryTotal></Invoice>"
        )

        doc = load(path)
        assert doc.is_readable
        assert not doc.has_hint
        assert doc.findings[0].code == FindingCode.UNSUPPORTED_FORMAT


class TestCsvVertical:
    """INV-1006 -- the repeated-key trap."""

    def test_both_line_items_survive(self):
        """`dict(rows)` keeps the last value for each duplicated key, silently
        discarding WidgetA and leaving an invoice whose line items no longer sum to
        its own subtotal. Nothing raises; the invoice becomes a different invoice.
        """
        lines = load(INVOICES / "invoice_1006.csv").structural_hint["line_items"]

        assert len(lines) == 2
        assert lines[0] == {"item": "WidgetA", "quantity": "5", "unit_price": "250.00"}
        assert lines[1] == {"item": "WidgetB", "quantity": "3", "unit_price": "500.00"}

    def test_quantities_stay_with_their_own_item(self):
        """Position in the file is the only signal separating WidgetA's quantity from
        WidgetB's -- the keys are identical."""
        lines = load(INVOICES / "invoice_1006.csv").structural_hint["line_items"]
        assert [line["quantity"] for line in lines] == ["5", "3"]

    def test_invoice_level_fields(self):
        hint = load(INVOICES / "invoice_1006.csv").structural_hint

        assert hint["invoice_number"] == "INV-1006"
        assert hint["vendor"] == "Acme Industrial Supplies"
        assert hint["subtotal"] == "2750.00"
        assert hint["tax_amount"] == "0.00"
        assert hint["payment_terms"] == "Net 15"

    def test_line_items_do_not_leak_into_invoice_fields(self):
        hint = load(INVOICES / "invoice_1006.csv").structural_hint
        assert "item" not in hint
        assert "quantity" not in hint


class TestCsvRowPerItem:
    """INV-1007 and INV-1015 -- the trailing summary row trap."""

    @pytest.mark.parametrize("stem", ["invoice_1007", "invoice_1015"])
    def test_summary_rows_are_not_line_items(self, stem):
        """Treated as data rows, the footer grows three phantom line items with no
        name, quantity or price.

        INV-1015 is documented as clean and has them too, so this is the normal shape
        of a row-per-item sheet rather than a quirk of INV-1007.
        """
        lines = load(INVOICES / f"{stem}.csv").structural_hint["line_items"]

        assert len(lines) == 3
        assert all(line.get("item") for line in lines)

    def test_summary_rows_become_totals(self):
        hint = load(INVOICES / "invoice_1007.csv").structural_hint

        assert hint["subtotal"] == "14750.00"
        assert hint["tax_amount"] == "885.00"
        assert hint["total"] == "15525.00"

    def test_tax_label_is_kept_verbatim(self):
        """"Tax (6%)" carries a rate check_math can use. Deriving it is
        interpretation, so the parser records what the document said and stops."""
        assert load(INVOICES / "invoice_1007.csv").structural_hint["tax_label"] == "Tax (6%):"

    def test_line_items_are_complete(self):
        lines = load(INVOICES / "invoice_1007.csv").structural_hint["line_items"]

        assert lines[0] == {
            "item": "WidgetA",
            "quantity": "20",
            "unit_price": "250.00",
            "amount": "5000.00",
        }

    def test_repeated_invoice_fields_are_read_once(self):
        hint = load(INVOICES / "invoice_1007.csv").structural_hint

        assert hint["invoice_number"] == "INV-1007"
        assert hint["vendor"] == "MegaWidgets Corp"
        assert hint["date"] == "01/28/2026"

    def test_header_aliases(self):
        """"Qty" -> quantity, "Line Total" -> amount."""
        lines = load(INVOICES / "invoice_1015.csv").structural_hint["line_items"]
        assert "quantity" in lines[0]
        assert "amount" in lines[0]
        assert "qty" not in lines[0]


class TestCsvDegradation:
    """A parser that knows two shapes should say so, not guess at a third."""

    def test_unrecognized_layout_produces_no_hint(self, tmp_path):
        """Different column names and no invoice-number column.

        Passing on a tidy dict with no invoice number would be worse than passing
        nothing: it looks like a reading of the document, and the downstream mismatch
        gets misdiagnosed as arithmetic rather than as a parse failure.
        """
        path = tmp_path / "invoice_odd.csv"
        path.write_text(
            "Description,Units,Rate,Extended\n"
            "WidgetA,4,250.00,1000.00\n"
            "WidgetB,2,500.00,1000.00\n"
        )

        doc = load(path)
        assert doc.is_readable
        assert not doc.has_hint
        assert doc.findings[0].code == FindingCode.UNSUPPORTED_FORMAT

    def test_empty_file_is_a_finding_not_a_crash(self, tmp_path):
        empty = tmp_path / "invoice_empty.csv"
        empty.write_text("")

        doc = load(empty)
        assert not doc.is_readable
        assert doc.findings[0].code == FindingCode.UNREADABLE_DOCUMENT

    def test_blank_rows_are_skipped(self, tmp_path):
        path = tmp_path / "invoice_blanks.csv"
        path.write_text(
            "field,value\n"
            "invoice_number,INV-9001\n"
            "\n"
            "item,WidgetA\n"
            "quantity,2\n"
            "\n"
            "unit_price,250.00\n"
        )
        hint = load(path).structural_hint

        assert hint["invoice_number"] == "INV-9001"
        assert hint["line_items"] == [
            {"item": "WidgetA", "quantity": "2", "unit_price": "250.00"}
        ]


class TestEncoding:
    def test_non_utf8_text_is_read_not_rejected(self, tmp_path):
        """A latin-1 invoice is a real thing, and read_text() dies on it with a byte
        offset instead of a description."""
        path = tmp_path / "invoice_latin1.txt"
        path.write_bytes("Vendor: Bäcker & Söhne GmbH\nTotal: 500.00\n".encode("latin-1"))

        doc = load(path)
        assert doc.is_readable
        assert "Total: 500.00" in doc.raw_text
        assert doc.findings[0].severity == Severity.INFO
