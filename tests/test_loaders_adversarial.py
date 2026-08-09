"""Robustness tests against formats the provided corpus does not contain.

Every file in data/invoices is a format this system has a parser for, so nothing in
there tests the case that matters most for a workflow whose invoices arrive by email
from arbitrary vendors: a document in a shape nobody anticipated.

These fixtures are mine, and they exist to make the claim falsifiable:

    No input crashes the pipeline. Every input produces a decision with reasoning.
"""

from pathlib import Path

import pytest

from galatiq.config import PROJECT_ROOT
from galatiq.loaders import discover, load, load_many
from galatiq.models import FindingCode, Severity

ADVERSARIAL = PROJECT_ROOT / "data" / "adversarial"


class TestUnanticipatedFormat:
    """A .yaml invoice. There is no YAML loader, and there is not going to be one."""

    def test_it_loads(self):
        doc = load(ADVERSARIAL / "invoice_A001.yaml")
        assert doc.is_readable
        assert doc.source_format == "yaml"

    def test_content_reaches_the_extractor_intact(self):
        """The extractor gets the same thing it would get from a .txt invoice --
        text. Format was never the thing that made a document readable."""
        text = load(ADVERSARIAL / "invoice_A001.yaml").raw_text

        assert "INV-A001" in text
        assert "Cascade Fabrication Co." in text
        assert "WidgetA" in text

    def test_no_structural_hint(self):
        assert not load(ADVERSARIAL / "invoice_A001.yaml").has_hint

    def test_missing_cross_check_is_recorded(self):
        """WARN rather than silence. The document was handled, but through a path
        with no independent second reading -- so the extraction has nothing to be
        verified against, and a reviewer weighing the decision should know that.
        """
        findings = load(ADVERSARIAL / "invoice_A001.yaml").findings

        assert len(findings) == 1
        assert findings[0].code == FindingCode.UNSUPPORTED_FORMAT
        assert findings[0].severity == Severity.WARN
        assert "invoice_A001.yaml" in findings[0].evidence


class TestUnreadableDocument:
    """A binary file. Someone photographed an invoice and sent the image."""

    def test_it_does_not_raise(self):
        doc = load(ADVERSARIAL / "invoice_A002.bin")
        assert doc.source_path.endswith("invoice_A002.bin")

    def test_it_is_reported_as_unreadable(self):
        """Critical, and named. Without this the file becomes an invoice that
        mysteriously has no content, and the rejection reason would be wrong."""
        doc = load(ADVERSARIAL / "invoice_A002.bin")

        assert not doc.is_readable
        assert doc.raw_text == ""
        assert doc.findings[0].code == FindingCode.UNREADABLE_DOCUMENT
        assert doc.findings[0].severity == Severity.CRITICAL
        assert "invoice_A002.bin" in doc.findings[0].evidence

    def test_decoding_is_not_evidence_of_text(self):
        """latin-1 decodes any byte sequence, so "it decoded" proves nothing.

        The printable-character ratio is what actually separates a document from a
        PNG someone renamed.
        """
        assert load(ADVERSARIAL / "invoice_A002.bin").raw_text == ""


class TestBatchIsolation:
    def test_one_bad_file_does_not_stop_the_batch(self):
        """The whole point of findings-instead-of-exceptions.

        The directory holds a readable YAML invoice, an unreadable binary, and a
        README. All three come back; none of them raises.
        """
        docs = load_many(ADVERSARIAL)

        assert len(docs) == len(list(discover(ADVERSARIAL)))
        assert any(doc.is_readable for doc in docs)
        assert any(not doc.is_readable for doc in docs)

    @pytest.mark.parametrize(
        "path", sorted(ADVERSARIAL.iterdir()), ids=lambda p: p.name
    )
    def test_every_file_produces_a_document(self, path: Path):
        """Including this directory's own README.

        A markdown file is not an invoice, and the system handles that the way it
        handles everything else: it reads it, finds no invoice in it, and says so.
        Not by crashing.
        """
        doc = load(path)
        assert doc.source_path == str(path)
        assert doc.is_readable or doc.findings

    def test_unreadable_files_carry_their_reason(self):
        for doc in load_many(ADVERSARIAL):
            if not doc.is_readable:
                assert doc.findings, f"{doc.source_path} unreadable with no explanation"
                assert any(
                    f.severity == Severity.CRITICAL for f in doc.findings
                )


class TestNoExtension:
    def test_a_file_with_no_extension_still_loads(self, tmp_path):
        """Email attachments lose their extension constantly."""
        path = tmp_path / "invoice_attachment"
        path.write_text("INVOICE\nVendor: Acme\nTotal: $500.00\n")

        doc = load(path)
        assert doc.is_readable
        assert "Total: $500.00" in doc.raw_text
        assert doc.findings[0].code == FindingCode.UNSUPPORTED_FORMAT

    def test_markdown_invoice(self, tmp_path):
        """A format with no parser and a layout no parser would help with."""
        path = tmp_path / "invoice_md.md"
        path.write_text(
            "# Invoice INV-9100\n\n"
            "**Vendor:** Northwind Traders\n\n"
            "| Item | Qty | Unit Price |\n"
            "|------|-----|------------|\n"
            "| WidgetA | 3 | $250.00 |\n\n"
            "**Total:** $750.00\n"
        )

        doc = load(path)
        assert doc.is_readable
        assert "INV-9100" in doc.raw_text
        assert not doc.has_hint
