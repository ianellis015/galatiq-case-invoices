"""PDF invoices, via pdfplumber.

The only format that is not already text, and so the only one where producing
`raw_text` takes real work.

Text extraction only -- no table detection. pdfplumber can find tables, but using it
would mean the PDF path produced structured data while the .txt path produced prose,
and the same invoice would travel two different routes depending on which twin it
arrived as. One route is easier to reason about and easier to test.

OCR damage passes through untouched. INV-1012 contains "2O26" and "$3,500.O0", where a
letter O stands in for a zero. Repairing that needs document context to justify it,
which the extractor has and this function does not.
"""

from pathlib import Path

import pdfplumber

from galatiq.loaders.base import LoadedDocument
from galatiq.models import Finding, FindingCode, Severity


def load_pdf(path: Path) -> LoadedDocument:
    """Extract text from every page, joined in page order.

    Two failure modes are reported rather than raised, because both are things real
    invoice inboxes contain:

      * **A scanned image with no text layer.** Extraction yields nothing. Without
        OCR there is nothing to process, and that has to be visible -- an empty
        string sailing onward becomes an invoice that mysteriously has no content.
      * **A corrupt or encrypted PDF.** pdfplumber raises; the batch should not stop
        because one of twenty files is damaged.

    Pages yielding no text contribute an empty string rather than being skipped, so a
    page break in the output corresponds to a page break in the document.
    """
    findings: list[Finding] = []
    pages: list[str] = []

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
    except Exception as exc:
        return LoadedDocument(
            source_path=str(path),
            source_format="pdf",
            raw_text="",
            findings=[
                Finding(
                    code=FindingCode.UNREADABLE_DOCUMENT,
                    severity=Severity.CRITICAL,
                    message=f"PDF could not be opened: {type(exc).__name__}.",
                    evidence=f"{path}: {exc}",
                )
            ],
        )

    text = "\n".join(pages)

    if not text.strip():
        findings.append(
            Finding(
                code=FindingCode.UNREADABLE_DOCUMENT,
                severity=Severity.CRITICAL,
                message=(
                    "PDF contains no extractable text -- likely a scanned image. "
                    "OCR would be required to process it."
                ),
                evidence=str(path),
            )
        )

    return LoadedDocument(
        source_path=str(path),
        source_format="pdf",
        raw_text=text,
        findings=findings,
    )
