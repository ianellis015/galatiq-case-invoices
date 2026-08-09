"""Plain text invoices.

The simplest loader, and the one every other format degrades toward. Everything hard
about a .txt invoice -- typo'd labels ("INVOCE", "Vndr", "Itms"), an ASCII table,
email prose with no table at all, a due date of "yesterday" -- is ambiguity, and
ambiguity is the extractor's problem. This hands over the bytes unaltered.
"""

from pathlib import Path

from galatiq.loaders.base import LoadedDocument, read_text_safely


def load_text(path: Path) -> LoadedDocument:
    """Read a .txt invoice.

    No stripping, no whitespace normalising, no line-ending fixes. The extractor sees
    the document exactly as it exists on disk, and `Finding.evidence` can quote it
    verbatim.
    """
    text, findings = read_text_safely(path)

    return LoadedDocument(
        source_path=str(path),
        source_format="txt",
        raw_text=text,
        findings=findings,
    )
