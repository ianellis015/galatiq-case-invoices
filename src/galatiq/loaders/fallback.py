"""Formats with no dedicated loader.

This is where the robustness claim is actually cashed. A `.yaml` invoice, a `.md`
table, an `.eml` export, a file with no extension at all -- none has a parser here,
and none needs one. If it decodes to text it goes to the extractor, which is the same
path a plain text invoice takes.

The alternative -- raising on an unrecognized extension -- would mean the system
handles exactly the five formats someone anticipated, and a sixth stops the batch.
For a workflow whose documents arrive by email from arbitrary vendors, that is the
wrong failure.

A genuinely unreadable file (a real PDF renamed to .txt, an image, a spreadsheet
binary) still cannot be processed. The difference is that it produces a
CRITICAL finding naming the file, and the invoice is held for a human rather than
taking the batch down.
"""

from pathlib import Path

from galatiq.loaders.base import LoadedDocument, read_text_safely
from galatiq.models import Finding, FindingCode, Severity


def load_unknown(path: Path) -> LoadedDocument:
    """Read a file of unrecognized format as text.

    The WARN finding is deliberate and stays on the record even when processing
    succeeds. The document *was* handled, but through a path with no structural
    cross-check -- so the extraction has nothing independent to be verified against,
    and a reviewer should know that when weighing the decision.
    """
    text, findings = read_text_safely(path)
    extension = path.suffix.lstrip(".").lower()

    if not text:
        # read_text_safely has already explained why.
        return LoadedDocument(
            source_path=str(path),
            source_format=extension or "unknown",
            raw_text="",
            findings=findings,
        )

    findings.append(
        Finding(
            code=FindingCode.UNSUPPORTED_FORMAT,
            severity=Severity.WARN,
            message=(
                f"No structural parser for '{path.suffix or 'no extension'}'; "
                f"processed as plain text without a cross-check."
            ),
            evidence=str(path),
        )
    )

    return LoadedDocument(
        source_path=str(path),
        source_format=extension or "unknown",
        raw_text=text,
        findings=findings,
    )
