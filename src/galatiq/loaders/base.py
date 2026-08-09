"""What a loader returns, and the one rule every loader obeys.

    No input crashes the pipeline. Every input produces a decision with reasoning.

That invariant is why nothing in this package raises for a content problem. An
unreadable file, an unknown extension, a malformed CSV -- each comes back as a
`LoadedDocument` carrying a `Finding` that explains itself. A stack trace tells the
person running a batch of twenty invoices nothing useful; a rejection naming the file
and the reason tells them everything.

**Text is the universal interface.** Every document produces `raw_text`, always. That
is what makes the system indifferent to format: an invoice in a layout nobody
anticipated is still text, and reading meaning out of text is what the extractor
does. It costs nothing to guarantee, because JSON, XML and CSV files already *are*
text.

**Structured parsing is a cross-check, not a bypass.** When a parser recognizes a
shape it produces a `structural_hint` -- a second, independent reading of the same
document. The extractor gets it as context, and the critic gets it as something to
disagree with: when a deterministic parse says the subtotal is 21040.00 and the model
says 21400.00, that disagreement is a caught misparse. When no parser recognizes the
shape the hint is simply absent and extraction proceeds from text, so degradation
needs no detection logic and no branch that can be wrong.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from galatiq.models import Finding, FindingCode, Severity

# The key set structural parsers aim for. Modelled on the JSON invoices, which are the
# most explicit documents in the corpus.
#
# Absent keys stay absent rather than being filled with defaults: a document stating
# no subtotal (INV-1003) differs from one stating a subtotal of zero, and collapsing
# them would erase a finding.
CANONICAL_FIELDS = (
    "invoice_number",
    "revision",
    "vendor",
    "vendor_address",
    "date",
    "due_date",
    "line_items",     # list of {item, quantity, unit_price, amount?, note?}
    "subtotal",
    "tax_rate",
    "tax_amount",
    "total",
    "currency",
    "payment_terms",
    "notes",
)

# Extensions with a dedicated structural parser. Anything else still loads -- it just
# arrives as text with no hint.
STRUCTURED_EXTENSIONS = frozenset({".json", ".xml", ".csv"})
TEXT_EXTENSIONS = frozenset({".txt"})
BINARY_EXTENSIONS = frozenset({".pdf"})

# Below this proportion of printable characters, a successful decode is meaningless --
# latin-1 decodes any byte sequence, so "it decoded" is not evidence of text.
_PRINTABLE_THRESHOLD = 0.85

# Tried in order. latin-1 never fails, so it is the backstop and the printable-ratio
# check is what actually distinguishes text from binary.
_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


class LoadedDocument(BaseModel):
    """One file, read.

    `raw_text` is always populated for a readable document -- that is the guarantee
    the rest of the system is built on. `structural_hint` is a bonus when a parser
    recognized the shape.
    """

    model_config = ConfigDict(extra="forbid")

    # Both feed `Invoice.source_path` / `source_format`. INV-1011, 1012 and 1013 each
    # exist in two formats whose contents differ, so "which invoice" is not a specific
    # enough answer for an audit trail -- a finding has to trace to a file.
    source_path: str
    source_format: str

    raw_text: str = ""

    structural_hint: dict[str, Any] | None = None
    hint_source: str | None = None

    # Problems found while reading. These travel with the document into the same
    # reporting channel as every validation finding, rather than into an exception
    # handler somewhere else.
    findings: list[Finding] = Field(default_factory=list)

    @property
    def is_readable(self) -> bool:
        """False for a binary file or a PDF with no text layer."""
        return bool(self.raw_text.strip())

    @property
    def has_hint(self) -> bool:
        """True when a structural parser recognized this document's shape."""
        return self.structural_hint is not None


def read_text_safely(path: Path) -> tuple[str, list[Finding]]:
    """Decode a file to text, reporting rather than raising when it cannot.

    Returns the text (empty if the file is not text) and any findings raised while
    reading.

    Three cases worth naming:

      * **Non-UTF-8 text.** A latin-1 invoice is a real thing, and the default
        `read_text()` dies on it with a byte offset instead of a description. Falling
        back is an INFO finding, not a failure.
      * **Binary content.** latin-1 decodes anything, so a successful decode proves
        nothing. The printable-character ratio is what actually separates a document
        from a JPEG someone renamed.
      * **Empty file.** Reported, not silently treated as an invoice with no content.
    """
    data = path.read_bytes()

    if not data:
        return "", [
            Finding(
                code=FindingCode.UNREADABLE_DOCUMENT,
                severity=Severity.CRITICAL,
                message="File is empty.",
                evidence=str(path),
            )
        ]

    findings: list[Finding] = []
    text = ""
    used = ""

    for encoding in _ENCODINGS:
        try:
            text = data.decode(encoding)
            used = encoding
            break
        except UnicodeDecodeError:
            continue

    if used != "utf-8":
        findings.append(
            Finding(
                code=FindingCode.UNSUPPORTED_FORMAT,
                severity=Severity.INFO,
                message=f"File is not UTF-8; decoded as {used}.",
                evidence=str(path),
            )
        )

    if _printable_ratio(text) < _PRINTABLE_THRESHOLD:
        return "", [
            Finding(
                code=FindingCode.UNREADABLE_DOCUMENT,
                severity=Severity.CRITICAL,
                message=(
                    "File does not appear to be a text document "
                    "(mostly non-printable bytes)."
                ),
                evidence=str(path),
            )
        ]

    return text, findings


def _printable_ratio(text: str) -> float:
    """Proportion of characters that are printable or ordinary whitespace."""
    if not text:
        return 0.0

    printable = sum(1 for char in text if char.isprintable() or char in "\n\r\t")
    return printable / len(text)


def hint_is_useful(hint: dict[str, Any] | None) -> bool:
    """Did a structural parse actually recognize this document?

    A parser can succeed mechanically and understand nothing -- a CSV whose columns
    are all named differently produces a tidy dict with no invoice number and no line
    items. Passing that on as a hint is worse than passing nothing: it looks like a
    reading of the document, and the downstream mismatch gets misdiagnosed as an
    arithmetic problem rather than a parse failure.

    An invoice number or at least one line item is the bar for "recognized".
    """
    if not hint:
        return False

    return bool(hint.get("invoice_number") or hint.get("line_items"))


def hint_unavailable(path: Path, reason: str) -> Finding:
    """A structural parse failed, and extraction will proceed from text instead.

    INFO rather than WARN on purpose. This is the fallback working: the document is
    still fully processable, it just takes the same route a plain text invoice takes.
    Only a document that cannot be read at all is a real problem.
    """
    return Finding(
        code=FindingCode.UNSUPPORTED_FORMAT,
        severity=Severity.INFO,
        message=f"Structural pre-parse unavailable ({reason}); using text extraction.",
        evidence=str(path),
    )
