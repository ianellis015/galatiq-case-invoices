"""Turning whatever a document called a date into a date, or admitting it isn't one.

Deliberately narrow. This does not decide whether a date is *reasonable* -- past due,
before the issue date, inconsistent with stated terms -- because those are business
judgements the checks make. Its only question is whether a string denotes a calendar
day.

The two failure cases are different and stay different. A document that states no due
date (INV-1009's `null`) and one that states "yesterday" (INV-1003) both end up with
`due_date is None`, and collapsing them would mean reporting "missing due date" on an
invoice that loudly stated one. That distinction survives because the raw text is kept
alongside the parsed value -- this function fills the second and never touches the
first.
"""

from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for the annotation
    from galatiq.models import Invoice

# Every layout the corpus actually uses, most-specific first.
#
# %m/%d/%Y before any day-first variant: INV-1007's "01/28/2026" is US convention, and
# the corpus is a US manufacturer's accounts payable. Ambiguous dates -- 03/04/2026 --
# are genuinely undecidable from the string alone, and guessing the other way would
# silently shift a due date by months. Recorded in the README as an assumption.
_FORMATS = (
    "%Y-%m-%d",       # 2026-01-24
    "%m/%d/%Y",       # 01/28/2026
    "%d-%b-%Y",       # 26-Jan-2026
    "%d %b %Y",       # 26 Jan 2026
    "%b %d, %Y",      # Jan 26, 2026
    "%B %d, %Y",      # January 26, 2026
    "%Y/%m/%d",       # 2026/01/24
    "%d.%m.%Y",       # 24.01.2026
)


def parse_date(text: str | None) -> date | None:
    """Parse a date string, returning None when it does not denote a day.

    None rather than an exception. An unparseable date is a normal property of a real
    invoice, not an error condition -- "yesterday" is a thing a vendor wrote, and the
    system's job is to reject the invoice with that reason rather than to crash while
    reading it.

    OCR damage is not repaired here either. INV-1012's "2O26" has a letter O where a
    zero belongs; this returns None, the raw text is preserved, and the extractor --
    which has the whole document in view -- is the thing entitled to decide what it
    was meant to say.
    """
    if text is None:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    for fmt in _FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue

    # ISO-8601 with a time component, which some exports produce.
    try:
        return datetime.fromisoformat(cleaned).date()
    except ValueError:
        return None


def parse_invoice_dates(invoice: "Invoice") -> "Invoice":
    """Fill an invoice's parsed date fields from its raw ones.

    Lives here rather than inside the extractor because two paths produce invoices --
    the model's transcription and a structural parser's reading -- and both need the
    same conversion. When only one of them did it, a structurally-parsed invoice reached
    the date check with `due_date` still null and was reported as having an unparseable
    due date of "2026-02-22". Found by running the corpus through the checks by hand.

    An already-parsed field is left alone, so a caller that supplies one keeps it.
    """
    return invoice.model_copy(
        update={
            "issue_date": invoice.issue_date or parse_date(invoice.issue_date_raw),
            "due_date": invoice.due_date or parse_date(invoice.due_date_raw),
        }
    )


def is_parseable(text: str | None) -> bool:
    """True when `text` denotes a day.

    Distinct from `parse_date(text) is not None` only in intent: this reads as a
    question at a call site that does not want the value.
    """
    return parse_date(text) is not None
