"""The extractor: a document becomes an Invoice.

The one place in the system where a language model is asked to read something. Every
stage after this works on typed data, and every stage before it works on bytes.

Two things it deliberately does not do. It does not judge -- a document with a negative
quantity and a subtotal that contradicts its own line items is transcribed faithfully,
because a rejection with a reason is the product and a crash is not. And it does not
repair -- OCR damage, typo'd labels and inconsistent invoice numbers survive, because a
correction the system made silently is indistinguishable from a value that was always
right.
"""

from galatiq.agents.prompts import extraction_messages
from galatiq.dates import parse_invoice_dates
from galatiq.llm import AGENT_EXCLUDED_FIELDS, LLMClient, LLMResponseError
from galatiq.amounts import parse_amounts
from galatiq.models import Invoice


class ExtractionOutcome:
    """What one attempt produced.

    A small carrier rather than a tuple: three call sites read these fields, and
    `outcome.invoice` says more than `result[0]`.
    """

    __slots__ = ("invoice", "validation_error")

    def __init__(
        self,
        invoice: Invoice | None,
        validation_error: str | None = None,
    ) -> None:
        self.invoice = invoice
        self.validation_error = validation_error

    @property
    def succeeded(self) -> bool:
        return self.invoice is not None


def extract_invoice(
    client: LLMClient,
    *,
    raw_text: str,
    structural_hint: dict | None = None,
    validation_error: str | None = None,
    critique: object | None = None,
    source_path: str | None = None,
    source_format: str | None = None,
) -> ExtractionOutcome:
    """Transcribe a document into an Invoice.

    Returns an outcome rather than raising on a malformed response. The validation
    detail is what the next attempt needs -- telling the model *which field* failed is
    the difference between a correction and a re-roll -- so it comes back as data for
    the routing function to act on rather than as an exception to catch somewhere.

    An unreadable document never reaches the model. A binary file has nothing to
    transcribe, and spending an API call to discover that is waste; it comes back as an
    empty Invoice carrying its provenance, and the load findings already explain why.
    """
    if not raw_text.strip():
        return ExtractionOutcome(
            invoice=Invoice(source_path=source_path, source_format=source_format)
        )

    messages = extraction_messages(
        raw_text,
        structural_hint=structural_hint,
        validation_error=validation_error,
        critique=critique,
    )

    try:
        result = client.complete(
            messages, Invoice, exclude=frozenset(AGENT_EXCLUDED_FIELDS)
        )
    except LLMResponseError as exc:
        return ExtractionOutcome(
            invoice=None, validation_error=exc.detail or str(exc)
        )

    invoice = result.value

    # Provenance is ours. The model was never shown these fields -- it has no way to
    # know which file it is reading, and asking would only invite it to invent one.
    invoice = invoice.model_copy(
        update={"source_path": source_path, "source_format": source_format}
    )

    return ExtractionOutcome(invoice=parse_amounts(parse_invoice_dates(invoice)))

