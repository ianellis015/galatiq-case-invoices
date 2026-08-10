"""Tests for the extract node.

Everything runs against `FakeLLM`, so what is being checked is the extractor's own
behaviour — when it calls a model, what it sends, what it does with a malformed reply —
rather than whether a model is any good at reading invoices.
"""

from datetime import date

from galatiq.agents.extractor import extract_invoice
from galatiq.llm import AGENT_EXCLUDED_FIELDS, LLMResponseError
from galatiq.models import Invoice

from conftest import FakeLLM


class TestSuccessfulExtraction:
    def test_returns_the_invoice(self, sample_invoice):
        client = FakeLLM(sample_invoice)

        outcome = extract_invoice(client, raw_text="INVOICE\nTotal: $5,000")

        assert outcome.succeeded
        assert outcome.invoice.invoice_number == "INV-1001"

    def test_provenance_is_set_by_us(self, sample_invoice):
        """The model was never shown these fields — it has no way to know which file it
        is reading, and asking would only invite it to invent one."""
        client = FakeLLM(sample_invoice)

        outcome = extract_invoice(
            client,
            raw_text="INVOICE",
            source_path="data/invoices/invoice_1001.txt",
            source_format="txt",
        )

        assert outcome.invoice.source_path == "data/invoices/invoice_1001.txt"
        assert outcome.invoice.source_format == "txt"

    def test_provenance_is_excluded_from_the_schema(self, sample_invoice):
        client = FakeLLM(sample_invoice)

        extract_invoice(client, raw_text="INVOICE")

        assert client.calls[0].exclude == frozenset(AGENT_EXCLUDED_FIELDS)

    def test_the_hint_is_passed_as_context(self, sample_invoice):
        """The commitment from the loaders ticket: if the hint has no consumer, the
        structural parsers are dead weight."""
        client = FakeLLM(sample_invoice)

        extract_invoice(
            client,
            raw_text="INVOICE",
            structural_hint={"invoice_number": "INV-1006"},
        )

        assert "INV-1006" in client.calls[0].prompt


class TestDateParsing:
    def test_parsed_dates_are_filled_from_raw(self, sample_invoice):
        client = FakeLLM(sample_invoice)

        invoice = extract_invoice(client, raw_text="INVOICE").invoice

        assert invoice.issue_date == date(2026, 1, 15)
        assert invoice.due_date == date(2026, 2, 1)

    def test_raw_text_is_never_overwritten(self, sample_invoice):
        client = FakeLLM(sample_invoice)

        invoice = extract_invoice(client, raw_text="INVOICE").invoice

        assert invoice.due_date_raw == "2026-02-01"

    def test_unparseable_dates_stay_null_with_the_text_intact(self):
        """INV-1003's "yesterday". Both halves matter: the parsed field is null so the
        date check can report it, and the raw field holds what the vendor wrote so the
        finding can quote it."""
        client = FakeLLM(
            Invoice(invoice_number="INV-1003", due_date_raw="yesterday")
        )

        invoice = extract_invoice(client, raw_text="INVOICE").invoice

        assert invoice.due_date is None
        assert invoice.due_date_raw == "yesterday"


class TestMalformedResponses:
    def test_failure_comes_back_as_data(self):
        """Not an exception. The validation detail is what the next attempt needs, so
        it belongs in the return value where a routing function can act on it."""
        client = FakeLLM(
            LLMResponseError(
                "bad", detail="line_items.0.quantity: input should be a valid integer"
            )
        )

        outcome = extract_invoice(client, raw_text="INVOICE")

        assert not outcome.succeeded
        assert outcome.invoice is None
        assert "quantity" in outcome.validation_error

    def test_the_error_is_fed_back_on_the_next_attempt(self, sample_invoice):
        """Telling the model which field failed is the difference between a correction
        and a re-roll."""
        client = FakeLLM(sample_invoice)

        extract_invoice(
            client,
            raw_text="INVOICE",
            validation_error="total: input should be a valid string",
        )

        assert "total: input should be a valid string" in client.calls[0].prompt


class TestUnreadableDocuments:
    def test_no_model_call_is_made(self):
        """A binary file has nothing to transcribe, and spending an API call to
        discover that is waste."""
        client = FakeLLM()

        outcome = extract_invoice(client, raw_text="", source_format="bin")

        assert outcome.succeeded
        assert client.call_count == 0

    def test_an_empty_invoice_still_comes_back(self):
        """The document has to reach a decision. "We could not read this" is a decision
        a human can act on; a missing invoice is not."""
        client = FakeLLM()

        invoice = extract_invoice(
            client,
            raw_text="   \n  ",
            source_path="data/adversarial/invoice_A002.bin",
            source_format="bin",
        ).invoice

        assert invoice.invoice_number is None
        assert invoice.line_items == []
        assert invoice.source_path.endswith("invoice_A002.bin")


class TestInjectionBoundary:
    def test_manipulation_text_reaches_the_model_as_data(self):
        """INV-1003's urgency language has to be visible to the model — it is a fact
        about the invoice the fraud check will score — but fenced, so it cannot be read
        as an instruction."""
        client = FakeLLM(Invoice(invoice_number="INV-1003"))

        extract_invoice(
            client,
            raw_text="Notes: URGENT - Pay immediately!!! Wire transfer preferred.",
        )
        call = client.calls[0]

        assert "URGENT" in call.prompt
        assert "BEGIN UNTRUSTED DOCUMENT" in call.prompt
        assert "URGENT" not in call.system
