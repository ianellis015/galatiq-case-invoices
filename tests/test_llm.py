"""Tests for the xAI adapter.

A test double is injected in place of the OpenAI client, so the suite needs no key, no
network, and no HTTP mocking library. What is worth testing here is the adapter's own
logic -- schema construction, retry behaviour, validation, error translation -- and
none of that requires a real endpoint.

The error translation gets the most attention. Three distinct exception types exist
because three different callers handle them, and collapsing them would mean the
extraction retry loop could not tell "the network blipped" from "the model misread the
document".
"""

from decimal import Decimal
from types import SimpleNamespace

import httpx
import openai
import pytest

from galatiq.llm import (
    LLMConfigError,
    LLMResponseError,
    LLMTransportError,
    get_client,
    system,
    user,
)
from galatiq.llm.xai import XAIClient
from galatiq.models import Invoice

VALID_JSON = (
    '{"invoice_number": "INV-1001", "revision": null, "vendor": "Widgets Inc.", '
    '"vendor_address": null, "issue_date_raw": "2026-01-15", "issue_date": null, '
    '"due_date_raw": "2026-02-01", "due_date": null, '
    '"line_items": [{"raw_name": "WidgetA", "item": null, "quantity": 10, '
    '"quantity_raw": "10", "unit_price": "250.00", "stated_amount": null, '
    '"note": null}], '
    '"subtotal": "5000.00", "tax_rate": "0.0", "tax_amount": "0.00", '
    '"total": "5000.00", "currency": "USD", "payment_terms": "Net 15", '
    '"notes": null}'
)


def _response(content=None, *, refusal=None, tokens=(120, 45)):
    """Mimic the shape the adapter reads off a chat completion."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=refusal)
            )
        ],
        usage=SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1]),
    )


class FakeCompletions:
    """Returns queued responses, or raises queued exceptions, in order."""

    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.queue.pop(0) if self.queue else _response(VALID_JSON)
        if isinstance(item, Exception):
            raise item
        return item


def fake_client(*queue):
    completions = FakeCompletions(queue)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def build(*queue, **kwargs):
    client, completions = fake_client(*queue)
    adapter = XAIClient(
        api_key="",
        base_url="https://api.x.ai/v1",
        model="grok-4.5",
        client=client,
        **kwargs,
    )
    return adapter, completions


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Backoff is real; waiting for it in a test suite is not."""
    monkeypatch.setattr("galatiq.llm.xai.time.sleep", lambda _: None)


class TestTypedOutput:
    def test_returns_a_validated_model(self):
        adapter, _ = build(_response(VALID_JSON))

        result = adapter.complete([user("read this")], Invoice)

        assert isinstance(result.value, Invoice)
        assert result.value.invoice_number == "INV-1001"
        assert result.value.total == Decimal("5000.00")

    def test_money_arrives_as_decimal(self):
        """The whole point of transcribing amounts as strings: they survive the model
        boundary exactly, and become Decimal on the way in."""
        adapter, _ = build(_response(VALID_JSON))

        line = adapter.complete([user("x")], Invoice).value.line_items[0]

        assert line.unit_price == Decimal("250.00")
        assert isinstance(line.unit_price, Decimal)

    def test_result_carries_provenance(self):
        """"What decided to pay this" has to be a stored fact -- these two fields are
        the ledger columns waiting for them."""
        adapter, _ = build(_response(VALID_JSON))

        result = adapter.complete([user("x")], Invoice)

        assert result.provider == "xai"
        assert result.model == "grok-4.5"
        assert result.latency_ms >= 0
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 45


class TestRequestConstruction:
    def test_messages_are_passed_through_in_order(self):
        adapter, completions = build(_response(VALID_JSON))

        adapter.complete([system("instructions"), user("document")], Invoice)

        assert completions.calls[0]["messages"] == [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "document"},
        ]

    def test_strict_schema_is_sent(self):
        adapter, completions = build(_response(VALID_JSON))

        adapter.complete([user("x")], Invoice)
        payload = completions.calls[0]["response_format"]

        assert payload["json_schema"]["strict"] is True
        assert payload["json_schema"]["name"] == "Invoice"

    def test_excluded_fields_are_not_offered_to_the_model(self):
        adapter, completions = build(_response(VALID_JSON))

        adapter.complete(
            [user("x")], Invoice, exclude=frozenset({"source_path", "extra"})
        )
        properties = completions.calls[0]["response_format"]["json_schema"]["schema"][
            "properties"
        ]

        assert "source_path" not in properties
        assert "extra" not in properties

    def test_temperature_defaults_to_zero(self):
        """Extraction is transcription, not composition. The same document should
        produce the same reading twice."""
        adapter, completions = build(_response(VALID_JSON))

        adapter.complete([user("x")], Invoice)

        assert completions.calls[0]["temperature"] == 0.0

    def test_temperature_can_be_overridden_per_call(self):
        adapter, completions = build(_response(VALID_JSON))

        adapter.complete([user("x")], Invoice, temperature=0.7)

        assert completions.calls[0]["temperature"] == 0.7


class TestResponseErrors:
    """Everything the retry loop will catch."""

    def test_malformed_json(self):
        adapter, _ = build(_response('{"invoice_number": "INV-1001"'))

        with pytest.raises(LLMResponseError) as exc:
            adapter.complete([user("x")], Invoice)

        assert exc.value.raw.startswith('{"invoice_number"')

    def test_schema_mismatch_names_the_field(self):
        """The detail is the point.

        Handing the model back "that was invalid" produces another guess. Handing it
        back "line_items.0.quantity: input should be a valid integer" produces a
        correction -- which is the difference between a retry and a re-roll.
        """
        adapter, _ = build(
            _response(
                '{"invoice_number": "INV-1001", "line_items": '
                '[{"raw_name": "WidgetA", "quantity": "ten"}]}'
            )
        )

        with pytest.raises(LLMResponseError) as exc:
            adapter.complete([user("x")], Invoice)

        assert "quantity" in exc.value.detail

    def test_float_money_is_rejected_at_the_boundary(self):
        """The guard from money.py, holding all the way out to the model.

        A model that returns 5000.00 as a JSON number rather than a string gets
        rejected rather than silently introducing float error into a payment.
        """
        adapter, _ = build(
            _response('{"invoice_number": "INV-1001", "total": 5000.00}')
        )

        with pytest.raises(LLMResponseError) as exc:
            adapter.complete([user("x")], Invoice)

        assert "total" in exc.value.detail

    def test_refusal_says_so(self):
        """A refusal is a deliberate non-answer, not a malformed one. It still cannot
        become an Invoice, but the message should not blame the schema."""
        adapter, _ = build(_response(None, refusal="I can't help with that."))

        with pytest.raises(LLMResponseError, match="declined"):
            adapter.complete([user("x")], Invoice)

    def test_empty_response(self):
        adapter, _ = build(_response("   "))

        with pytest.raises(LLMResponseError, match="empty"):
            adapter.complete([user("x")], Invoice)


class TestTransport:
    """Retried inside the adapter. Nothing above it should know the network exists."""

    def _connection_error(self):
        return openai.APIConnectionError(
            request=httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
        )

    def test_transient_failure_is_retried(self):
        adapter, completions = build(
            self._connection_error(), _response(VALID_JSON), max_retries=3
        )

        result = adapter.complete([user("x")], Invoice)

        assert result.value.invoice_number == "INV-1001"
        assert len(completions.calls) == 2

    def test_gives_up_after_the_budget(self):
        adapter, completions = build(
            self._connection_error(),
            self._connection_error(),
            self._connection_error(),
            max_retries=3,
        )

        with pytest.raises(LLMTransportError, match="3 attempts"):
            adapter.complete([user("x")], Invoice)

        assert len(completions.calls) == 3

    def test_status_errors_are_not_retried(self):
        """A bad key or a rejected schema fails identically on every attempt.

        Retrying only makes the error slower to arrive.
        """
        request = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
        error = openai.APIStatusError(
            "invalid api key",
            response=httpx.Response(401, request=request),
            body=None,
        )
        adapter, completions = build(error, max_retries=3)

        with pytest.raises(LLMTransportError, match="401"):
            adapter.complete([user("x")], Invoice)

        assert len(completions.calls) == 1


class TestConfiguration:
    def test_missing_key_fails_immediately(self):
        """Not partway through a batch.

        Discovering a missing key on invoice fourteen of twenty means fourteen
        invoices have to be reprocessed, and the error arrives long after its cause.
        """
        with pytest.raises(LLMConfigError, match="XAI_API_KEY"):
            XAIClient(api_key="", base_url="https://api.x.ai/v1", model="grok-4.5")

    def test_get_client_uses_an_explicit_key(self):
        client = get_client(api_key="test-key")

        assert client.provider == "xai"
        assert client.model == "grok-4.5"

    def test_get_client_without_a_key_fails(self, monkeypatch):
        monkeypatch.setattr("galatiq.llm.XAI_API_KEY", "")

        with pytest.raises(LLMConfigError):
            get_client()

    def test_model_can_be_overridden(self):
        assert get_client(api_key="k", model="grok-4-fast").model == "grok-4-fast"
