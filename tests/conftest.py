"""Shared test fixtures.

The important one is `FakeLLM`. Every agent test runs against it rather than a real
endpoint, which keeps the suite free, deterministic, and runnable with no key -- and
means what gets tested is our logic rather than xAI's uptime.
"""

from types import SimpleNamespace

import pytest

from galatiq.agents import Critique
from galatiq.llm import LLMResult
from galatiq.models import Invoice, LineItem


class FakeLLM:
    """A scripted stand-in for an LLM client.

    Responses are queued and returned in order. A queued exception is raised instead,
    which is how the malformed-response paths get exercised without needing a model to
    actually misbehave.

    Running out of queued responses is an error rather than a default: a test that
    makes more calls than it scripted has found a real behaviour change, and silently
    handing back something plausible would hide it.
    """

    provider = "fake"
    model = "fake-model-1"

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def complete(self, messages, response_model, *, temperature=None, exclude=None):
        self.calls.append(
            SimpleNamespace(
                messages=messages,
                response_model=response_model,
                exclude=exclude,
                system=messages[0].content if messages else "",
                prompt=messages[-1].content if messages else "",
            )
        )

        if not self.queue:
            raise AssertionError(
                f"FakeLLM: unscripted call #{len(self.calls)} for "
                f"{response_model.__name__}"
            )

        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item

        return LLMResult(
            value=item,
            provider=self.provider,
            model=self.model,
            latency_ms=1,
            prompt_tokens=10,
            completion_tokens=5,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def sample_invoice():
    """A clean, complete extraction — INV-1001's shape."""
    return Invoice(
        invoice_number="INV-1001",
        vendor="Widgets Inc.",
        issue_date_raw="2026-01-15",
        due_date_raw="2026-02-01",
        line_items=[
            LineItem(raw_name="WidgetA", quantity=10, unit_price="250.00"),
            LineItem(raw_name="WidgetB", quantity=5, unit_price="500.00"),
        ],
        subtotal="5000.00",
        tax_amount="0.00",
        total="5000.00",
        currency="USD",
        payment_terms="Net 15",
    )


@pytest.fixture
def sound():
    return Critique(verdict="PARSE_SOUND", reasoning="Transcription matches.")


@pytest.fixture
def inconsistent():
    return Critique(
        verdict="DOCUMENT_INCONSISTENT",
        reasoning="Stated subtotal does not match the stated line items.",
    )
