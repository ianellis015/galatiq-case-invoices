"""xAI Grok, through the OpenAI-compatible endpoint.

The brief's snippet -- `from xai import Grok` against `https://grok.x.ai` -- does not
describe a real SDK or a real endpoint. The working integration is the OpenAI client
pointed at `https://api.x.ai/v1`, which speaks the same protocol including structured
outputs.

Responses are validated here rather than through the SDK's `.parse()` helper, for two
reasons. The validation error becomes ours to shape, which is what lets the extraction
retry loop feed a specific field failure back to the model. And it does not depend on
xAI matching OpenAI's helper behaviour exactly -- the wire format is a much safer thing
to rely on than a convenience method.
"""

import time
from typing import Any

import openai
from pydantic import BaseModel, ValidationError

from galatiq.llm.base import (
    LLMConfigError,
    LLMResponseError,
    LLMResult,
    LLMTransportError,
    Message,
)
from galatiq.llm.schema import response_format

# Failures worth trying again: the network dropped, we were throttled, or the far end
# had a bad moment. Everything else -- a bad key, a malformed request -- will fail the
# same way on every attempt, and retrying it just makes the error slower.
_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class XAIClient:
    """An `LLMClient` backed by Grok.

    `client` exists for tests: injecting a double means the suite needs no key, no
    network, and no HTTP mocking library. The adapter's own logic -- schema
    construction, retry, validation, error translation -- is the part worth testing,
    and none of it requires a real endpoint.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        if client is None and not api_key:
            raise LLMConfigError(
                "No xAI API key. Set XAI_API_KEY in .env or in the environment."
            )

        self.provider = "xai"
        self.model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._client = client or openai.OpenAI(api_key=api_key, base_url=base_url)

    def complete[T: BaseModel](
        self,
        messages: list[Message],
        response_model: type[T],
        *,
        temperature: float | None = None,
        exclude: frozenset[str] | None = None,
    ) -> LLMResult[T]:
        """Send a conversation and return a validated `response_model`."""
        started = time.monotonic()

        raw_response = self._call_with_retry(
            messages=[{"role": m.role, "content": m.content} for m in messages],
            response_format=response_format(response_model, exclude=exclude),
            temperature=self._temperature if temperature is None else temperature,
        )

        latency_ms = int((time.monotonic() - started) * 1000)
        choice = raw_response.choices[0]

        # A refusal is a deliberate non-answer, not a malformed one. It still cannot
        # be turned into an Invoice, so it surfaces the same way -- but the message
        # says what actually happened rather than blaming the schema.
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise LLMResponseError(
                "Model declined to answer.", detail=refusal
            )

        content = choice.message.content or ""
        if not content.strip():
            raise LLMResponseError("Model returned an empty response.")

        try:
            value = response_model.model_validate_json(content)
        except ValidationError as exc:
            # The detail is the point. Handing the model back "that was invalid"
            # produces another guess; handing it back "line_items[0].quantity: input
            # should be a valid integer" produces a correction.
            raise LLMResponseError(
                f"Response did not match {response_model.__name__}.",
                detail=str(exc),
                raw=content,
            ) from exc

        usage = getattr(raw_response, "usage", None)

        return LLMResult(
            value=value,
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def _call_with_retry(self, **kwargs: Any) -> Any:
        """Call the endpoint, retrying transient failures with exponential backoff.

        Deliberately separate from the extraction retry loop. This one handles an
        unreliable network; that one handles a model that misread a document. Merging
        them would spend the extraction budget on connection errors and leave nothing
        for the problem it exists to solve.
        """
        last: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                return self._client.chat.completions.create(model=self.model, **kwargs)
            except _RETRYABLE as exc:
                last = exc
                if attempt < self._max_retries - 1:
                    time.sleep(2**attempt)
            except openai.APIStatusError as exc:
                # 4xx that is not a rate limit: a bad key, a rejected schema, a
                # malformed request. Retrying cannot help.
                raise LLMTransportError(
                    f"xAI rejected the request ({exc.status_code}): {exc}"
                ) from exc

        raise LLMTransportError(
            f"xAI unreachable after {self._max_retries} attempts: {last}"
        ) from last
