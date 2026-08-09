"""Calling a language model.

    from galatiq.llm import get_client, system, user

    client = get_client()
    result = client.complete(
        [system(INSTRUCTIONS), user(document_text)],
        Invoice,
        exclude=NOT_THE_MODELS_TO_FILL,
    )
    invoice = result.value          # a validated Invoice, not a string

One interface, one method. Every agent in the system calls through this, and no agent
anywhere parses free-form text out of a response.
"""

from galatiq.config import (
    LLM_MAX_RETRIES,
    LLM_TEMPERATURE,
    XAI_API_KEY,
    XAI_BASE_URL,
    XAI_MODEL,
)
from galatiq.llm.base import (
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMResponseError,
    LLMResult,
    LLMTransportError,
    Message,
    assistant,
    system,
    user,
)
from galatiq.llm.schema import build_strict_schema, response_format
from galatiq.llm.xai import XAIClient

# Fields no agent should be asked to fill. Provenance is ours -- we know which file we
# opened -- and `extra` is decided when a document is mapped onto the model, not by
# the model itself. Offering either to an LLM only invites invention.
AGENT_EXCLUDED_FIELDS = frozenset({"source_path", "source_format", "extra"})


def get_client(
    *,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> LLMClient:
    """Build the configured client.

    Raises `LLMConfigError` immediately when no key is available, naming the variable
    to set. Failing at startup is the whole point: discovering a missing key on
    invoice fourteen of twenty means fourteen invoices have to be reprocessed, and the
    error arrives long after the thing that caused it.
    """
    return XAIClient(
        api_key=api_key if api_key is not None else XAI_API_KEY,
        base_url=XAI_BASE_URL,
        model=model or XAI_MODEL,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        max_retries=LLM_MAX_RETRIES,
    )


__all__ = [
    "get_client",
    "LLMClient",
    "LLMResult",
    "Message",
    "system",
    "user",
    "assistant",
    "LLMError",
    "LLMConfigError",
    "LLMTransportError",
    "LLMResponseError",
    "build_strict_schema",
    "response_format",
    "AGENT_EXCLUDED_FIELDS",
    "XAIClient",
]
