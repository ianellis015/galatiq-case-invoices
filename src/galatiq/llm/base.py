"""The contract every agent calls through.

One method, one job: given a conversation and a pydantic model, return a validated
instance of that model. No agent anywhere parses free-form text out of a response --
if it is not typed, it did not come from here.

The interface takes no LangChain or LangGraph dependency on purpose. Agents stay
plain functions that can be called directly or from a graph node, and the structured
output path and its validation errors stay ours to shape. That second part matters
more than it sounds: `LLMResponseError` is the hook the extraction retry loop
attaches to, and a wrapper library owning that error would own the loop's behaviour.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One turn of a conversation.

    A list of these rather than a system/user pair, because the extraction retry loop
    works by feeding a specific validation error back as another turn. That is
    multi-turn by construction, and an interface that assumed two messages would have
    had to be torn up the moment the first self-correction loop was wired.
    """

    role: Role
    content: str


def system(content: str) -> Message:
    """Instructions. This channel carries authority; the user channel does not."""
    return Message(role="system", content=content)


def user(content: str) -> Message:
    """Content for the model to work on."""
    return Message(role="user", content=content)


def assistant(content: str) -> Message:
    """A prior model turn, replayed so a retry can see what it said last time."""
    return Message(role="assistant", content=content)


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    """A validated response, plus what it cost and who produced it.

    `provider` and `model` map onto two ledger columns: "what decided to pay this" has
    to be a stored fact rather than something inferred from whose machine the batch ran
    on.

    The token counts are captured on every call and nothing currently totals them. I
    carry them anyway because the provider returns them for free, and a cost figure
    reconstructed later from a billing dashboard is not the same evidence as one the
    run reported itself.
    """

    value: T
    provider: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    """What an adapter has to provide.

    A Protocol rather than a base class -- adapters have nothing to inherit, and
    structural typing means a test double is a client without needing to say so.
    """

    provider: str
    model: str

    def complete[T: BaseModel](
        self,
        messages: list[Message],
        response_model: type[T],
        *,
        temperature: float | None = None,
        exclude: frozenset[str] | None = None,
    ) -> LLMResult[T]:
        """Send a conversation, get back a validated `response_model`.

        `exclude` drops fields from the schema the model is asked to fill. Provenance
        fields (`source_path`, `source_format`) and `extra` are set by us and would
        only invite the model to invent them.
        """
        ...


# --- errors ------------------------------------------------------------------
#
# Three distinct types because three different callers handle them, and collapsing
# them into one would mean the retry loop could not tell "the network blipped" from
# "the model misread the document".


class LLMError(Exception):
    """Base for anything that goes wrong talking to a model."""


class LLMConfigError(LLMError):
    """No API key, or a base URL that cannot be used.

    Raised at startup rather than partway through a batch, and phrased to name the
    variable that needs setting. Discovering a missing key on invoice fourteen of
    twenty is a worse experience than discovering it before invoice one.
    """


class LLMTransportError(LLMError):
    """A connection failure, timeout, rate limit, or server error.

    Retried inside the adapter with backoff. Nothing above the adapter should need to
    know the network is unreliable.
    """


class LLMResponseError(LLMError):
    """The model replied, but the reply is not a valid `response_model`.

    This is the one the extraction critic catches. `detail` carries the pydantic
    validation message verbatim so it can be fed back into the next attempt -- telling
    the model *which field* was wrong is what makes a retry more than a re-roll.
    """

    def __init__(self, message: str, *, detail: str = "", raw: str = "") -> None:
        super().__init__(message)
        self.detail = detail
        self.raw = raw
