"""The extraction critic: a second agent asking whether the first one misread.

Separate from the extractor, with its own prompt and its own output shape. A generator
and its reviewer sharing a prompt is one agent talking to itself, and it inherits the
mistake it is supposed to catch.

The critic's whole value is in a distinction it is easy to build without noticing:

    "Did I misread the document?"      <- this agent
    "Is the document itself wrong?"    <- the validation checks, two tickets later

A critic that conflates them cannot terminate on a bad document. INV-1009 states a
subtotal of 1000.00 while its line items sum to -250.00. Transcribe that faithfully and
a naive critic sees a contradiction, calls it a misread, sends the extractor back, and
gets the same values -- because they are the correct values. Two round trips later the
budget is gone and an invoice the system understood perfectly is escalated to a human.

So the verdict is a three-way choice, not a boolean.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from galatiq.agents.prompts import critique_messages
from galatiq.llm import AGENT_EXCLUDED_FIELDS, LLMClient, LLMResponseError
from galatiq.models import Finding, FindingCode, Invoice, Severity


class Discrepancy(BaseModel):
    """One place the transcription and the document disagree.

    Structured rather than prose so it can be fed back to the extractor as a specific
    correction. "You got something wrong" produces another guess; "line 3 quantity: you
    transcribed 5, the document says 15" produces a fix.
    """

    model_config = ConfigDict(extra="forbid")

    field: str
    transcribed: str
    document_says: str


class Critique(BaseModel):
    """The critic's verdict on a transcription."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PARSE_SOUND", "MISPARSE_SUSPECTED", "DOCUMENT_INCONSISTENT"]
    reasoning: str
    discrepancies: list[Discrepancy] = Field(default_factory=list)

    @property
    def suspects_misparse(self) -> bool:
        """True only when re-reading could plausibly produce a better answer.

        The routing function's whole question. DOCUMENT_INCONSISTENT deliberately
        answers False: the document is the problem, and reading it again will not
        change that.
        """
        return self.verdict == "MISPARSE_SUSPECTED"


def critique_extraction(
    client: LLMClient,
    *,
    raw_text: str,
    invoice: Invoice,
) -> Critique:
    """Audit a transcription against its source document.

    A critic that cannot be reached -- malformed response, budget-exhausted extraction,
    empty document -- must not block the pipeline. Callers get a PARSE_SOUND verdict
    saying the audit did not happen, and the invoice proceeds to the checks, which are
    deterministic and do not depend on the model having been available.
    """
    try:
        result = client.complete(
            critique_messages(
                raw_text,
                invoice.model_dump_json(indent=2, exclude=set(AGENT_EXCLUDED_FIELDS)),
            ),
            Critique,
        )
    except LLMResponseError as exc:
        return Critique(
            verdict="PARSE_SOUND",
            reasoning=f"Critique unavailable: {exc}",
        )

    return result.value


def findings_for(critique: Critique) -> list[Finding]:
    """Findings the critique produces for downstream stages.

    Only DOCUMENT_INCONSISTENT travels onward. PARSE_SOUND has nothing to say, and
    MISPARSE_SUSPECTED is a routing instruction rather than a fact about the invoice --
    by the time the loop exits, either the misparse was corrected or the budget ran out,
    and each of those is reported by whoever handled it.
    """
    if critique.verdict != "DOCUMENT_INCONSISTENT":
        return []

    evidence = "; ".join(
        f"{d.field}: transcribed {d.transcribed!r}, document says {d.document_says!r}"
        for d in critique.discrepancies
    )

    return [
        Finding(
            code=FindingCode.DOC_INCONSISTENT,
            severity=Severity.WARN,
            message=(
                "The document was read correctly but is internally inconsistent. "
                f"{critique.reasoning}"
            ),
            evidence=evidence or critique.reasoning,
        )
    ]
