"""The approval critic: what did the reviewer miss?

The second of the two self-correction loops, and a different question from the first.
The extraction critic asks whether the document was *read* correctly. This one assumes
the reading is right and asks whether the *conclusion* is.

A separate agent with its own prompt, for the same reason as before: a generator and its
reviewer sharing one prompt is one agent talking to itself, and it inherits the blind
spot it exists to find.

The termination discipline is the same too. Two verdicts, and only one of them sends work
back -- because re-reviewing the same decision costs money, delays a payment, and "I
would have phrased it differently" is not a missed signal.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from galatiq.agents.approver import format_findings
from galatiq.agents.prompts import approval_critique_messages
from galatiq.llm import LLMClient, LLMResponseError
from galatiq.models import ApprovalDecision, Finding, Invoice, Outcome


class ApprovalCritique(BaseModel):
    """A second reviewer's audit of a decision."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["SOUND", "MISSED_SIGNALS"]
    reasoning: str
    missed: list[str] = Field(default_factory=list)

    # A recommendation, not an instruction. It still passes through the same interlock,
    # so it can make an outcome more conservative and never less.
    recommended_outcome: Outcome | None = None

    @property
    def found_something(self) -> bool:
        """The routing function's only question."""
        return self.verdict == "MISSED_SIGNALS"


def critique_decision(
    client: LLMClient | None,
    *,
    invoice: Invoice,
    findings: list[Finding],
    decision: ApprovalDecision,
) -> ApprovalCritique:
    """Audit a decision.

    An unreachable critic returns SOUND. The decision already passed the rules engine,
    which is the authoritative half -- losing the discretionary second look is a
    reduction in scrutiny, not a hole in the interlock, and blocking the pipeline on it
    would mean a model outage stops payments entirely.
    """
    if client is None:
        return ApprovalCritique(verdict="SOUND", reasoning="No critic configured.")

    try:
        result = client.complete(
            approval_critique_messages(
                invoice.model_dump_json(indent=2, exclude={"extra"}),
                format_findings(findings),
                decision.model_dump_json(indent=2),
            ),
            ApprovalCritique,
        )
    except LLMResponseError as exc:
        return ApprovalCritique(
            verdict="SOUND", reasoning=f"Audit unavailable: {exc}"
        )

    return result.value
