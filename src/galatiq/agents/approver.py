"""The approver: findings become a decision, with reasons a human can read.

Worth being precise about what this agent does, because it is easy to overstate. It does
not decide whether the invoice is paid -- the rules engine does that, and the interlock
in `policy.combine` means the model can only ever make the outcome more conservative.

What it does is the part rules are bad at:

**Writes the reasoning.** Someone reads it months later to understand why money moved.
"R2 fired" is not that; "GadgetX was ordered nine times across three lines against five
in stock, and the total does not reconcile with the line items" is.

**Notices combinations.** Rules see facts one at a time. An unfamiliar vendor, an unusual
amount, urgency in the notes, items slightly outside their normal line -- each
unremarkable, and together a pattern. That is a judgement call, and judgement is what a
model is for.

So: the rules decide, and the model explains and escalates. The README says exactly that,
because implying the model approves invoices would be a more impressive claim and a false
one.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from galatiq.agents.prompts import approval_messages
from galatiq.llm import LLMClient, LLMResponseError
from galatiq.models import ApprovalDecision, Finding, Invoice, Outcome
from galatiq.policy import PolicyOutcome, combine


class ApproverVerdict(BaseModel):
    """What the model returns.

    Narrower than `ApprovalDecision` on purpose: `policy_refs` records which rules fired,
    and that is a fact about the engine rather than an opinion the model gets to have.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: Outcome
    rationale: str
    risk_score: int = Field(default=0, ge=0, le=100)
    concerns: list[str] = Field(default_factory=list)


def decide(
    client: LLMClient | None,
    *,
    invoice: Invoice,
    findings: list[Finding],
    policy: PolicyOutcome,
) -> ApprovalDecision:
    """Produce the final decision for an invoice.

    The rules' outcome and the model's are reconciled by `combine`, which takes the more
    conservative of the two. That is the whole fail-closed guarantee, and it is why this
    function cannot approve something the rules rejected however the model answers.
    """
    verdict = _ask(client, invoice=invoice, findings=findings, policy=policy)

    if verdict is None:
        # The review did not happen. The rules alone are enough to reject or hold, but
        # they are not enough to *approve* -- approving means the discretionary review
        # found nothing, and no review took place. Hold instead.
        outcome = (
            Outcome.HELD_FOR_REVIEW
            if policy.outcome is Outcome.APPROVED
            else policy.outcome
        )
        return ApprovalDecision(
            outcome=outcome,
            rationale=(
                "Automated review was unavailable, so this invoice has not had a "
                "discretionary check. "
                + _policy_sentence(policy)
            ),
            policy_refs=policy.policy_refs,
            risk_score=max(policy.risk, 50),
        )

    return ApprovalDecision(
        outcome=combine(policy.outcome, verdict.outcome),
        rationale=verdict.rationale.strip(),
        policy_refs=policy.policy_refs,
        concerns=_concerns(verdict, policy),
        risk_score=max(policy.risk, verdict.risk_score),
    )


def _ask(
    client: LLMClient | None,
    *,
    invoice: Invoice,
    findings: list[Finding],
    policy: PolicyOutcome,
    critique: Any | None = None,
) -> ApproverVerdict | None:
    """One call, or None when the model could not be reached."""
    if client is None:
        return None

    try:
        result = client.complete(
            approval_messages(
                invoice.model_dump_json(indent=2, exclude={"extra"}),
                format_findings(findings),
                format_policy(policy),
                critique=critique,
            ),
            ApproverVerdict,
        )
    except LLMResponseError:
        return None

    return result.value


def revise(
    client: LLMClient | None,
    *,
    invoice: Invoice,
    findings: list[Finding],
    policy: PolicyOutcome,
    critique: Any,
) -> ApprovalDecision:
    """Reconsider a decision in light of what a second reviewer flagged."""
    verdict = _ask(
        client, invoice=invoice, findings=findings, policy=policy, critique=critique
    )

    if verdict is None:
        return decide(None, invoice=invoice, findings=findings, policy=policy)

    return ApprovalDecision(
        outcome=combine(policy.outcome, verdict.outcome),
        rationale=verdict.rationale.strip(),
        policy_refs=policy.policy_refs,
        concerns=_concerns(verdict, policy),
        risk_score=max(policy.risk, verdict.risk_score),
    )


def _concerns(verdict: ApproverVerdict, policy: PolicyOutcome) -> list[str]:
    """The short reasons this invoice is where it is: rule labels, then the model's.

    Rules first because they are the ones that *bound* the outcome -- the model's
    concerns are discretionary and the rules' are not. Deduplicated, because the model
    routinely restates a rule label in its own words and the same reason listed twice
    reads like two problems.
    """
    seen: dict[str, str] = {}

    for part in [*policy.blocking_labels, *verdict.concerns]:
        text = part.strip()
        if text:
            seen.setdefault(text.casefold(), text)

    return list(seen.values())


def _policy_sentence(policy: PolicyOutcome) -> str:
    if not policy.fired:
        return "No policy rules were triggered."
    return "Policy: " + "; ".join(policy.labels) + "."


def format_findings(findings: list[Finding]) -> str:
    """Findings as text for the prompt, severity-tagged and quoting the evidence."""
    return "\n".join(
        f"[{f.severity}] {f.code}: {f.message}"
        + (f"\n    evidence: {f.evidence}" if f.evidence else "")
        for f in findings
    )


def format_policy(policy: PolicyOutcome) -> str:
    """The rules result as text, including what the model may and may not do with it."""
    lines = [f"Verdict: {policy.outcome}", f"Risk floor: {policy.risk}"]

    if policy.usd_total is not None:
        lines.append(f"Total in USD: {policy.usd_total}")

    if policy.fired:
        lines.append("Rules triggered:")
        lines.extend(f"  {rule.id} ({rule.effect}): {rule.label}" for rule in policy.fired)
    else:
        lines.append("Rules triggered: none")

    if policy.outcome is Outcome.REJECTED:
        lines.append("This invoice is rejected. You cannot approve it.")
    elif policy.outcome is Outcome.HELD_FOR_REVIEW:
        lines.append("This invoice requires a human. You cannot approve it.")

    return "\n".join(lines)
