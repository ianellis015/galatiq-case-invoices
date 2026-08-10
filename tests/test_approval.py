"""Tests for the approver and the approval critic.

The interlock itself is proven in `test_policy.py` against plain data. These check that
the agents are wired into it correctly — in particular that a model returning APPROVED
on a rejected invoice changes nothing.
"""

from galatiq.agents.approval_critic import ApprovalCritique, critique_decision
from galatiq.agents.approver import ApproverVerdict, decide, format_policy, revise
from galatiq.llm import LLMResponseError
from galatiq.models import (
    ApprovalDecision,
    Finding,
    FindingCode,
    Invoice,
    Outcome,
    Severity,
)
from galatiq.policy import evaluate

from conftest import FakeLLM


def invoice(total="5000.00", **kwargs):
    return Invoice(
        invoice_number="INV-1001", vendor="Widgets Inc.", total=total,
        currency="USD", **kwargs,
    )


def critical(code=FindingCode.STOCK_EXCEEDED):
    return [Finding(code=code, severity=Severity.CRITICAL, message="x", evidence="y")]


def verdict(outcome=Outcome.APPROVED, **kwargs):
    return ApproverVerdict(
        outcome=outcome, rationale=kwargs.pop("rationale", "Looks fine."), **kwargs
    )


class TestTheModelCannotApproveWhatTheRulesRejected:
    """The property the whole design turns on."""

    def test_a_rejected_invoice_stays_rejected(self):
        inv, findings = invoice(), critical()
        policy = evaluate(inv, findings)
        client = FakeLLM(verdict(Outcome.APPROVED, rationale="Seems fine to me!"))

        decision = decide(client, invoice=inv, findings=findings, policy=policy)

        assert policy.outcome is Outcome.REJECTED
        assert decision.outcome is Outcome.REJECTED

    def test_a_held_invoice_is_not_approved_by_the_model(self):
        inv = invoice("50000.00")
        policy = evaluate(inv, [])
        client = FakeLLM(verdict(Outcome.APPROVED))

        decision = decide(client, invoice=inv, findings=[], policy=policy)

        assert decision.outcome is Outcome.HELD_FOR_REVIEW

    def test_the_model_is_told_it_cannot(self):
        """Stated in the prompt as well as enforced in code.

        The enforcement is what makes it true; saying so keeps the rationale coherent
        with the outcome instead of arguing with it.
        """
        text = format_policy(evaluate(invoice(), critical()))
        assert "You cannot approve it" in text


class TestTheModelCanAddConservatism:
    def test_it_can_reject_something_the_rules_would_have_approved(self):
        """Veto power is real. It only goes one way."""
        inv = invoice("500.00")
        policy = evaluate(inv, [])
        client = FakeLLM(verdict(Outcome.REJECTED, rationale="Vendor unrecognised."))

        decision = decide(client, invoice=inv, findings=[], policy=policy)

        assert policy.outcome is Outcome.APPROVED
        assert decision.outcome is Outcome.REJECTED

    def test_it_can_escalate(self):
        inv = invoice("500.00")
        client = FakeLLM(verdict(Outcome.HELD_FOR_REVIEW))

        decision = decide(client, invoice=inv, findings=[], policy=evaluate(inv, []))

        assert decision.outcome is Outcome.HELD_FOR_REVIEW


class TestTheDecisionRecord:
    def test_policy_refs_come_from_the_engine_not_the_model(self):
        """Which rules fired is a fact, not an opinion."""
        inv = invoice("50000.00")
        client = FakeLLM(verdict())

        decision = decide(client, invoice=inv, findings=[], policy=evaluate(inv, []))

        assert "R1" in decision.policy_refs

    def test_blocking_labels_lead_the_rationale(self):
        """A reviewer opening a held invoice needs to know why it is in their queue
        before they need the narrative."""
        inv = invoice("50000.00")
        client = FakeLLM(verdict(rationale="Ordinary hardware order."))

        decision = decide(client, invoice=inv, findings=[], policy=evaluate(inv, []))

        assert decision.rationale.startswith("[Over $10,000")
        assert "Ordinary hardware order." in decision.rationale

    def test_risk_is_the_higher_of_the_two(self):
        inv = invoice("50000.00")  # R1 floor is 30
        client = FakeLLM(verdict(risk_score=5))

        decision = decide(client, invoice=inv, findings=[], policy=evaluate(inv, []))

        assert decision.risk_score == 30


class TestModelUnavailable:
    def test_a_clean_invoice_is_held_rather_than_approved(self):
        """Approving means the discretionary review found nothing. No review took
        place, so there is nothing to have found."""
        inv = invoice("500.00")
        client = FakeLLM(LLMResponseError("malformed"))

        decision = decide(client, invoice=inv, findings=[], policy=evaluate(inv, []))

        assert decision.outcome is Outcome.HELD_FOR_REVIEW
        assert "not had a discretionary check" in decision.rationale

    def test_a_rejection_still_stands(self):
        """The rules are the authoritative half, and they ran."""
        inv, findings = invoice(), critical()
        client = FakeLLM(LLMResponseError("malformed"))

        decision = decide(
            client, invoice=inv, findings=findings, policy=evaluate(inv, findings)
        )

        assert decision.outcome is Outcome.REJECTED

    def test_no_client_at_all_behaves_the_same(self):
        inv = invoice("500.00")
        decision = decide(None, invoice=inv, findings=[], policy=evaluate(inv, []))

        assert decision.outcome is Outcome.HELD_FOR_REVIEW


class TestApprovalCritic:
    def _decision(self):
        return ApprovalDecision(outcome=Outcome.APPROVED, rationale="Clean.")

    def test_sound_does_not_send_work_back(self):
        critique = ApprovalCritique(verdict="SOUND", reasoning="Follows the evidence.")
        assert not critique.found_something

    def test_missed_signals_does(self):
        critique = ApprovalCritique(
            verdict="MISSED_SIGNALS", reasoning="Urgency ignored.", missed=["urgency"]
        )
        assert critique.found_something

    def test_it_audits_the_decision(self):
        client = FakeLLM(
            ApprovalCritique(verdict="MISSED_SIGNALS", reasoning="x", missed=["y"])
        )

        critique = critique_decision(
            client, invoice=invoice(), findings=[], decision=self._decision()
        )

        assert critique.verdict == "MISSED_SIGNALS"
        assert "Clean." in client.calls[0].prompt

    def test_an_unreachable_critic_does_not_block_the_pipeline(self):
        """A model outage should not stop payments. The rules engine — the
        authoritative half — already ran."""
        client = FakeLLM(LLMResponseError("malformed"))

        critique = critique_decision(
            client, invoice=invoice(), findings=[], decision=self._decision()
        )

        assert critique.verdict == "SOUND"
        assert "unavailable" in critique.reasoning


class TestRevision:
    def test_the_critique_is_fed_back(self):
        """Naming what was missed is the difference between reconsidering and
        re-rolling."""
        inv = invoice("500.00")
        critique = ApprovalCritique(
            verdict="MISSED_SIGNALS",
            reasoning="x",
            missed=["Vendor has never invoiced before"],
        )
        client = FakeLLM(verdict(Outcome.HELD_FOR_REVIEW))

        decision = revise(
            client,
            invoice=inv,
            findings=[],
            policy=evaluate(inv, []),
            critique=critique,
        )

        assert "never invoiced before" in client.calls[0].prompt
        assert decision.outcome is Outcome.HELD_FOR_REVIEW

    def test_a_revision_still_cannot_approve_a_rejection(self):
        inv, findings = invoice(), critical()
        critique = ApprovalCritique(verdict="MISSED_SIGNALS", reasoning="x", missed=["y"])
        client = FakeLLM(verdict(Outcome.APPROVED))

        decision = revise(
            client,
            invoice=inv,
            findings=findings,
            policy=evaluate(inv, findings),
            critique=critique,
        )

        assert decision.outcome is Outcome.REJECTED
