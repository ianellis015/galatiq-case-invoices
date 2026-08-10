"""Tests for the policy engine and the interlock.

No model, no graph, no database — the rules are a pure function of an invoice and its
findings, which is what lets the interlock be proven exhaustively rather than argued
about.

`combine` gets a full 3×3 matrix. It is one line of code and the single thing standing
between a persuasive document and a payment, so it is worth testing every cell.
"""

from decimal import Decimal

import pytest

from galatiq.models import Finding, FindingCode, Invoice, Outcome, Severity
from galatiq.policy import Rule, combine, evaluate, load_rules, usd_total_of
from galatiq.policy.predicates import REGISTRY


def finding(code, severity=Severity.CRITICAL):
    return Finding(code=code, severity=severity, message="test", evidence="test")


def usd(amount):
    return Invoice(invoice_number="INV-T", total=amount, currency="USD")


class TestRulesLoad:
    def test_the_shipped_rules_parse(self):
        rules = load_rules()

        assert [r.id for r in rules] == ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]

    def test_every_rule_names_a_real_predicate(self):
        for rule in load_rules():
            assert rule.predicate in REGISTRY

    def test_every_rule_has_a_human_readable_label(self):
        """The label is what a reviewer sees at the top of a held invoice. A rule id is
        not an explanation."""
        for rule in load_rules():
            assert rule.label and rule.label != rule.id

    def test_an_unknown_predicate_fails_at_load(self, tmp_path):
        """Not silently at evaluation.

        A rule that never fires is the worst failure a config file has: it looks
        present, it reviews well, and it protects nothing.
        """
        path = tmp_path / "bad.yaml"
        path.write_text(
            "rules:\n  - id: RX\n    predicate: no_such_thing\n    effect: reject\n"
        )

        with pytest.raises(ValueError, match="unknown predicate"):
            load_rules(str(path))

    def test_an_unknown_effect_fails_at_load(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "rules:\n  - id: RX\n    predicate: always\n    effect: maybe\n"
        )

        with pytest.raises(ValueError, match="unknown effect"):
            load_rules(str(path))


class TestThreshold:
    def test_under_the_threshold_approves(self):
        assert evaluate(usd("5000.00"), []).outcome is Outcome.APPROVED

    def test_at_the_threshold_holds(self):
        """"Over $10K requires additional scrutiny" — the boundary is inclusive."""
        assert evaluate(usd("10000.00"), []).outcome is Outcome.HELD_FOR_REVIEW

    def test_a_large_clean_invoice_is_held_not_rejected(self):
        """The amount never rejects anything on its own.

        Correctness decides approve-vs-reject; size decides automatic-vs-human. Two
        independent axes.
        """
        outcome = evaluate(usd("100000.00"), [])

        assert outcome.outcome is Outcome.HELD_FOR_REVIEW
        assert "R1" in outcome.policy_refs

    def test_the_structuring_band_warns_without_holding(self):
        """INV-1012 at $9,975. One invoice just under the line is unremarkable; a
        pattern of them is somebody splitting orders."""
        outcome = evaluate(usd("9975.00"), [])

        assert outcome.outcome is Outcome.APPROVED
        assert "R6" in outcome.policy_refs
        assert outcome.risk >= 40


class TestCurrencyNormalisation:
    def test_eur_is_converted_before_the_threshold_applies(self):
        """INV-1014's mechanism. EUR 9,500 is about USD 10,355 — over the line.

        Comparing the raw number would let it through, which is the failure the whole
        conversion exists to prevent.
        """
        invoice = Invoice(invoice_number="INV-1014", total="9500.00", currency="EUR")
        outcome = evaluate(invoice, [])

        assert outcome.usd_total > Decimal("10000")
        assert outcome.outcome is Outcome.HELD_FOR_REVIEW
        assert "R1" in outcome.policy_refs

    def test_an_unconvertible_currency_is_held(self):
        invoice = Invoice(invoice_number="INV-X", total="500.00", currency="XYZ")

        assert usd_total_of(invoice) is None
        assert evaluate(invoice, []).outcome is Outcome.HELD_FOR_REVIEW

    def test_an_unreadable_total_is_held_not_approved(self):
        """Every threshold rule returns False when the amount is unknown, so without
        this the least understood documents would sail through as small clean ones."""
        assert evaluate(Invoice(invoice_number="INV-X"), []).outcome is (
            Outcome.HELD_FOR_REVIEW
        )


class TestFindingsDriveOutcomes:
    def test_a_critical_finding_rejects(self):
        outcome = evaluate(usd("500.00"), [finding(FindingCode.STOCK_EXCEEDED)])

        assert outcome.outcome is Outcome.REJECTED
        assert "R2" in outcome.policy_refs

    def test_a_small_broken_invoice_is_rejected(self):
        """Size does not rescue an invoice, any more than it condemns one."""
        assert evaluate(usd("50.00"), [finding(FindingCode.UNKNOWN_ITEM)]).outcome is (
            Outcome.REJECTED
        )

    def test_soft_fraud_signals_hold_rather_than_reject(self):
        """A real vendor chasing a late payment writes "URGENT".

        Rejecting them for it is the more expensive mistake, so these escalate to a
        human with a label rather than refusing outright.
        """
        outcome = evaluate(
            usd("500.00"), [finding(FindingCode.FRAUD_SIGNAL, Severity.WARN)]
        )

        assert outcome.outcome is Outcome.HELD_FOR_REVIEW
        assert "Potential fraud detected — review before paying" in outcome.labels

    def test_injection_language_rejects(self):
        """Nobody writes "ignore previous instructions" on a real invoice."""
        outcome = evaluate(
            usd("500.00"), [finding(FindingCode.PROMPT_INJECTION, Severity.WARN)]
        )

        assert outcome.outcome is Outcome.REJECTED
        assert "R5" in outcome.policy_refs

    def test_a_revision_of_a_paid_invoice_holds(self):
        outcome = evaluate(
            usd("500.00"), [finding(FindingCode.REVISION_SUPERSEDES, Severity.WARN)]
        )

        assert outcome.outcome is Outcome.HELD_FOR_REVIEW

    def test_the_most_conservative_effect_wins(self):
        """Held and rejected together is rejected."""
        outcome = evaluate(
            usd("50000.00"),
            [finding(FindingCode.STOCK_EXCEEDED), finding(FindingCode.FRAUD_SIGNAL, Severity.WARN)],
        )

        assert outcome.outcome is Outcome.REJECTED
        assert {"R1", "R2", "R4"} <= set(outcome.policy_refs)


class TestBlockingLabels:
    def test_only_reasons_that_changed_the_outcome(self):
        """A held invoice should lead with why it is held, not with every note that
        happened to apply."""
        outcome = evaluate(usd("9975.00"), [finding(FindingCode.FRAUD_SIGNAL, Severity.WARN)])

        assert outcome.blocking_labels == [
            "Potential fraud detected — review before paying"
        ]
        assert len(outcome.labels) == 2  # the structuring warning is recorded too


class TestTheInterlock:
    """The model has veto power, not approval power.

    One line of code, and the single thing between a persuasive document and a payment.
    Worth testing every cell.
    """

    @pytest.mark.parametrize(
        "rules,model,expected",
        [
            (Outcome.APPROVED, Outcome.APPROVED, Outcome.APPROVED),
            (Outcome.APPROVED, Outcome.HELD_FOR_REVIEW, Outcome.HELD_FOR_REVIEW),
            (Outcome.APPROVED, Outcome.REJECTED, Outcome.REJECTED),
            (Outcome.HELD_FOR_REVIEW, Outcome.APPROVED, Outcome.HELD_FOR_REVIEW),
            (Outcome.HELD_FOR_REVIEW, Outcome.HELD_FOR_REVIEW, Outcome.HELD_FOR_REVIEW),
            (Outcome.HELD_FOR_REVIEW, Outcome.REJECTED, Outcome.REJECTED),
            (Outcome.REJECTED, Outcome.APPROVED, Outcome.REJECTED),
            (Outcome.REJECTED, Outcome.HELD_FOR_REVIEW, Outcome.REJECTED),
            (Outcome.REJECTED, Outcome.REJECTED, Outcome.REJECTED),
        ],
    )
    def test_every_combination(self, rules, model, expected):
        assert combine(rules, model) is expected

    def test_the_model_cannot_approve_what_the_rules_rejected(self):
        """The one that matters. Stated separately from the matrix because it is the
        property, not a case."""
        assert combine(Outcome.REJECTED, Outcome.APPROVED) is Outcome.REJECTED

    def test_the_model_cannot_approve_what_the_rules_held(self):
        assert combine(Outcome.HELD_FOR_REVIEW, Outcome.APPROVED) is (
            Outcome.HELD_FOR_REVIEW
        )

    def test_the_model_may_add_conservatism(self):
        """Veto power is real power. It just only goes one way."""
        assert combine(Outcome.APPROVED, Outcome.REJECTED) is Outcome.REJECTED
        assert combine(Outcome.APPROVED, Outcome.HELD_FOR_REVIEW) is (
            Outcome.HELD_FOR_REVIEW
        )


class TestRulesAreConfiguration:
    def test_the_threshold_can_be_changed_without_touching_code(self, tmp_path):
        """The point of the YAML. A finance lead retunes the number; nobody deploys."""
        path = tmp_path / "rules.yaml"
        path.write_text(
            "rules:\n"
            "  - id: R1\n"
            "    label: Over the threshold\n"
            "    predicate: total_usd_at_least\n"
            "    params: {amount: '500.00'}\n"
            "    effect: hold\n"
            "    risk: 30\n"
        )
        rules = load_rules(str(path))

        assert evaluate(usd("600.00"), [], rules=rules).outcome is (
            Outcome.HELD_FOR_REVIEW
        )
        assert evaluate(usd("400.00"), [], rules=rules).outcome is Outcome.APPROVED

    def test_a_signal_can_be_moved_between_hold_and_reject(self, tmp_path):
        path = tmp_path / "rules.yaml"
        path.write_text(
            "rules:\n"
            "  - id: RX\n"
            "    label: Fraud\n"
            "    predicate: any_finding_code_in\n"
            "    params: {codes: [FRAUD_SIGNAL]}\n"
            "    effect: reject\n"
            "    risk: 100\n"
        )
        rules = load_rules(str(path))
        outcome = evaluate(
            usd("100.00"), [finding(FindingCode.FRAUD_SIGNAL, Severity.WARN)], rules=rules
        )

        assert outcome.outcome is Outcome.REJECTED
