"""The policy engine: findings become an outcome.

Two things live here, and the second one is the important one.

**Evaluation.** Rules from `rules.yaml` are applied to an invoice and its findings, and
the most conservative effect wins. Deterministic, no model, no I/O.

**The interlock.** `combine()` is how the model's opinion and the rules' verdict are
reconciled, and it is one line:

    the more conservative of the two

The model may reject something the rules would have approved, and may escalate something
the rules would have paid. It can never approve something the rules rejected. A document
that talks its way past the language model still meets the policy engine, and the policy
engine cannot be talked to.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

from galatiq.fx import UnknownCurrencyError, to_usd
from galatiq.models import Finding, Invoice, Outcome
from galatiq.policy.predicates import REGISTRY, PolicyInput

# Effects, most conservative first. A rule set that both rejects and holds rejects.
_EFFECT_OUTCOME = {
    "reject": Outcome.REJECTED,
    "hold": Outcome.HELD_FOR_REVIEW,
    "warn": Outcome.APPROVED,
}

# How conservative each outcome is. Used by `combine`, and the only place this ordering
# is written down.
_SEVERITY = {
    Outcome.APPROVED: 0,
    Outcome.HELD_FOR_REVIEW: 1,
    Outcome.REJECTED: 2,
}


@dataclass(frozen=True)
class Rule:
    """One rule, as loaded from configuration."""

    id: str
    description: str
    label: str
    predicate: str
    params: dict[str, Any]
    effect: str
    risk: int

    def applies_to(self, data: PolicyInput) -> bool:
        return REGISTRY[self.predicate](data, self.params)


@dataclass(frozen=True)
class PolicyOutcome:
    """What the rules concluded, and why."""

    outcome: Outcome
    fired: list[Rule] = field(default_factory=list)
    risk: int = 0
    usd_total: Decimal | None = None

    @property
    def policy_refs(self) -> list[str]:
        """Rule ids, for the decision record.

        Between these and the findings' evidence, any decision can be reconstructed
        afterwards without re-running anything.
        """
        return [rule.id for rule in self.fired]

    @property
    def labels(self) -> list[str]:
        """What a reviewer sees at the top of the invoice."""
        return [rule.label for rule in self.fired]

    @property
    def blocking_labels(self) -> list[str]:
        """Only the reasons that changed the outcome.

        A held invoice should lead with why it is held, not with every note that
        happened to apply.
        """
        return [rule.label for rule in self.fired if rule.effect in ("reject", "hold")]


@lru_cache(maxsize=1)
def load_rules(path: str | None = None) -> tuple[Rule, ...]:
    """Load and validate the rule set.

    Cached, because it is read once per process and re-parsing YAML on every invoice in a
    batch would be silly.

    An unknown predicate name raises at load time rather than silently never firing.
    A rule that quietly does nothing is the worst failure mode a config file has: it
    looks present, it reviews well, and it protects nothing.
    """
    if path:
        text = open(path, encoding="utf-8").read()
    else:
        text = resources.files("galatiq.policy").joinpath("rules.yaml").read_text()

    raw = yaml.safe_load(text)
    rules: list[Rule] = []

    for entry in raw.get("rules", []):
        predicate = entry["predicate"]
        if predicate not in REGISTRY:
            raise ValueError(
                f"rule {entry.get('id')}: unknown predicate {predicate!r}; "
                f"available: {', '.join(sorted(REGISTRY))}"
            )
        if entry["effect"] not in _EFFECT_OUTCOME:
            raise ValueError(
                f"rule {entry.get('id')}: unknown effect {entry['effect']!r}"
            )

        rules.append(
            Rule(
                id=entry["id"],
                description=entry.get("description", "").strip(),
                label=entry.get("label", entry["id"]),
                predicate=predicate,
                params=entry.get("params") or {},
                effect=entry["effect"],
                risk=int(entry.get("risk", 0)),
            )
        )

    return tuple(rules)


def usd_total_of(invoice: Invoice) -> Decimal | None:
    """The invoice total in USD, or None when it cannot be established.

    None rather than a guess. A threshold comparison against an unconverted foreign
    amount is not a comparison, and the currency check has already reported the problem
    -- this returns nothing rather than inventing a number for a rule to act on.
    """
    if invoice.total is None:
        return None
    try:
        return to_usd(invoice.total, invoice.currency)
    except UnknownCurrencyError:
        return None


def evaluate(
    invoice: Invoice,
    findings: list[Finding],
    *,
    rules: tuple[Rule, ...] | None = None,
) -> PolicyOutcome:
    """Apply the rules. Deterministic, no model, no I/O."""
    data = PolicyInput(
        invoice=invoice, findings=findings, usd_total=usd_total_of(invoice)
    )

    fired = [rule for rule in (rules or load_rules()) if rule.applies_to(data)]

    outcome = Outcome.APPROVED
    for rule in fired:
        outcome = _more_conservative(outcome, _EFFECT_OUTCOME[rule.effect])

    # An invoice with an unreadable total is held rather than approved. Every threshold
    # rule returns False when the amount is unknown, so without this the least
    # understood documents would sail through as small clean ones.
    if data.usd_total is None and outcome is Outcome.APPROVED:
        outcome = Outcome.HELD_FOR_REVIEW

    return PolicyOutcome(
        outcome=outcome,
        fired=fired,
        risk=max((rule.risk for rule in fired), default=0),
        usd_total=data.usd_total,
    )


def combine(rules_outcome: Outcome, model_outcome: Outcome) -> Outcome:
    """Reconcile the rules and the model. The interlock.

    Take the more conservative of the two:

        REJECT   if the rules reject   -- the model cannot override
        REJECT   if the model rejects  -- the model may add rejections
        HELD     if either escalates
        APPROVE  only if both concur

    Fail-closed, in one line, with no branch for a future maintainer to soften.
    """
    return _more_conservative(rules_outcome, model_outcome)


def _more_conservative(a: Outcome, b: Outcome) -> Outcome:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b
