"""The fixed vocabulary the rules are written in.

Rules live in `rules.yaml` as configuration, which is a good idea that can go wrong in a
specific way: a config file containing arbitrary boolean expressions is a programming
language with no type checking, no tests and no debugger, sitting in the path of every
payment decision.

So the config picks from this menu and fills in the numbers. A finance lead can change
the threshold, retune the structuring band, or move a signal from "hold" to "reject" by
editing YAML. They cannot invent a new *kind* of condition -- that means adding a
function here, with a test, which is the right amount of friction for changing what the
system is capable of noticing.

Every predicate takes the same input and returns a bool. Nothing here reads a database,
calls a model, or has an opinion about what to do with the answer.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from galatiq.models import Finding, Invoice, Severity


@dataclass(frozen=True)
class PolicyInput:
    """Everything a rule is allowed to look at.

    `usd_total` is normalised before it gets here, so no predicate has to think about
    currency. That is the entire reason INV-1014's EUR total is converted upstream:
    a threshold comparison against a foreign amount is not a comparison.
    """

    invoice: Invoice
    findings: list[Finding]
    usd_total: Decimal | None


Predicate = Callable[[PolicyInput, dict[str, Any]], bool]


def any_finding_with_severity(data: PolicyInput, params: dict[str, Any]) -> bool:
    """Is any finding at least this severe?

    The workhorse. Most rejections come through here, because the checks already did the
    thinking -- "stock exceeded" is CRITICAL because running out of stock is disqualifying,
    not because a rule later decided it was.
    """
    wanted = Severity(params["severity"])
    return any(f.severity == wanted for f in data.findings)


def any_finding_code_in(data: PolicyInput, params: dict[str, Any]) -> bool:
    """Did any of these specific problems occur?

    Used where the *reason* changes the response rather than the severity: a fraud signal
    and a stock shortfall are both worth a human's time, and they are worth it for
    entirely different reasons.
    """
    codes = set(params["codes"])
    return any(f.code in codes for f in data.findings)


def total_usd_at_least(data: PolicyInput, params: dict[str, Any]) -> bool:
    """Is the invoice worth at least this much?

    False when the total is unknown. An invoice whose amount cannot be read is a problem,
    but it is the integrity check's problem -- and treating unknown as "under the
    threshold" would auto-approve exactly the documents we understand least.
    """
    if data.usd_total is None:
        return False
    return data.usd_total >= Decimal(str(params["amount"]))


def total_usd_between(data: PolicyInput, params: dict[str, Any]) -> bool:
    """Is the total inside a band? Lower bound inclusive, upper exclusive."""
    if data.usd_total is None:
        return False
    return Decimal(str(params["low"])) <= data.usd_total < Decimal(str(params["high"]))


def currency_not_usd(data: PolicyInput, params: dict[str, Any]) -> bool:
    """Was this invoice denominated in something else?"""
    stated = (data.invoice.currency or "USD").strip().upper()
    return stated != "USD"


def always(data: PolicyInput, params: dict[str, Any]) -> bool:
    """Fires unconditionally. For a baseline rule, and for testing the loader."""
    return True


REGISTRY: dict[str, Predicate] = {
    "any_finding_with_severity": any_finding_with_severity,
    "any_finding_code_in": any_finding_code_in,
    "total_usd_at_least": total_usd_at_least,
    "total_usd_between": total_usd_between,
    "currency_not_usd": currency_not_usd,
    "always": always,
}
