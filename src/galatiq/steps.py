"""Names for the graph's nodes, in words a person can read.

Every surface that reports progress needs the same vocabulary: the terminal under
`--verbose`, the web dashboard, and the log file. Keeping one table means they cannot
drift into calling the same node two different things.

**`kind` is the load-bearing field.** Five nodes call a model and the rest are arithmetic
with tests. That distinction is the system's central claim, and any surface that rendered
the two identically would misrepresent it.

This module deliberately imports nothing from the rest of the package. The CLI reads it,
and a CLI that had to import the API -- and through it FastAPI, and through that a web
server -- to find out what to call a node would have the dependency arrow backwards.
"""

from dataclasses import dataclass
from typing import Literal

Kind = Literal["agent", "deterministic"]


@dataclass(frozen=True)
class StepInfo:
    """How to describe one node to somebody who did not build it."""

    label: str
    kind: Kind


# A person watching should read "Extraction critic is re-reading this", not
# `extract_critic`.
STEPS: dict[str, StepInfo] = {
    "load": StepInfo("Reading the document", "deterministic"),
    "extract": StepInfo("Extractor", "agent"),
    "extract_critic": StepInfo("Extraction critic", "agent"),
    "finalize": StepInfo("Collecting findings", "deterministic"),
    "prepare_checks": StepInfo("Reading inventory", "deterministic"),
    "normalize": StepInfo("Normalizer", "agent"),
    "check_stock": StepInfo("Checking stock", "deterministic"),
    "check_pricing": StepInfo("Checking prices against the catalog", "deterministic"),
    "check_arithmetic": StepInfo("Checking arithmetic", "deterministic"),
    "check_integrity": StepInfo("Checking required fields", "deterministic"),
    "check_duplicates": StepInfo("Checking for duplicates", "deterministic"),
    "check_dates": StepInfo("Checking dates", "deterministic"),
    "check_currency": StepInfo("Checking currency", "deterministic"),
    "check_fraud": StepInfo("Checking for fraud signals", "deterministic"),
    "merge_findings": StepInfo("Collating findings", "deterministic"),
    "approve": StepInfo("Approver", "agent"),
    "approval_critic": StepInfo("Approval critic", "agent"),
    "pay": StepInfo("Releasing payment", "deterministic"),
    "reject": StepInfo("Recording the rejection", "deterministic"),
    "hold": StepInfo("Holding for review", "deterministic"),
}

# The checks that run at once. The only genuinely parallel moment in the pipeline, and
# worth reporting as one thing rather than as eight flickering past.
PARALLEL_CHECKS = frozenset(name for name in STEPS if name.startswith("check_"))


def describe(step: str) -> StepInfo:
    """A step's label and kind, with a safe fallback for a node added later."""
    return STEPS.get(
        step, StepInfo(step.replace("_", " ").capitalize(), "deterministic")
    )


__all__ = ["Kind", "StepInfo", "STEPS", "PARALLEL_CHECKS", "describe"]
