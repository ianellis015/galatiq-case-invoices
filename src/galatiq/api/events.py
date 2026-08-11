"""Turning graph steps into events a browser can render.

The pipeline already emits everything needed -- `graph.stream(stream_mode="debug")`
yields a `task` when a node starts and a `task_result` when it finishes. This module
translates those into something with names a person can read, and spots the two moments
worth calling out.

**Nothing here changes how the pipeline runs.** It watches. The CLI does not import this,
and a batch produces identical decisions whether anyone is watching or not.

Two things are derived rather than reported:

**Handoffs.** A node sequence that goes backwards -- `extract_critic` then `extract`
again -- is one agent sending work to another. The graph does not label it as such; it
falls out of the order, and it is the most interesting thing that happens.

**Agent versus machinery.** Five nodes call a model and the rest are arithmetic. The
distinction is the system's central claim, and a UI that rendered them identically would
misrepresent it.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["agent", "deterministic"]


@dataclass(frozen=True)
class StepInfo:
    """How to describe one node to somebody who did not build it."""

    label: str
    kind: Kind


# Plain descriptions. A person watching should read "Extraction critic is re-reading
# this", not `extract_critic`.
#
# `kind` is the load-bearing field. The extractor, the two critics, the normalizer and
# the approver are the only nodes that call a model; everything else is arithmetic with
# tests. Rendering them the same way would flatten the one distinction the architecture
# is built on.
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

# The eight that run at once. The only genuinely parallel moment in the pipeline, and
# worth showing as one rather than as eight things happening to flicker together.
PARALLEL_CHECKS = frozenset(
    name for name in STEPS if name.startswith("check_")
)

# Backwards edges: an agent handing work to another with something to say. The value is
# what to say when no specific reason can be recovered from the state.
_HANDOFFS = {
    ("extract_critic", "extract"): "sent back for a re-read",
    ("approval_critic", "approve"): "sent back for reconsideration",
}


def describe(step: str) -> StepInfo:
    """A step's label and kind, with a safe fallback for a node added later."""
    return STEPS.get(step, StepInfo(step.replace("_", " ").capitalize(), "deterministic"))


@dataclass
class DocumentTracker:
    """Watches one document's journey and produces its events.

    Stateful because a handoff is only visible in the transition: knowing that `extract`
    is starting says nothing until you know what finished immediately before it.
    """

    doc: str
    last_completed: str | None = None
    invoice_number: str | None = None
    _state: dict[str, Any] = field(default_factory=dict)

    def on_task(self, step: str) -> list[dict[str, Any]]:
        """A node is about to run."""
        events: list[dict[str, Any]] = []

        handoff = _HANDOFFS.get((self.last_completed or "", step))
        if handoff:
            events.append(
                {
                    "type": "handoff",
                    "doc": self.doc,
                    "invoice": self.invoice_number,
                    "from": describe(self.last_completed).label,
                    "to": describe(step).label,
                    "reason": self._handoff_reason(step) or handoff,
                }
            )

        info = describe(step)
        events.append(
            {
                "type": "step.start",
                "doc": self.doc,
                "invoice": self.invoice_number,
                "step": step,
                "label": info.label,
                "kind": info.kind,
                "parallel": step in PARALLEL_CHECKS,
            }
        )
        return events

    def on_task_result(self, step: str, state: dict[str, Any]) -> list[dict[str, Any]]:
        """A node has finished. `state` is the accumulated run state, if available."""
        self.last_completed = step
        if state:
            self._state = state

        invoice = self._state.get("invoice")
        if invoice is not None and getattr(invoice, "invoice_number", None):
            self.invoice_number = invoice.invoice_number

        return [
            {
                "type": "step.end",
                "doc": self.doc,
                "invoice": self.invoice_number,
                "step": step,
                "label": describe(step).label,
            }
        ]

    def _handoff_reason(self, target: str) -> str | None:
        """What the critic actually objected to, if it said.

        A specific reason is the difference between "an agent sent it back" and
        "the critic thought line 3's quantity was misread" -- and the second is the
        thing worth watching.
        """
        if target == "extract":
            critique = self._state.get("critique")
            discrepancies = getattr(critique, "discrepancies", None)
            if discrepancies:
                first = discrepancies[0]
                return f"{first.field}: read {first.transcribed!r}, document says {first.document_says!r}"
            return getattr(critique, "reasoning", None)

        if target == "approve":
            critique = self._state.get("approval_critique")
            missed = getattr(critique, "missed", None)
            if missed:
                return missed[0]
            return getattr(critique, "reasoning", None)

        return None
