"""Turning graph steps into events a browser can render.

The pipeline already emits everything needed -- `graph.stream(stream_mode="debug")`
yields a `task` when a node starts and a `task_result` when it finishes. This module
translates those into events with names a person can read, and spots the moment worth
calling out.

**Nothing here changes how the pipeline runs.** It watches, and a batch produces
identical decisions whether anyone is watching or not.

**Handoffs** are derived rather than reported. A node sequence that goes backwards --
`extract_critic` then `extract` again -- is one agent sending work to another. The graph
does not label it as such; it falls out of the order, and it is the most interesting
thing that happens in a run.

The step names live in `galatiq.steps`, shared with the terminal and the log file so the
three surfaces cannot drift into calling the same node different things.
"""

from dataclasses import dataclass, field
from typing import Any

from galatiq.steps import PARALLEL_CHECKS, STEPS, Kind, StepInfo, describe

# Backwards edges: an agent handing work to another with something to say. The value is
# what to say when no specific reason can be recovered from the state.
_HANDOFFS = {
    ("extract_critic", "extract"): "sent back for a re-read",
    ("approval_critic", "approve"): "sent back for reconsideration",
}


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
