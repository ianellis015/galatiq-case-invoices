"""The shared document every node writes to.

Nodes do not message each other. They take turns writing to one typed structure, and
the state after each node *is* the audit trail -- the checkpointer snapshots it on every
step, so "what did the system know before it rejected this invoice" is a question with
an exact answer rather than a log-grepping exercise.

Each node returns a partial update. LangGraph merges it: last write wins, except where
a reducer says otherwise.
"""

import operator
from typing import Annotated, Any, TypedDict

from galatiq.models import Finding, Invoice


class InvoiceState(TypedDict, total=False):
    """One invoice's journey through the graph."""

    # --- ingestion ---
    source_path: str
    raw_text: str
    structural_hint: dict[str, Any] | None

    # --- extraction ---
    invoice: Invoice | None

    # --- findings ---
    #
    # `operator.add` concatenates instead of overwriting. Not yet load-bearing -- one
    # node writes findings at a time -- but it becomes the thing that makes the six
    # validation checks legal when they run concurrently. Without a reducer, parallel
    # writes to a single key raise InvalidUpdateError, and this one annotation is the
    # difference between a fan-out and a crash.
    #
    # It does mean a node running twice contributes twice, which is why findings are
    # emitted once in `finalize` rather than inside the retry loop.
    findings: Annotated[list[Finding], operator.add]

    # --- control ---
    #
    # Budgets live here and are compared in routing functions. A prompt saying "retry
    # twice" is a suggestion; an integer is not, and a document trying to talk its way
    # into more attempts has nothing to talk to.
    schema_attempts: int
    critic_attempts: int
    last_validation_error: str | None
    critique: Any | None

    # --- validation ---
    #
    # A snapshot of everything the checks are allowed to know: stock, catalog prices,
    # paid invoice numbers, today's date, FX rates. Taken once before the fan-out.
    #
    # Kept in state rather than read per-check so the audit trail answers "what stock
    # did we see when we rejected this?" rather than "what does stock say now". Plain
    # dicts, because the checkpointer serialises this and depending on how it handles
    # custom types is a dependency worth not having.
    check_context: dict[str, Any] | None

    # Deduplicated and severity-sorted. Written once, after the fan-out. The raw
    # `findings` list stays as it was so a checkpoint records what each check actually
    # said, not just the tidied summary.
    merged_findings: list[Finding]


def initial_state(source_path: str) -> InvoiceState:
    """A fresh state for one document.

    Counters start explicitly at zero rather than relying on `total=False` defaults --
    routing functions compare them on the first pass, and a missing key would make the
    budget check depend on `.get()` semantics instead of on arithmetic.
    """
    return InvoiceState(
        source_path=source_path,
        raw_text="",
        structural_hint=None,
        invoice=None,
        findings=[],
        schema_attempts=0,
        critic_attempts=0,
        last_validation_error=None,
        critique=None,
        check_context=None,
        merged_findings=[],
    )
