"""The pipeline as a state graph.

    START -> load -> extract <-> extract_critic -> finalize -> prepare_checks
                                                                    |
                                                                normalize
                                                                    |
                                             [ eight checks, all at once ]
                                                                    |
                                                             merge_findings
                                                                    |
                                                    approve <-> approval_critic
                                                                    |
                                                        pay | reject | hold

A chain of function calls would handle the happy path. I used a graph for three things
a chain cannot do:

  * **Cycles.** `extract <-> extract_critic` and `approve <-> approval_critic` both loop
    backwards. That is the self-correction, and a chain has no way to express it.
  * **Checkpointing.** State is snapshotted after every node, which gives me the audit
    trail and the resume mechanism for the cost of one argument.
  * **Interrupts.** `hold` durably suspends a run for a human and resumes it later,
    which is the honest implementation of "obtain VP approval".

The fan-out is written as a loop over the check registry rather than as eight hand-wired
pairs of edges, so adding a check means adding it to the registry and nothing else.

Budgets live in the routing functions below, not in prompts. They are plain functions of
a dict -- testable with no graph and no model, and nothing inside a document can argue
with an integer comparison.
"""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from langgraph.graph import END, START, StateGraph

from galatiq.agents.approval_critic import critique_decision
from galatiq.agents.approver import decide, revise
from galatiq.agents.extract_critic import critique_extraction, findings_for
from galatiq.agents.extractor import extract_invoice
from galatiq.agents.normalizer import normalize_invoice
from galatiq.amounts import unparsed_amount_findings
from galatiq.checks import (
    Check,
    CheckContext,
    all_checks,
    merge_findings,
    run_check,
    snapshot,
)
from galatiq.config import (
    AUDIT_DB_PATH,
    MAX_CRITIC_ATTEMPTS,
    MAX_REFLECT_ATTEMPTS,
    MAX_SCHEMA_ATTEMPTS,
)
from galatiq.llm import LLMClient
from galatiq.loaders import load
from galatiq.mapping import compare_to_hint
from galatiq.models import (
    Finding,
    FindingCode,
    Invoice,
    Outcome,
    PaymentResult,
    PaymentStatus,
    Severity,
)
from galatiq.payment import execute_payment
from galatiq.policy import evaluate
from galatiq.state import InvoiceState, initial_state
from galatiq.store.db import connect


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def load_node(state: InvoiceState) -> dict[str, Any]:
    """Read the file. Deterministic, no model.

    Load-time problems arrive as findings rather than exceptions, so a binary file or
    an unknown format flows through the same pipeline as a well-formed invoice and
    reaches a human with a reason attached.
    """
    document = load(state["source_path"])

    return {
        "raw_text": document.raw_text,
        "structural_hint": document.structural_hint,
        "findings": document.findings,
    }


def make_extract_node(client: LLMClient):
    """Build the extract node bound to a client.

    A closure rather than a global: tests inject a double, and the graph has no opinion
    about where the model came from.
    """

    def extract_node(state: InvoiceState) -> dict[str, Any]:
        outcome = extract_invoice(
            client,
            raw_text=state.get("raw_text", ""),
            structural_hint=state.get("structural_hint"),
            validation_error=state.get("last_validation_error"),
            critique=state.get("critique"),
            source_path=state.get("source_path"),
            source_format=_format_of(state.get("source_path")),
        )

        if outcome.succeeded:
            return {
                "invoice": outcome.invoice,
                "last_validation_error": None,
            }

        attempts = state.get("schema_attempts", 0) + 1

        # Budget spent. Produce an empty invoice rather than leaving the pipeline with
        # nothing: the document still has to reach a decision, and "we could not read
        # this" is a decision a human can act on.
        if attempts >= MAX_SCHEMA_ATTEMPTS:
            return {
                "invoice": Invoice(
                    source_path=state.get("source_path"),
                    source_format=_format_of(state.get("source_path")),
                ),
                "schema_attempts": attempts,
                "last_validation_error": outcome.validation_error,
            }

        return {
            "invoice": None,
            "schema_attempts": attempts,
            "last_validation_error": outcome.validation_error,
        }

    return extract_node


def make_critic_node(client: LLMClient):
    """Build the critic node bound to a client."""

    def critic_node(state: InvoiceState) -> dict[str, Any]:
        invoice = state.get("invoice")
        raw_text = state.get("raw_text", "")

        # Nothing to audit: an unreadable document, or an extraction that spent its
        # budget and fell back to an empty invoice. Auditing a structure the extractor
        # never managed to produce would cost a call to confirm what is already known.
        #
        # Skipping is not a failure. The deterministic checks run either way, and they
        # do not depend on the model having been reachable.
        if invoice is None or not raw_text.strip():
            return {}
        if state.get("last_validation_error"):
            return {}

        critique = critique_extraction(client, raw_text=raw_text, invoice=invoice)

        update: dict[str, Any] = {"critique": critique}

        # The counter increments only when the critic actually sends work back. A
        # PARSE_SOUND verdict costs nothing and should not consume a budget that exists
        # to bound re-reading.
        if critique.suspects_misparse:
            update["critic_attempts"] = state.get("critic_attempts", 0) + 1

        return update

    return critic_node


def finalize_node(state: InvoiceState) -> dict[str, Any]:
    """Emit the extraction phase's findings, once.

    Deliberately not inside the loop. `findings` concatenates, so a node that runs
    three times contributes three copies of the same finding -- and an invoice whose
    audit trail says the same thing three times reads as three problems.

    Everything reported here is a fact about the finished extraction rather than about
    an attempt along the way.
    """
    invoice = state.get("invoice")
    findings: list[Finding] = []

    critique = state.get("critique")
    if critique is not None:
        findings.extend(findings_for(critique))

    if invoice is not None:
        findings.extend(unparsed_amount_findings(invoice))
        findings.extend(compare_to_hint(invoice, state.get("structural_hint")))

    # The extractor never produced a valid structure. The specific validation failure
    # is the evidence -- it is what a human needs in order to tell a broken document
    # from a broken prompt.
    if state.get("last_validation_error"):
        findings.append(
            Finding(
                code=FindingCode.NEEDS_HUMAN_REVIEW,
                severity=Severity.CRITICAL,
                message=(
                    "Extraction did not produce a valid structure within "
                    f"{MAX_SCHEMA_ATTEMPTS} attempts."
                ),
                evidence=str(state["last_validation_error"])[:500],
            )
        )

    # The critic still suspected a misread when the re-read budget ran out. The
    # extraction is reported as uncertain rather than trusted or discarded.
    if (
        critique is not None
        and critique.suspects_misparse
        and state.get("critic_attempts", 0) >= MAX_CRITIC_ATTEMPTS
    ):
        findings.append(
            Finding(
                code=FindingCode.EXTRACTION_UNCERTAIN,
                severity=Severity.WARN,
                message=(
                    "The extraction critic still suspected a misread after "
                    f"{MAX_CRITIC_ATTEMPTS} re-reads."
                ),
                evidence=critique.reasoning,
            )
        )

    return {"findings": findings}


def make_prepare_checks_node(connect_db: Callable[[], Any], as_of: date | None):
    """Snapshot everything the checks are allowed to know, once.

    Before the fan-out rather than inside it: eight concurrent database reads become
    one, the checks become pure functions of explicit data, and the snapshot lands in
    the checkpointed state so the audit trail records the stock position that produced
    the decision rather than whatever stock says when someone reads it later.
    """

    def prepare_checks_node(state: InvoiceState) -> dict[str, Any]:
        conn = connect_db()
        try:
            return {"check_context": snapshot(conn, today=as_of).to_snapshot()}
        finally:
            conn.close()

    return prepare_checks_node


def make_normalize_node(client: LLMClient | None):
    """Build the normalize node.

    The client is optional. Deterministic matching handles every name in the corpus, and
    an unreachable model leaves unresolved names unresolved -- which is the same outcome
    as a genuine non-match, and the safe direction to degrade in.
    """

    def normalize_node(state: InvoiceState) -> dict[str, Any]:
        invoice = state.get("invoice")
        context = state.get("check_context")

        if invoice is None or context is None:
            return {}

        return {
            "invoice": normalize_invoice(
                invoice, context.get("stock", {}), client=client
            )
        }

    return normalize_node


def make_check_node(name: str, check: Check):
    """Wrap one check as a graph node.

    Each returns only `findings`, which is the key with a reducer -- so eight of these
    running concurrently concatenate rather than collide. Without that annotation on the
    state, parallel writes to one key raise InvalidUpdateError, and this fan-out would
    not be legal.
    """

    def check_node(state: InvoiceState) -> dict[str, Any]:
        invoice = state.get("invoice")
        raw_context = state.get("check_context")

        if invoice is None or raw_context is None:
            return {"findings": []}

        context = CheckContext.from_snapshot(raw_context)
        return {"findings": run_check(name, check, invoice, context)}

    return check_node


def make_approve_node(client: LLMClient | None):
    """Build the approve node.

    The rules engine runs here too, on every pass. It is deterministic and cheap, and
    re-evaluating means the decision is always a function of the findings as they stand
    rather than of a policy result computed three nodes ago.
    """

    def approve_node(state: InvoiceState) -> dict[str, Any]:
        invoice = state.get("invoice")
        findings = state.get("merged_findings") or state.get("findings", [])

        if invoice is None:
            return {}

        policy = evaluate(invoice, findings)
        critique = state.get("approval_critique")

        if critique is not None and getattr(critique, "found_something", False):
            decision = revise(
                client,
                invoice=invoice,
                findings=findings,
                policy=policy,
                critique=critique,
            )
        else:
            decision = decide(
                client, invoice=invoice, findings=findings, policy=policy
            )

        return {"policy": policy, "decision": decision}

    return approve_node


def make_approval_critic_node(client: LLMClient | None):
    """Build the approval critic node."""

    def approval_critic_node(state: InvoiceState) -> dict[str, Any]:
        invoice = state.get("invoice")
        decision = state.get("decision")

        if invoice is None or decision is None:
            return {}

        critique = critique_decision(
            client,
            invoice=invoice,
            findings=state.get("merged_findings") or state.get("findings", []),
            decision=decision,
        )

        update: dict[str, Any] = {"approval_critique": critique}

        # Same discipline as the extraction loop: the counter moves only when the critic
        # actually sends work back. A SOUND verdict costs nothing and should not consume
        # a budget that exists to bound re-reviewing.
        if critique.found_something:
            update["reflect_attempts"] = state.get("reflect_attempts", 0) + 1

        return update

    return approval_critic_node


def make_pay_node(connect_db: Callable[[], Any], provider: str, model: str):
    """Build the payment node.

    Not an agent. A tool call behind a deterministic edge, reached only when the rules
    and the model both concurred.
    """

    def pay_node(state: InvoiceState) -> dict[str, Any]:
        invoice = state.get("invoice")
        decision = state.get("decision")

        if invoice is None or decision is None:
            return {}

        conn = connect_db()
        try:
            payment = execute_payment(
                conn, invoice, decision, provider=provider, model=model
            )
        finally:
            conn.close()

        return {"payment": payment}

    return pay_node


def reject_node(state: InvoiceState) -> dict[str, Any]:
    """Terminal. The decision and its reasoning are already in state.

    Nothing is written here beyond a payment record saying none was attempted. The
    rejection *is* the decision plus the findings that produced it, and the checkpointer
    has already durably recorded both -- a separate rejection log would be a second copy
    of the same facts, free to drift from the first.
    """
    decision = state.get("decision")
    invoice = state.get("invoice")

    return {
        "payment": PaymentResult(
            invoice_number=(invoice.invoice_number if invoice else None) or "",
            status=PaymentStatus.NOT_ATTEMPTED,
            message=(
                f"Rejected: {decision.rationale}" if decision else "Rejected."
            ),
        )
    }


def make_hold_node(interactive: bool):
    """Build the hold node.

    `interrupt()` durably suspends the graph and persists state, so a human decision can
    arrive minutes or days later and the run continues from exactly where it paused. That
    is the honest implementation of "obtain VP approval" -- the brief describes it as a
    real step in the current process, and a simulated pause would not survive the process
    it is modelling.

    Off by default. A batch of twenty invoices should not block on the third one waiting
    for someone to come back from lunch: held invoices are recorded and the run moves on,
    and the review queue is worked through afterwards.
    """

    def hold_node(state: InvoiceState) -> dict[str, Any]:
        decision = state.get("decision")
        invoice = state.get("invoice")

        if interactive:
            from langgraph.types import interrupt

            verdict = interrupt(
                {
                    "invoice_number": invoice.invoice_number if invoice else None,
                    "rationale": decision.rationale if decision else "",
                    "findings": [
                        f.model_dump() for f in state.get("merged_findings", [])
                    ],
                }
            )
            # Both spellings, because the caller's vocabulary and the graph's do not
            # have to match and a reviewer typing "deny" should not silently do nothing.
            answer = str(verdict).strip().lower()

            if answer in ("approve", "approved", "yes", "y"):
                return {
                    "decision": decision.model_copy(
                        update={
                            "outcome": Outcome.APPROVED,
                            "rationale": f"[Approved by reviewer] {decision.rationale}",
                        }
                    )
                }
            if answer in ("deny", "denied", "reject", "rejected", "no", "n"):
                return {
                    "decision": decision.model_copy(
                        update={
                            "outcome": Outcome.REJECTED,
                            "rationale": f"[Denied by reviewer] {decision.rationale}",
                        }
                    )
                }

        return {
            "payment": PaymentResult(
                invoice_number=(invoice.invoice_number if invoice else None) or "",
                status=PaymentStatus.NOT_ATTEMPTED,
                message="Held for human review.",
            )
        }

    return hold_node


def merge_findings_node(state: InvoiceState) -> dict[str, Any]:
    """Deduplicate and order what the checks produced.

    Two checks can legitimately notice the same thing, and an audit trail that says it
    twice reads as two problems. Returning the merged list replaces nothing -- the
    reducer concatenates -- so this writes to its own key and the raw findings stay
    intact in the checkpoint for anyone reconstructing the run.
    """
    return {"merged_findings": merge_findings(state.get("findings", []))}


# ---------------------------------------------------------------------------
# Routing
#
# Plain functions of a dict. No graph, no model, no I/O -- which is what makes the
# budgets testable and what keeps them out of reach of anything a document might say.
# ---------------------------------------------------------------------------


def after_extract(state: InvoiceState) -> Literal["extract", "extract_critic"]:
    """Retry a malformed response, or move on to the audit.

    Only the *shape* of the response is in question here. Whether the content is a
    faithful reading of the document is the critic's job, and asking it before there is
    a structure to audit would be asking about nothing.
    """
    if state.get("invoice") is None:
        return "extract"
    return "extract_critic"


def after_approval_critic(
    state: InvoiceState,
) -> Literal["approve", "pay", "reject", "hold"]:
    """Reconsider, or act.

    One function handling both the loop and the terminal routing, because they are the
    same question asked once: is there more thinking to do, and if not, what happens?
    Splitting them into a routing function and a pass-through `route` node would add a
    node that decides nothing.

    The three terminals are mutually exclusive and exhaustive -- every invoice ends at
    exactly one, which is the invariant the whole system is built to.
    """
    critique = state.get("approval_critique")

    if (
        critique is not None
        and critique.found_something
        and state.get("reflect_attempts", 0) < MAX_REFLECT_ATTEMPTS
    ):
        return "approve"

    decision = state.get("decision")
    if decision is None:
        # No decision at all. Held rather than paid: whatever went wrong, a human
        # should look at it.
        return "hold"

    if decision.outcome is Outcome.APPROVED:
        return "pay"
    if decision.outcome is Outcome.REJECTED:
        return "reject"
    return "hold"


def after_hold(state: InvoiceState) -> Literal["pay", "reject", "__end__"]:
    """Where a held invoice goes once a human has answered.

    On the way in, `interrupt()` suspends the node before this runs, so a run that is
    still waiting never reaches here. On resume the node completes with the reviewer's
    verdict applied, and the invoice takes the same paths an automatic decision would --
    approving pays through the same idempotent ledger write, denying logs the same
    reasoned rejection.

    Reusing the terminals rather than paying inline is deliberate: a human approval and
    an automatic one should be indistinguishable downstream, including in the audit
    trail.
    """
    decision = state.get("decision")

    if decision is None:
        return END
    if decision.outcome is Outcome.APPROVED:
        return "pay"
    if decision.outcome is Outcome.REJECTED:
        return "reject"
    return END


def after_critic(state: InvoiceState) -> Literal["extract", "finalize"]:
    """Re-read the document, or finish.

    Three verdicts collapse to two routes, and the collapsing is the point.
    PARSE_SOUND and DOCUMENT_INCONSISTENT both mean "stop" -- the first because there
    is nothing wrong, the second because the problem is in the document and reading it
    again cannot fix it. Only MISPARSE_SUSPECTED asks for another attempt, and only
    while the budget allows.
    """
    critique = state.get("critique")

    if critique is None or not critique.suspects_misparse:
        return "finalize"
    if state.get("critic_attempts", 0) >= MAX_CRITIC_ATTEMPTS:
        return "finalize"
    return "extract"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_graph(
    client: LLMClient,
    *,
    checkpointer: Any | None = None,
    connect_db: Callable[[], Any] | None = None,
    interactive: bool = False,
    provider: str | None = None,
    model: str | None = None,
    as_of: date | None = None,
):
    """Compile the pipeline.

    `checkpointer` is optional so tests can run in memory. In real use it is a
    SqliteSaver, and passing it is the whole cost of durable state, replay and resume.

    `connect_db` is injectable so tests point the checks at a temporary database rather
    than the developer's real one.
    """
    provider = provider or getattr(client, "provider", "unknown")
    model = model or getattr(client, "model", "unknown")

    builder = StateGraph(InvoiceState)

    builder.add_node("load", load_node)
    builder.add_node("extract", make_extract_node(client))
    builder.add_node("extract_critic", make_critic_node(client))
    builder.add_node("finalize", finalize_node)
    builder.add_node(
        "prepare_checks", make_prepare_checks_node(connect_db or connect, as_of)
    )
    builder.add_node("normalize", make_normalize_node(client))
    builder.add_node("merge_findings", merge_findings_node)

    checks = all_checks()
    for name, check in checks.items():
        builder.add_node(name, make_check_node(name, check))

    builder.add_edge(START, "load")
    builder.add_edge("load", "extract")

    # The two cycles. Conditional edges pointing backwards are the entire reason this
    # is a graph and not a chain.
    builder.add_conditional_edges(
        "extract",
        after_extract,
        {"extract": "extract", "extract_critic": "extract_critic"},
    )
    builder.add_conditional_edges(
        "extract_critic",
        after_critic,
        {"extract": "extract", "finalize": "finalize"},
    )

    builder.add_edge("finalize", "prepare_checks")
    builder.add_edge("prepare_checks", "normalize")

    # The fan-out. Eight edges out of normalize and eight back into merge_findings,
    # written as a loop because the checks are interchangeable by construction -- same
    # signature in, same type out. Adding an eighth check means adding it to the
    # registry and nothing else.
    #
    # Static edges rather than `Send`, because the set of checks is known at build time.
    # `Send` is for dynamic fan-out -- one branch per line item, say -- which this design
    # does not need.
    for name in checks:
        builder.add_edge("normalize", name)
        builder.add_edge(name, "merge_findings")

    builder.add_node("approve", make_approve_node(client))
    builder.add_node("approval_critic", make_approval_critic_node(client))
    builder.add_node("pay", make_pay_node(connect_db or connect, provider, model))
    builder.add_node("reject", reject_node)
    builder.add_node("hold", make_hold_node(interactive))

    builder.add_edge("merge_findings", "approve")
    builder.add_edge("approve", "approval_critic")

    # The second cycle, and the three terminals. Every invoice ends at exactly one of
    # pay, reject or hold -- there is no path that reaches END without a decision.
    builder.add_conditional_edges(
        "approval_critic",
        after_approval_critic,
        {
            "approve": "approve",
            "pay": "pay",
            "reject": "reject",
            "hold": "hold",
        },
    )

    builder.add_edge("pay", END)
    builder.add_edge("reject", END)
    builder.add_conditional_edges(
        "hold", after_hold, {"pay": "pay", "reject": "reject", END: END}
    )

    return builder.compile(checkpointer=checkpointer)


@contextmanager
def extraction_graph(
    client: LLMClient, *, audit_db: str | Path | None = None
) -> Iterator[Any]:
    """A compiled graph with durable checkpointing.

        with extraction_graph(client) as graph:
            state = graph.invoke(initial_state(path), config=thread_for(path))

    The saver is a context manager holding a SQLite connection, which is why this is
    one too.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    from galatiq.config import ensure_var_dir

    path = Path(audit_db) if audit_db else AUDIT_DB_PATH
    ensure_var_dir(path)

    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield build_graph(client, checkpointer=saver)


def thread_for(source_path: str) -> dict[str, Any]:
    """Checkpointer config for one document.

    Threading on the source path gives every document its own independently
    inspectable timeline. Not the invoice number: that is not known until extraction
    has already run, and the twins (INV-1011 as both PDF and text) are genuinely
    different documents that should not share a history.
    """
    return {"configurable": {"thread_id": source_path}}


def run_document(client: LLMClient, source_path: str, **kwargs: Any) -> InvoiceState:
    """Convenience: load, extract, critique and finalize one document."""
    with extraction_graph(client, **kwargs) as graph:
        return graph.invoke(
            initial_state(source_path), config=thread_for(source_path)
        )


def _format_of(source_path: str | None) -> str | None:
    """The extension, as the loaders record it."""
    if not source_path:
        return None
    return Path(source_path).suffix.lstrip(".").lower() or "unknown"
