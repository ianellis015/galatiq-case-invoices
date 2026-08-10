"""The pipeline as a state graph.

    START -> load -> extract <-> extract_critic -> finalize -> END
                       (<=2)         (<=2)

A chain of four function calls would handle the happy path. The graph earns its place
on three counts, and the first one is already here:

  * **Cycles.** `extract <-> extract_critic` loops backwards. A chain cannot.
  * **Checkpointing.** State is snapshotted after every node, which is the audit trail
    and the resume mechanism, for the cost of one argument.
  * **Interrupts.** A durable pause for human review, when approval exists.

Later tickets add `normalize`, the six checks, `approve <-> approval_critic`, `route`,
`pay` and `reject`. END moves right; nothing here gets rebuilt.

Budgets live in the routing functions below, not in prompts. They are plain functions of
a dict -- testable with no graph and no model, and nothing inside a document can argue
with an integer comparison.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from langgraph.graph import END, START, StateGraph

from galatiq.agents.extract_critic import critique_extraction, findings_for
from galatiq.agents.extractor import extract_invoice
from galatiq.amounts import unparsed_amount_findings
from galatiq.config import AUDIT_DB_PATH, MAX_CRITIC_ATTEMPTS, MAX_SCHEMA_ATTEMPTS
from galatiq.llm import LLMClient
from galatiq.loaders import load
from galatiq.mapping import compare_to_hint
from galatiq.models import Finding, FindingCode, Invoice, Severity
from galatiq.state import InvoiceState, initial_state


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


def build_graph(client: LLMClient, *, checkpointer: Any | None = None):
    """Compile the extraction pipeline.

    `checkpointer` is optional so tests can run in memory. In real use it is a
    SqliteSaver, and passing it is the whole cost of durable state, replay and resume.
    """
    builder = StateGraph(InvoiceState)

    builder.add_node("load", load_node)
    builder.add_node("extract", make_extract_node(client))
    builder.add_node("extract_critic", make_critic_node(client))
    builder.add_node("finalize", finalize_node)

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

    builder.add_edge("finalize", END)

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
