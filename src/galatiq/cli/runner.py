"""Running documents through the graph, one or many.

Two things here that are not obvious from the outside.

**Held invoices do not block.** `interrupt()` suspends the graph, persists its state, and
returns -- so a batch runs at full speed and every held invoice comes back marked
`awaiting_review` rather than occupying a worker while somebody thinks. They are resumed
afterwards, each against its own thread, from exactly where it paused.

**Concurrency is safe because the pipeline was built for it.** Inventory is read-only and
validation is a pure function of the invoice plus a snapshot, which is what makes twenty
documents in any order produce the same twenty answers. I made inventory read-only for
correctness rather than for speed; this is where that choice pays for itself.

Each worker gets its own graph and its own checkpointer connection. A shared SqliteSaver
across threads is not safe, and the failure would be intermittent -- the worst kind.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator

import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from galatiq.config import AUDIT_DB_PATH, ensure_var_dir
from galatiq.graph import build_graph, thread_for
from galatiq.models import Invoice, Outcome, RunRecord
from galatiq.policy import usd_total_of
from galatiq.state import initial_state


@dataclass
class RunOptions:
    """Everything a run needs that is not the document itself."""

    client: Any
    connect_db: Callable[[], Any] | None = None
    audit_db: Path | None = None
    as_of: date | None = None
    interactive: bool = True
    concurrency: int = 8

    # Set only by the API, so a browser can watch a batch happen. When it is None the
    # run uses `invoke` exactly as it always has -- the CLI passes nothing and behaves
    # identically. Watching changes nothing about what gets decided.
    on_event: Callable[[dict[str, Any]], None] | None = None


@dataclass
class BatchResult:
    records: list[RunRecord] = field(default_factory=list)

    @property
    def held(self) -> list[RunRecord]:
        """Suspended at a review pause, in the order they were processed."""
        return [r for r in self.records if r.awaiting_review]

    @property
    def unique_invoices(self) -> int:
        """Distinct invoice numbers.

        Different from the document count: INV-1011, 1012 and 1013 each arrive as two
        files. Reporting only one of the two numbers would either hide work or invent it.
        """
        return len({r.invoice_number for r in self.records if r.invoice_number})


def _audit_path(options: RunOptions) -> Path:
    return Path(options.audit_db) if options.audit_db else AUDIT_DB_PATH


# The types the graph puts into its own state, and the only ones that should ever come
# back out of a checkpoint.
#
# Naming them does two things. It silences the warning LangGraph prints for every
# unregistered type it deserialises -- eight of them, mid-run, in the middle of a review
# prompt. And it is the safer setting: the default permissive mode will rebuild *any*
# type named in the file, so anyone able to write to the checkpoint database could
# choose what gets constructed. An allowlist means a tampered checkpoint fails instead.
_CHECKPOINT_TYPES = (
    ("galatiq.models", "Invoice"),
    ("galatiq.models", "LineItem"),
    ("galatiq.models", "Finding"),
    ("galatiq.models", "FindingCode"),
    ("galatiq.models", "Severity"),
    ("galatiq.models", "Outcome"),
    ("galatiq.models", "PaymentStatus"),
    ("galatiq.models", "PaymentResult"),
    ("galatiq.models", "ApprovalDecision"),
    ("galatiq.policy", "PolicyOutcome"),
    ("galatiq.policy", "Rule"),
    ("galatiq.agents.extract_critic", "Critique"),
    ("galatiq.agents.extract_critic", "Discrepancy"),
    ("galatiq.agents.approval_critic", "ApprovalCritique"),
)


@contextmanager
def _saver(path: Path) -> Iterator[SqliteSaver]:
    """A checkpointer whose serializer knows our types.

    `SqliteSaver.from_conn_string` gives no way to pass a serializer, so the connection
    is opened here instead. `check_same_thread=False` matches what it does, and is safe
    because each worker builds its own.
    """
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(
            conn, serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)
        )
    finally:
        conn.close()


def prepare_checkpointer(options: RunOptions) -> None:
    """Create the checkpointer's database before any worker opens it.

    Each worker builds its own connection, and an sqlite file that does not exist yet
    gets *created* by whichever connection opens it first. With eight opening at once,
    the others can arrive mid-creation and find a file that is not yet a database --
    which surfaces as "file is not a database" or "database is locked" on a handful of
    documents while the rest of the batch succeeds.

    Doing it once, up front, means every worker opens a file that already exists and is
    already initialised. Cheap, and it removes the race rather than retrying through it.
    """
    path = _audit_path(options)
    ensure_var_dir(path)

    with _saver(path) as saver:
        saver.setup()


@contextmanager
def _graph(options: RunOptions) -> Iterator[Any]:
    """A compiled graph with its own checkpointer connection.

    Built per call rather than shared, because SqliteSaver holds a connection and
    sqlite3 connections are not safe to use from several threads at once.
    """
    path = _audit_path(options)
    ensure_var_dir(path)

    with _saver(path) as saver:
        yield build_graph(
            options.client,
            checkpointer=saver,
            connect_db=options.connect_db,
            interactive=options.interactive,
            as_of=options.as_of,
        )


def process_document(source_path: str | Path, options: RunOptions) -> RunRecord:
    """Run one document to a decision, or to a review pause.

    Exceptions become a record with an error rather than propagating. A batch of twenty
    should not stop because one file surprised us, and "this document failed and here is
    why" is information; a traceback that ends the run is not.
    """
    source_path = str(source_path)
    started = time.monotonic()

    try:
        with _graph(options) as graph:
            if options.on_event is None:
                state = graph.invoke(
                    initial_state(source_path), config=thread_for(source_path)
                )
            else:
                state = _run_watched(graph, source_path, options.on_event)
    except Exception as exc:  # noqa: BLE001 - one document must not fail the batch
        return RunRecord(
            source_path=source_path,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    return _record(source_path, state, options, int((time.monotonic() - started) * 1000))


def process_batch(paths: list[Path], options: RunOptions) -> BatchResult:
    """Run many documents concurrently, preserving input order in the results.

    Order is restored deliberately. The work happens in whatever order the pool finishes
    it, and a report that lists invoices in racing order is a report whose diff between
    two runs is meaningless.
    """
    if len(paths) == 1 or options.concurrency <= 1:
        return BatchResult(records=[process_document(p, options) for p in paths])

    # Before the pool, not inside it. See prepare_checkpointer.
    prepare_checkpointer(options)

    records: dict[str, RunRecord] = {}

    with ThreadPoolExecutor(max_workers=options.concurrency) as pool:
        futures = {
            pool.submit(process_document, path, options): str(path) for path in paths
        }
        for future in as_completed(futures):
            record = future.result()
            records[record.source_path] = record

    return BatchResult(records=[records[str(p)] for p in paths])


def resume_document(
    source_path: str, verdict: str, options: RunOptions
) -> RunRecord:
    """Continue a suspended run with a human's answer.

    The graph picks up inside the hold node exactly where `interrupt()` left it, with all
    of its state intact -- the invoice, the findings, the decision and the reasoning that
    produced it. Nothing is recomputed, so no model is called and the review costs
    nothing but the reviewer's attention.
    """
    started = time.monotonic()

    with _graph(options) as graph:
        state = graph.invoke(Command(resume=verdict), config=thread_for(source_path))

    return _record(source_path, state, options, int((time.monotonic() - started) * 1000))


def _run_watched(
    graph: Any, source_path: str, emit: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    """Run a document, reporting each step as it happens.

    Identical to `invoke` in what it produces -- same nodes, same order, same decision.
    The only difference is that somebody is watching.

    The final state is read back from the checkpointer rather than accumulated from the
    stream, because a run that stops at a human-review pause has no final chunk to
    accumulate. The checkpointer knows either way.
    """
    from galatiq.api.events import DocumentTracker

    tracker = DocumentTracker(doc=source_path)
    config = thread_for(source_path)

    emit({"type": "document.start", "doc": source_path})

    latest: dict[str, Any] = {}

    for mode, chunk in graph.stream(
        initial_state(source_path), config=config, stream_mode=["debug", "values"]
    ):
        if mode == "values":
            latest = chunk
            continue

        payload = chunk.get("payload", {})
        name = payload.get("name")
        if not name:
            continue

        if chunk.get("type") == "task":
            for event in tracker.on_task(name):
                emit(event)
        elif chunk.get("type") == "task_result":
            for event in tracker.on_task_result(name, latest):
                emit(event)

    snapshot = graph.get_state(config)
    state = dict(snapshot.values)

    # A suspended run has somewhere left to go. `_record` reads this to decide whether
    # the document is waiting for a person or genuinely finished.
    if snapshot.next:
        state["__interrupt__"] = True

    return state


def _record(
    source_path: str, state: dict[str, Any], options: RunOptions, latency_ms: int
) -> RunRecord:
    """Turn a finished or suspended run into a record."""
    invoice: Invoice | None = state.get("invoice")
    decision = state.get("decision")
    payment = state.get("payment")

    # LangGraph marks a suspended run this way. Its presence is the difference between
    # "held, and waiting for someone" and "held, and that was the final answer".
    awaiting = "__interrupt__" in state

    return RunRecord(
        source_path=source_path,
        source_format=invoice.source_format if invoice else None,
        invoice_number=invoice.invoice_number if invoice else None,
        vendor=(invoice.vendor or None) if invoice else None,
        usd_total=usd_total_of(invoice) if invoice else None,
        outcome=(
            Outcome.HELD_FOR_REVIEW
            if awaiting
            else (decision.outcome if decision else None)
        ),
        rationale=decision.rationale if decision else "",
        policy_refs=list(decision.policy_refs) if decision else [],
        concerns=list(decision.concerns) if decision else [],
        risk_score=decision.risk_score if decision else 0,
        findings=list(state.get("merged_findings") or state.get("findings") or []),
        payment_status=payment.status if payment else None,
        invoice=invoice,
        awaiting_review=awaiting,
        latency_ms=latency_ms,
        provider=getattr(options.client, "provider", None),
        model=getattr(options.client, "model", None),
    )
