"""Running documents through the graph, one or many.

Two things here that are not obvious from the outside.

**Held invoices do not block.** `interrupt()` suspends the graph, persists its state, and
returns -- so a batch runs at full speed and every held invoice comes back marked
`awaiting_review` rather than occupying a worker while somebody thinks. They are resumed
afterwards, each against its own thread, from exactly where it paused.

**Concurrency is safe because the pipeline was built for it.** Inventory is read-only and
validation is a pure function of the invoice plus a snapshot, which is what makes twenty
documents in any order produce the same twenty answers. That decision was made in the
first ticket for correctness reasons; this is where it pays for itself.

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


@contextmanager
def _graph(options: RunOptions) -> Iterator[Any]:
    """A compiled graph with its own checkpointer connection.

    Built per call rather than shared, because SqliteSaver holds a connection and
    sqlite3 connections are not safe to use from several threads at once.
    """
    path = Path(options.audit_db) if options.audit_db else AUDIT_DB_PATH
    ensure_var_dir(path)

    with SqliteSaver.from_conn_string(str(path)) as saver:
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
            state = graph.invoke(
                initial_state(source_path), config=thread_for(source_path)
            )
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
        risk_score=decision.risk_score if decision else 0,
        findings=list(state.get("merged_findings") or state.get("findings") or []),
        payment_status=payment.status if payment else None,
        awaiting_review=awaiting,
        latency_ms=latency_ms,
        provider=getattr(options.client, "provider", None),
        model=getattr(options.client, "model", None),
    )
