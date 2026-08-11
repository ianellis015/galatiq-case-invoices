"""HTTP over the pipeline.

    uv run uvicorn galatiq.api:app --port 8000

**This changes nothing about how the system works.** It is a second way in, for people who
would rather not use a terminal. The CLI is untouched, the graph is untouched, and a batch
run through here reaches identical decisions to one run through `main.py` -- the same
nodes in the same order against the same rules. The only difference is that somebody can
watch, and that what happened is written down.

State lives as long as the process does. Decisions survive a refresh or a closed tab,
because they are in SQLite rather than in a browser; they do not survive a restart,
because the service clears its stores on startup and a demonstration should open on a
full inbox.
"""

import json
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from typing import Any, AsyncIterator, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from galatiq import logs
from galatiq.api.batches import manager
from galatiq.cli.runner import RunOptions, resume_document
from galatiq.config import AUDIT_DB_PATH, DEFAULT_CONCURRENCY, INVOICE_DIR
from pathlib import Path

from galatiq.loaders import discover, load
from galatiq.mapping import hint_to_invoice
from galatiq.payment import execute_payment
from galatiq.policy import usd_total_of
from galatiq.llm import LLMConfigError, get_client
from galatiq.models import (
    ApprovalDecision,
    Invoice,
    Outcome,
    PaymentStatus,
    RunRecord,
)
from galatiq.store.db import connection
from galatiq.store.repository import (
    get_run,
    get_runs,
    latest_batch_id,
    save_run,
    update_run,
)
from galatiq.store.seed import seed_inventory

log = logs.logger(__name__)

def _clear_state() -> None:
    """Put the system back to the position it seeds from.

    Inventory is re-seeded to its known stock levels, the ledger is emptied so no
    invoice looks already-paid, and the run records are dropped so the inbox is full
    again.

    The checkpointer's database goes too, and it has to. Threads are keyed by source
    path, so a checkpoint left over from a previous session is a graph that the *same
    document* would resume into rather than start fresh -- and if that graph was
    suspended at a hold, it is waiting on a human for a run record that no longer
    exists. Clearing three of the four stores is worse than clearing none.
    """
    with connection() as conn:
        seed_inventory(conn)
        conn.execute("DELETE FROM ledger")
        conn.execute("DELETE FROM runs")
        conn.commit()

    # Deleted rather than emptied: the tables in there are LangGraph's, and I would
    # rather not hard-code assumptions about their names. Callers reset before starting
    # work, never during it, so nothing is holding the file open.
    AUDIT_DB_PATH.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Start every server from the same place.

    State lasts as long as the process does -- decisions survive a page refresh, a
    navigation, a closed tab -- but not past a restart. That boundary is deliberate:
    this is a demonstration, and a demonstration that opens onto sixteen invoices
    somebody already dealt with yesterday is not demonstrating anything.

    I run this without `--reload` when demonstrating it. Reload restarts the process on
    every file save, and every restart lands here and wipes the board.
    """
    # Verbose, because the terminal running the server is where somebody watches a batch
    # happen. The CLI has a report to protect; this has nothing to bury.
    path = logs.configure(verbose=True)

    _clear_state()
    log.info("ready  invoices=%s  log=%s", INVOICE_DIR, path)
    yield


app = FastAPI(
    title="galatiq",
    description="Invoice processing: ingest, validate, approve, pay.",
    lifespan=lifespan,
)

# The frontend runs on its own port in development. Narrow rather than "*", because a
# permissive default has a way of surviving into places it should not.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class StartBatch(BaseModel):
    invoice_path: str = str(INVOICE_DIR)
    as_of: str | None = None
    concurrency: int = DEFAULT_CONCURRENCY

    # Clear the ledger and prior run records first. Default on, because the second
    # demo run would otherwise reject everything as a duplicate of the first --
    # correct behaviour, and a terrible first impression. The interface says so
    # rather than doing it quietly.
    reset: bool = True


class Review(BaseModel):
    verdict: str  # "approve" | "deny"


class ManualDecision(BaseModel):
    path: str
    verdict: str  # "approve" | "deny"
    note: str = ""


def _options(as_of: str | None = None, concurrency: int = DEFAULT_CONCURRENCY) -> RunOptions:
    """Build run options, failing early if there is no model to call."""
    try:
        client = get_client()
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RunOptions(
        client=client,
        as_of=date.fromisoformat(as_of) if as_of else date.today(),
        # Held invoices suspend rather than terminate, so the review endpoint has
        # something to resume. Same behaviour as `--interactive` in the CLI.
        interactive=True,
        concurrency=concurrency,
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Is the service up, and does it have a model to call?

    The key check matters: a frontend that renders a "Run batch" button against a
    service with no credentials is a button that fails on click for no visible reason.
    """
    try:
        client = get_client()
        return {"ok": True, "provider": client.provider, "model": client.model}
    except LLMConfigError as exc:
        return {"ok": False, "detail": str(exc)}


@app.get("/api/documents")
def documents(path: str | None = None) -> dict[str, Any]:
    """Whatever is in the invoice directory, right now.

    Discovered rather than listed, so nothing about any particular set of files is
    baked into the service or the interface. Point it at a different folder and that
    folder is what appears.

    The interface needs this to draw its grid before anything has run -- the tiles
    exist, waiting, before the first document is read.
    """
    target = path or str(INVOICE_DIR)

    try:
        found = discover(target)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "path": target,
        "documents": [
            {
                "path": str(p),
                "name": p.name,
                "format": p.suffix.lstrip(".").lower() or "unknown",
                "bytes": p.stat().st_size,
            }
            for p in found
        ],
    }


@app.get("/api/documents/preview")
def preview(path: str) -> dict[str, Any]:
    """The document itself, so a person can read it before deciding.

    No model involved -- this is the loader, which is free and instant. Somebody choosing
    to review an invoice themselves should not have to wait for an agent to finish
    reading it first.
    """
    document = load(path)

    return {
        "path": path,
        "name": Path(path).name,
        "format": document.source_format,
        "text": document.raw_text,
        "structured": document.structural_hint,
        "findings": [json.loads(f.model_dump_json()) for f in document.findings],
    }


@app.post("/api/documents/decide")
def decide_manually(body: ManualDecision) -> dict[str, Any]:
    """Record a decision a person made without running the agents.

    The third path, alongside "let the agents look at this one" and "let them do all of
    them". Somebody who already knows this vendor and this amount should not have to wait
    two minutes to say so.

    **Payment needs numbers, and a document nobody extracted has none.** Where a
    structural parser recognised the file we have an invoice number and a total, and an
    approval pays through the same idempotent ledger write everything else uses. Where it
    did not -- a PDF, an email -- the decision is recorded and the record says plainly
    that no payment was made, rather than inventing an amount.
    """
    document = load(body.path)
    verdict = body.verdict.strip().lower()

    if verdict not in ("approve", "deny"):
        raise HTTPException(status_code=400, detail="verdict must be 'approve' or 'deny'")

    invoice = (
        hint_to_invoice(
            document.structural_hint,
            source_path=body.path,
            source_format=document.source_format,
        )
        if document.has_hint
        else Invoice(source_path=body.path, source_format=document.source_format)
    )

    outcome = Outcome.APPROVED if verdict == "approve" else Outcome.REJECTED
    decision = ApprovalDecision(
        outcome=outcome,
        rationale=(
            f"Decided by a reviewer without automated checks. {body.note}".strip()
            if body.note
            else "Decided by a reviewer without automated checks."
        ),
        risk_score=0,
    )

    payment = None
    if outcome is Outcome.APPROVED:
        with connection() as conn:
            payment = execute_payment(
                conn, invoice, decision, provider="human", model="manual review"
            )

    record = RunRecord(
        source_path=body.path,
        source_format=document.source_format,
        invoice_number=invoice.invoice_number,
        vendor=invoice.vendor or None,
        usd_total=usd_total_of(invoice),
        outcome=outcome,
        rationale=decision.rationale
        + (
            ""
            if payment and payment.status is PaymentStatus.PAID
            else " No payment was made: this document was never extracted, so it has no"
            " amount to pay."
            if outcome is Outcome.APPROVED
            else ""
        ),
        risk_score=0,
        findings=document.findings,
        payment_status=payment.status if payment else PaymentStatus.NOT_ATTEMPTED,
        invoice=invoice,
        provider="human",
        model="manual review",
    )

    with connection() as conn:
        batch = latest_batch_id(conn) or "manual"
        save_run(conn, batch, record)

    return json.loads(record.model_dump_json())


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    """Clear payment history and prior runs.

    Destructive, and deliberately explicit. The same clearing the service does at
    startup, exposed so a batch can ask for it without a restart.
    """
    _clear_state()
    return {"ok": True}


@app.post("/api/runs")
def start_batch(body: StartBatch) -> dict[str, Any]:
    """Start processing, and return immediately.

    Returns a batch id rather than results. The work takes minutes, and a request that
    blocks for minutes is one the browser gives up on.
    """
    if body.reset:
        reset()

    try:
        batch = manager.start(body.invoice_path, _options(body.as_of, body.concurrency))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"batch_id": batch.id, "total": batch.total}


@app.get("/api/stream/{batch_id}")
def stream(batch_id: str) -> StreamingResponse:
    """Server-sent events for a running batch.

    Replays from the beginning before following live, so a client that connects a moment
    after starting the batch -- which is every client -- does not miss the first
    documents.
    """
    batch = manager.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"no such batch: {batch_id}")

    def events() -> Iterator[str]:
        for event in batch.follow():
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nginx and friends buffer by default, which turns a live stream into one
            # large response at the end.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/runs")
def list_runs(batch_id: str | None = None) -> list[dict[str, Any]]:
    """Every record from a batch, most recent batch by default."""
    with connection() as conn:
        return get_runs(conn, batch_id)


@app.get("/api/runs/{run_id}")
def read_run(run_id: int) -> dict[str, Any]:
    """One record: the decision, its findings, and the invoice behind it."""
    with connection() as conn:
        record = get_run(conn, run_id)

    if record is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return record


@app.post("/api/runs/{run_id}/review")
def review(run_id: int, body: Review) -> dict[str, Any]:
    """Resume a held invoice with a human's decision.

    The graph picks up inside the hold node exactly where it paused, so approving pays
    through the same idempotent ledger write an automatic approval would. Nothing is
    recomputed and no model is called.
    """
    verdict = body.verdict.strip().lower()
    if verdict not in ("approve", "deny"):
        raise HTTPException(
            status_code=400, detail="verdict must be 'approve' or 'deny'"
        )

    with connection() as conn:
        record = get_run(conn, run_id)

    if record is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    if not record.get("awaiting_review"):
        raise HTTPException(
            status_code=409, detail="this invoice is not waiting for review"
        )

    updated = resume_document(record["source_path"], verdict, _options())

    with connection() as conn:
        update_run(conn, run_id, updated)

    return {"id": run_id, **json.loads(updated.model_dump_json())}


@app.get("/api/summary")
def summary(batch_id: str | None = None) -> dict[str, Any]:
    """The counts and totals, computed the same way the CLI computes them."""
    with connection() as conn:
        records = get_runs(conn, batch_id)

    def count(outcome: Outcome) -> int:
        return sum(1 for r in records if r.get("outcome") == outcome)

    # Rejected only. A held invoice is pending a human, not prevented -- counting it
    # would inflate the figure, and an inflated figure is the kind of thing a reader
    # checks once and then stops trusting the rest of the page.
    prevented = sum(
        Decimal(r["usd_total"])
        for r in records
        if r.get("outcome") == Outcome.REJECTED and r.get("usd_total")
    )

    return {
        "documents": len(records),
        "unique_invoices": len({r["invoice_number"] for r in records if r.get("invoice_number")}),
        "approved": count(Outcome.APPROVED),
        "held": count(Outcome.HELD_FOR_REVIEW),
        "rejected": count(Outcome.REJECTED),
        "awaiting_review": sum(1 for r in records if r.get("awaiting_review")),
        "prevented_usd": str(prevented),
        "elapsed_ms": sum(r.get("latency_ms", 0) for r in records),
    }


__all__ = ["app"]
