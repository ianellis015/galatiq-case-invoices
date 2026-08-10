"""Tests for batch execution and the review queue.

The interesting property is that a held invoice does not block. `interrupt()` suspends the
graph and returns, so a batch runs at full speed and the held ones are picked up
afterwards from exactly where they paused.
"""

import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from galatiq.cli.runner import (
    RunOptions,
    process_batch,
    process_document,
    resume_document,
)
from galatiq.config import PROJECT_ROOT
from galatiq.models import Invoice, LineItem, Outcome, PaymentStatus
from galatiq.store.db import connect, init_db
from galatiq.store.seed import seed_inventory

from conftest import AutoLLM, FakeLLM

INVOICES = PROJECT_ROOT / "data" / "invoices"
ADVERSARIAL = PROJECT_ROOT / "data" / "adversarial"
TODAY = date(2026, 2, 15)


@pytest.fixture
def workspace(tmp_path):
    """A seeded inventory database and a fresh audit database."""
    db = tmp_path / "invoices.db"
    conn = connect(db)
    init_db(conn)
    seed_inventory(conn)
    conn.close()
    return db, tmp_path / "audit.db"


def options(workspace, client, **kwargs):
    db, audit = workspace
    return RunOptions(
        client=client,
        connect_db=lambda: connect(db),
        audit_db=audit,
        as_of=TODAY,
        **kwargs,
    )


def clean_invoice(number="INV-CLEAN", total="2500.00"):
    return Invoice(
        invoice_number=number,
        vendor="Widgets Inc.",
        total=total,
        currency="USD",
        due_date_raw="2026-03-01",
        line_items=[LineItem(raw_name="WidgetA", quantity=10, unit_price="250.00")],
    )


class TestSingleDocument:
    def test_a_clean_invoice_is_paid(self, workspace):
        client = AutoLLM(invoice=clean_invoice())

        record = process_document(INVOICES / "invoice_1001.txt", options(workspace, client))

        assert record.outcome is Outcome.APPROVED
        assert record.payment_status is PaymentStatus.PAID
        assert record.error is None

    def test_the_record_carries_what_the_summary_needs(self, workspace):
        client = AutoLLM(invoice=clean_invoice())

        record = process_document(INVOICES / "invoice_1001.txt", options(workspace, client))

        assert record.invoice_number == "INV-CLEAN"
        assert record.vendor == "Widgets Inc."
        assert record.usd_total == Decimal("2500.00")
        assert record.provider == "fake"
        assert record.latency_ms >= 0

    def test_a_failing_document_becomes_a_record_not_an_exception(self, workspace):
        """A batch of twenty must not stop because one file surprised us."""
        record = process_document(Path("/nonexistent/invoice.txt"), options(workspace, AutoLLM()))

        assert record.error is not None
        assert record.outcome is None


class TestHeldInvoices:
    def test_a_held_invoice_suspends_rather_than_blocking(self, workspace):
        """The property the whole review-at-the-end design rests on.

        `interrupt()` persists state and returns. If it blocked, a batch would stop on
        the first large invoice until somebody came back from lunch.
        """
        client = AutoLLM(invoice=clean_invoice(total="50000.00"))

        record = process_document(
            INVOICES / "invoice_1001.txt", options(workspace, client, interactive=True)
        )

        assert record.awaiting_review
        assert record.outcome is Outcome.HELD_FOR_REVIEW
        assert record.payment_status is None

    def test_approving_resumes_and_pays(self, workspace):
        opts = options(workspace, AutoLLM(invoice=clean_invoice(total="50000.00")), interactive=True)
        path = str(INVOICES / "invoice_1001.txt")

        held = process_document(path, opts)
        assert held.awaiting_review

        resumed = resume_document(path, "approve", opts)

        assert resumed.outcome is Outcome.APPROVED
        assert resumed.payment_status is PaymentStatus.PAID

    def test_denying_records_a_rejection(self, workspace):
        opts = options(workspace, AutoLLM(invoice=clean_invoice(total="50000.00")), interactive=True)
        path = str(INVOICES / "invoice_1001.txt")

        process_document(path, opts)
        resumed = resume_document(path, "deny", opts)

        assert resumed.outcome is Outcome.REJECTED
        assert resumed.payment_status is PaymentStatus.NOT_ATTEMPTED

    def test_resuming_costs_no_model_calls(self, workspace):
        """The graph picks up inside the hold node with everything intact. Nothing is
        recomputed, so a review costs the reviewer's attention and nothing else."""
        client = AutoLLM(invoice=clean_invoice(total="50000.00"))
        opts = options(workspace, client, interactive=True)
        path = str(INVOICES / "invoice_1001.txt")

        process_document(path, opts)
        calls_before = client.call_count

        resume_document(path, "approve", opts)

        assert client.call_count == calls_before

    def test_without_interactive_a_hold_is_terminal(self, workspace):
        client = AutoLLM(invoice=clean_invoice(total="50000.00"))

        record = process_document(
            INVOICES / "invoice_1001.txt", options(workspace, client, interactive=False)
        )

        assert record.outcome is Outcome.HELD_FOR_REVIEW
        assert not record.awaiting_review


class TestBatch:
    def _paths(self):
        return [
            INVOICES / "invoice_1001.txt",
            INVOICES / "invoice_1004.json",
            INVOICES / "invoice_1006.csv",
        ]

    def test_every_document_produces_a_record(self, workspace):
        client = AutoLLM()

        result = process_batch(self._paths(), options(workspace, client, concurrency=3))

        assert len(result.records) == 3

    def test_results_are_in_input_order(self, workspace):
        """Work happens in whatever order the pool finishes it. A report whose row order
        changes between runs has a meaningless diff."""
        paths = self._paths()

        result = process_batch(paths, options(workspace, AutoLLM(), concurrency=3))

        assert [r.source_path for r in result.records] == [str(p) for p in paths]

    def test_concurrency_does_not_change_the_answers(self, workspace, tmp_path):
        """Inventory is read-only and validation is a pure function of a snapshot, so
        parallel and serial runs must agree. That was decided in the first ticket for
        correctness; this is where it pays for itself."""
        paths = self._paths()

        serial = process_batch(paths, options(workspace, AutoLLM(), concurrency=1))

        db2 = tmp_path / "second.db"
        conn = connect(db2)
        init_db(conn)
        seed_inventory(conn)
        conn.close()
        parallel = process_batch(
            paths,
            RunOptions(
                client=AutoLLM(),
                connect_db=lambda: connect(db2),
                audit_db=tmp_path / "audit2.db",
                as_of=TODAY,
                concurrency=4,
            ),
        )

        assert [r.outcome for r in serial.records] == [
            r.outcome for r in parallel.records
        ]

    def test_one_bad_document_does_not_fail_the_batch(self, workspace):
        paths = [
            INVOICES / "invoice_1001.txt",
            ADVERSARIAL / "invoice_A002.bin",
            INVOICES / "invoice_1004.json",
        ]

        result = process_batch(paths, options(workspace, AutoLLM(), concurrency=3))

        assert len(result.records) == 3
        assert all(r.error is None for r in result.records)

    def test_unique_invoices_differs_from_document_count(self, workspace):
        """INV-1011 arrives as a PDF and a text file. Two documents, one invoice.

        The summary reports both numbers: saying "20 invoices" would be wrong and saying
        "16" would hide work that was done.
        """
        client = AutoLLM(invoice=clean_invoice("INV-1011"))
        paths = [INVOICES / "invoice_1011.pdf", INVOICES / "invoice_1011.txt"]

        result = process_batch(paths, options(workspace, client, concurrency=2))

        assert len(result.records) == 2
        assert result.unique_invoices == 1

    def test_held_invoices_are_collected(self, workspace):
        opts = options(workspace, AutoLLM(), interactive=True, concurrency=2)

        result = process_batch(
            [INVOICES / "invoice_1001.txt", INVOICES / "invoice_1004.json"], opts
        )

        assert all(r.awaiting_review for r in result.held)


class TestAsOf:
    def test_the_reference_date_changes_what_the_checks_report(self, workspace):
        """The corpus is dated January 2026, so without this everything reports as
        past due and the signal is worthless."""
        from galatiq.models import FindingCode

        client = AutoLLM(invoice=clean_invoice())
        record = process_document(
            INVOICES / "invoice_1001.txt",
            options(workspace, client, interactive=False),
        )
        codes_before = [f.code for f in record.findings]

        late = options(workspace, AutoLLM(invoice=clean_invoice()), interactive=False)
        late.as_of = date(2027, 1, 1)
        record_late = process_document(INVOICES / "invoice_1002.txt", late)

        assert FindingCode.DATE_PAST_DUE not in codes_before
        assert FindingCode.DATE_PAST_DUE in [f.code for f in record_late.findings]
