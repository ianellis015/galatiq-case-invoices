"""Tests for the held-invoice review queue.

This is the durable-pause feature made usable: the batch runs to completion, held
invoices come back suspended, and a human works through them afterwards with everything
still attached.

The prompt is stubbed. What is being tested is the loop around it — that a verdict
reaches the graph, that the invoice resumes to the right terminal, and that skipping
leaves a run suspended for a later session rather than silently discarding it.
"""

from datetime import date

import pytest

from galatiq.cli import review as review_module
from galatiq.cli.review import review_queue
from galatiq.cli.runner import RunOptions, process_document
from galatiq.config import PROJECT_ROOT
from galatiq.models import Invoice, LineItem, Outcome, PaymentStatus
from galatiq.store.db import connect, init_db
from galatiq.store.repository import get_payment
from galatiq.store.seed import seed_inventory

from conftest import AutoLLM

INVOICES = PROJECT_ROOT / "data" / "invoices"
TODAY = date(2026, 2, 15)


@pytest.fixture
def workspace(tmp_path):
    db = tmp_path / "invoices.db"
    conn = connect(db)
    init_db(conn)
    seed_inventory(conn)
    conn.close()
    return db, tmp_path / "audit.db"


def large_invoice(number="INV-BIG"):
    """Clean, and over the threshold — so the rules hold it for a human."""
    return Invoice(
        invoice_number=number,
        vendor="Widgets Inc.",
        total="50000.00",
        currency="USD",
        due_date_raw="2026-03-01",
        line_items=[LineItem(raw_name="WidgetA", quantity=10, unit_price="250.00")],
    )


def options(workspace, client):
    db, audit = workspace
    return RunOptions(
        client=client,
        connect_db=lambda: connect(db),
        audit_db=audit,
        as_of=TODAY,
        interactive=True,
    )


def answers(monkeypatch, *responses):
    """Stub the prompt with a scripted sequence of verdicts."""
    queue = list(responses)
    monkeypatch.setattr(
        review_module.Prompt, "ask", staticmethod(lambda *a, **k: queue.pop(0))
    )


def held_record(workspace, opts, path="invoice_1001.txt", number="INV-BIG"):
    opts.client = AutoLLM(invoice=large_invoice(number))
    record = process_document(INVOICES / path, opts)
    assert record.awaiting_review, "expected the invoice to be held"
    return record


class TestEmptyQueue:
    def test_nothing_to_review(self, workspace):
        assert review_queue([], options(workspace, AutoLLM())) == []


class TestApproving:
    def test_it_resumes_and_pays(self, workspace, monkeypatch):
        opts = options(workspace, AutoLLM())
        record = held_record(workspace, opts)
        answers(monkeypatch, "approve")

        resumed = review_queue([record], opts)

        assert resumed[0].outcome is Outcome.APPROVED
        assert resumed[0].payment_status is PaymentStatus.PAID

    def test_the_ledger_records_it(self, workspace, monkeypatch):
        db, _ = workspace
        opts = options(workspace, AutoLLM())
        record = held_record(workspace, opts)
        answers(monkeypatch, "approve")

        review_queue([record], opts)

        conn = connect(db)
        assert get_payment(conn, "INV-BIG") is not None
        conn.close()

    def test_the_reviewer_is_named_in_the_rationale(self, workspace, monkeypatch):
        """A payment released by a person should not be indistinguishable from one
        released automatically."""
        opts = options(workspace, AutoLLM())
        record = held_record(workspace, opts)
        answers(monkeypatch, "approve")

        assert "Approved by reviewer" in review_queue([record], opts)[0].rationale


class TestDenying:
    def test_it_records_a_rejection(self, workspace, monkeypatch):
        opts = options(workspace, AutoLLM())
        record = held_record(workspace, opts)
        answers(monkeypatch, "deny")

        resumed = review_queue([record], opts)

        assert resumed[0].outcome is Outcome.REJECTED
        assert resumed[0].payment_status is PaymentStatus.NOT_ATTEMPTED

    def test_nothing_is_paid(self, workspace, monkeypatch):
        db, _ = workspace
        opts = options(workspace, AutoLLM())
        record = held_record(workspace, opts)
        answers(monkeypatch, "deny")

        review_queue([record], opts)

        conn = connect(db)
        assert get_payment(conn, "INV-BIG") is None
        conn.close()


class TestSkipping:
    def test_the_run_stays_suspended(self, workspace, monkeypatch):
        """The checkpointer holds its state, so the same queue can be picked up on a
        later run. That is the point of a durable pause rather than an in-memory one.
        """
        opts = options(workspace, AutoLLM())
        record = held_record(workspace, opts)
        answers(monkeypatch, "skip")

        resumed = review_queue([record], opts)

        assert resumed[0].awaiting_review

    def test_a_skipped_invoice_can_be_decided_later(self, workspace, monkeypatch):
        opts = options(workspace, AutoLLM())
        record = held_record(workspace, opts)

        answers(monkeypatch, "skip")
        review_queue([record], opts)

        answers(monkeypatch, "approve")
        second = review_queue([record], opts)

        assert second[0].outcome is Outcome.APPROVED


class TestSeveralInvoices:
    def test_each_is_asked_about_separately(self, workspace, monkeypatch):
        opts = options(workspace, AutoLLM())
        first = held_record(workspace, opts, "invoice_1001.txt", "INV-BIG-1")
        second = held_record(workspace, opts, "invoice_1004.json", "INV-BIG-2")

        answers(monkeypatch, "approve", "deny")

        resumed = review_queue([first, second], opts)

        assert [r.outcome for r in resumed] == [Outcome.APPROVED, Outcome.REJECTED]

    def test_a_mixed_queue_pays_only_what_was_approved(self, workspace, monkeypatch):
        db, _ = workspace
        opts = options(workspace, AutoLLM())
        first = held_record(workspace, opts, "invoice_1001.txt", "INV-YES")
        second = held_record(workspace, opts, "invoice_1004.json", "INV-NO")

        answers(monkeypatch, "approve", "deny")
        review_queue([first, second], opts)

        conn = connect(db)
        assert get_payment(conn, "INV-YES") is not None
        assert get_payment(conn, "INV-NO") is None
        conn.close()
