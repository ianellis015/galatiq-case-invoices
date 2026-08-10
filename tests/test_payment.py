"""Tests for payment.

Two properties matter here and nothing else does much: an invoice is paid at most once,
and nothing that was not approved is paid at all.
"""

from decimal import Decimal

import pytest

from galatiq.models import (
    ApprovalDecision,
    Invoice,
    Outcome,
    PaymentStatus,
)
from galatiq.payment import execute_payment, mock_payment
from galatiq.store.db import connect, init_db
from galatiq.store.repository import get_payment
from galatiq.store.seed import seed_inventory


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "pay.db")
    init_db(conn)
    seed_inventory(conn)
    yield conn
    conn.close()


def approved(rationale="Clean invoice."):
    return ApprovalDecision(outcome=Outcome.APPROVED, rationale=rationale)


def invoice(number="INV-1001", total="5000.00", **kwargs):
    return Invoice(
        invoice_number=number,
        vendor=kwargs.pop("vendor", "Widgets Inc."),
        total=total,
        currency=kwargs.pop("currency", "USD"),
        **kwargs,
    )


def pay(conn, inv, decision):
    return execute_payment(conn, inv, decision, provider="fake", model="fake-1")


class TestPaying:
    def test_an_approved_invoice_is_paid(self, db):
        result = pay(db, invoice(), approved())

        assert result.status is PaymentStatus.PAID
        assert result.amount == Decimal("5000.00")

    def test_the_ledger_records_it(self, db):
        pay(db, invoice(), approved())
        row = get_payment(db, "INV-1001")

        assert row["amount"] == Decimal("5000.00")
        assert row["vendor"] == "Widgets Inc."

    def test_the_deciding_model_is_recorded(self, db):
        """"What decided to pay this" has to be a stored fact."""
        pay(db, invoice(), approved())
        row = get_payment(db, "INV-1001")

        assert row["provider"] == "fake"
        assert row["model"] == "fake-1"

    def test_currency_is_recorded_as_stated(self, db):
        pay(db, invoice("INV-1014", "4125.00", currency="EUR"), approved())

        assert get_payment(db, "INV-1014")["currency"] == "EUR"


class TestPaidAtMostOnce:
    def test_a_second_attempt_is_a_no_op(self, db):
        assert pay(db, invoice(), approved()).status is PaymentStatus.PAID
        assert pay(db, invoice(), approved()).status is PaymentStatus.ALREADY_PAID

    def test_1004_and_its_revision_produce_one_payment(self, db):
        """TP4. Same invoice number, different totals, and in a batch they look like
        two separate documents."""
        original = invoice("INV-1004", "1890.00", vendor="Precision Parts Ltd.")
        revised = invoice("INV-1004", "5940.00", vendor="Precision Parts Ltd.")
        revised = revised.model_copy(update={"revision": "R1"})

        assert pay(db, original, approved()).status is PaymentStatus.PAID
        assert pay(db, revised, approved()).status is PaymentStatus.ALREADY_PAID

        row = db.execute(
            "SELECT COUNT(*) AS n FROM ledger WHERE invoice_number = 'INV-1004'"
        ).fetchone()
        assert row["n"] == 1
        assert get_payment(db, "INV-1004")["amount"] == Decimal("1890.00")

    def test_the_payment_call_is_not_reached_on_a_duplicate(self, db, capsys):
        """Write-ahead: the ledger insert fails first, so `mock_payment` never runs.

        The reverse ordering — pay then record — double-pays after a crash between the
        two, quietly, and nobody notices until the vendor mentions it.
        """
        pay(db, invoice(), approved())
        capsys.readouterr()

        pay(db, invoice(), approved())

        assert capsys.readouterr().out == ""


class TestRefusals:
    @pytest.mark.parametrize(
        "outcome", [Outcome.REJECTED, Outcome.HELD_FOR_REVIEW]
    )
    def test_nothing_but_an_approval_is_paid(self, db, outcome):
        """The routing already ensures only approved invoices arrive here.

        A payment function that would release money for a rejected invoice if called
        wrongly is one bad refactor away from doing so.
        """
        decision = ApprovalDecision(outcome=outcome, rationale="No.")

        result = pay(db, invoice(), decision)

        assert result.status is PaymentStatus.NOT_ATTEMPTED
        assert get_payment(db, "INV-1001") is None

    def test_an_invoice_with_no_number_cannot_be_paid(self, db):
        """No number means no idempotency key, and no idempotency key means a retry
        pays again."""
        result = pay(db, invoice(number=None), approved())

        assert result.status is PaymentStatus.FAILED
        assert "idempotency key" in result.message

    def test_an_invoice_with_no_total_cannot_be_paid(self, db):
        result = pay(db, invoice(total=None), approved())

        assert result.status is PaymentStatus.FAILED


class TestMockPayment:
    def test_it_matches_the_brief(self, capsys):
        """Supplied with the assessment, kept as-is.

        Everything around it is built as though it were real, so swapping in a banking
        client is a one-file change rather than a redesign.
        """
        assert mock_payment("Widgets Inc.", Decimal("5000.00")) == {"status": "success"}
        assert "Paid 5000.00 to Widgets Inc." in capsys.readouterr().out
