"""Tests for the persistence layer.

Two guarantees under test, and I put both in the database rather than in calling
code:

  * re-seeding is safe and always lands on the same stock position
  * an invoice number can be paid at most once
"""

from decimal import Decimal

import pytest

from galatiq.store.db import connect, init_db
from galatiq.store.repository import (
    get_all_stock,
    get_catalog_price,
    get_payment,
    get_stock,
    is_paid,
    record_payment,
)
from galatiq.store.seed import seed_inventory

# The seed position every batch run starts from.
EXPECTED_STOCK = {"WidgetA": 15, "WidgetB": 10, "GadgetX": 5, "FakeItem": 0}


@pytest.fixture
def db(tmp_path):
    """A freshly created, seeded database in a temp directory.

    Never var/invoices.db — a test suite that writes to my actual ledger is a test
    suite I stop trusting.
    """
    conn = connect(tmp_path / "test.db")
    init_db(conn)
    seed_inventory(conn)
    yield conn
    conn.close()


class TestSchema:
    def test_init_is_idempotent(self, tmp_path):
        """Every statement in schema.sql uses IF NOT EXISTS, so re-running is safe."""
        conn = connect(tmp_path / "t.db")
        init_db(conn)
        init_db(conn)  # would raise "table already exists" without the guard
        conn.close()

    def test_stock_cannot_go_negative(self, db):
        """CHECK (stock >= 0) — the database refuses an oversell on its own.

        Unused until --reserve starts decrementing, but the constraint is what makes
        that decrement safe when it arrives.
        """
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE inventory SET stock = -1 WHERE item = 'WidgetA'")


class TestSeed:
    def test_seeds_the_assessment_stock_position(self, db):
        assert get_all_stock(db) == EXPECTED_STOCK

    def test_seeding_twice_does_not_raise(self, db):
        """The starter code's plain INSERT raises IntegrityError here.

        INSERT OR REPLACE is what makes the seed re-runnable, and re-runnable is what
        makes batch results reproducible.
        """
        seed_inventory(db)
        seed_inventory(db)

    def test_seeding_resets_modified_stock(self, db):
        """Reset semantics, not preserve semantics."""
        db.execute("UPDATE inventory SET stock = 3 WHERE item = 'WidgetA'")
        db.commit()
        assert get_stock(db, "WidgetA") == 3

        seed_inventory(db)
        assert get_stock(db, "WidgetA") == 15

    def test_seeding_does_not_clear_the_ledger(self, db):
        """Resetting stock and erasing payment history are different operations.

        Doing the second while asked for the first would silently destroy the
        idempotency record and re-enable double payment.
        """
        record_payment(
            db,
            invoice_number="INV-1001",
            amount=Decimal("5000.00"),
            currency="USD",
            provider="mock",
            model="mock",
        )
        seed_inventory(db)
        assert is_paid(db, "INV-1001")


class TestInventoryReads:
    def test_known_item_returns_stock(self, db):
        assert get_stock(db, "WidgetA") == 15

    def test_zero_stock_item_returns_zero(self, db):
        """FakeItem exists with stock 0 — invoice 1003 orders 100 of them."""
        assert get_stock(db, "FakeItem") == 0

    def test_unknown_item_returns_none(self, db):
        """None and 0 mean different things, and I keep them distinct.

        None is "this product does not exist" (invoice 1008's SuperGizmo, 1016's
        WidgetC); 0 is "we have none in stock". Different rejection reasons, and a
        human reviewer needs them told apart.
        """
        assert get_stock(db, "SuperGizmo") is None
        assert get_stock(db, "WidgetC") is None

    def test_catalog_prices(self, db):
        assert get_catalog_price(db, "WidgetA") == Decimal("250.00")
        assert get_catalog_price(db, "WidgetB") == Decimal("500.00")
        assert get_catalog_price(db, "GadgetX") == Decimal("750.00")

    def test_unpriced_and_unknown_items_return_none(self, db):
        assert get_catalog_price(db, "FakeItem") is None
        assert get_catalog_price(db, "SuperGizmo") is None

    def test_catalog_price_is_decimal_not_float(self, db):
        assert isinstance(get_catalog_price(db, "WidgetA"), Decimal)


class TestPaymentIdempotency:
    """The guarantee that stops the same invoice being paid twice."""

    def test_first_payment_is_recorded(self, db):
        assert record_payment(
            db,
            invoice_number="INV-1001",
            amount=Decimal("5000.00"),
            currency="USD",
            provider="xai",
            model="grok-4.5",
        )
        assert is_paid(db, "INV-1001")

    def test_second_payment_is_a_no_op(self, db):
        """A batch re-run must not pay again — and must not crash either.

        Attempting to re-pay is an ordinary event, since any re-run after a failure
        does it, so record_payment returns False rather than raising.
        """
        kwargs = dict(
            invoice_number="INV-1001",
            amount=Decimal("5000.00"),
            currency="USD",
            provider="mock",
            model="mock",
        )
        assert record_payment(db, **kwargs) is True
        assert record_payment(db, **kwargs) is False

        rows = db.execute(
            "SELECT COUNT(*) AS n FROM ledger WHERE invoice_number = 'INV-1001'"
        ).fetchone()
        assert rows["n"] == 1

    def test_revision_cannot_double_pay_the_original(self, db):
        """invoice_1004 and invoice_1004_revised share invoice number INV-1004.

        Different totals, and in a batch they look like two separate documents. The
        UNIQUE constraint is what stops the second one paying.
        """
        assert record_payment(
            db,
            invoice_number="INV-1004",
            amount=Decimal("3750.00"),
            currency="USD",
            provider="mock",
            model="mock",
        )
        assert (
            record_payment(
                db,
                invoice_number="INV-1004",
                amount=Decimal("4250.00"),
                currency="USD",
                provider="mock",
                model="mock",
                revision="R1",
            )
            is False
        )

        payment = get_payment(db, "INV-1004")
        assert payment["amount"] == Decimal("3750.00")  # the original, unchanged

    def test_real_constraint_violations_still_raise(self, db):
        """The IntegrityError handler must not swallow unrelated failures.

        A NOT NULL violation is a bug and has to surface; only the duplicate invoice
        number is the expected, handled case.
        """
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO ledger (invoice_number, amount_cents, currency,
                                    provider, model, paid_at)
                VALUES ('INV-9999', 100, 'USD', NULL, 'mock', '2026-01-01')
                """
            )


class TestPaymentRecords:
    def test_amount_round_trips_as_decimal(self, db):
        record_payment(
            db,
            invoice_number="INV-1012",
            amount=Decimal("9975.00"),
            currency="USD",
            provider="xai",
            model="grok-4.5",
        )
        payment = get_payment(db, "INV-1012")

        assert payment["amount"] == Decimal("9975.00")
        assert payment["amount_cents"] == 997500
        assert isinstance(payment["amount"], Decimal)

    def test_currency_is_stored_explicitly(self, db):
        """Invoice 1014 is in EUR. It must never read back as though it were USD."""
        record_payment(
            db,
            invoice_number="INV-1014",
            amount=Decimal("9000.00"),
            currency="EUR",
            provider="mock",
            model="mock",
        )
        assert get_payment(db, "INV-1014")["currency"] == "EUR"

    def test_provider_and_model_are_recorded(self, db):
        """What decided to pay this has to be a stored fact, not an inference."""
        record_payment(
            db,
            invoice_number="INV-1001",
            amount=Decimal("5000.00"),
            currency="USD",
            provider="xai",
            model="grok-4.5",
        )
        payment = get_payment(db, "INV-1001")

        assert payment["provider"] == "xai"
        assert payment["model"] == "grok-4.5"

    def test_unpaid_invoice_has_no_record(self, db):
        assert get_payment(db, "INV-9999") is None
        assert not is_paid(db, "INV-9999")
