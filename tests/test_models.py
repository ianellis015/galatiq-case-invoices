"""Tests for the data shapes.

Two things I care about here. First, that the float guard from `money.py` survives
into the models — pydantic would otherwise coerce a float to Decimal silently.
Second, and more important, that the models can hold the *broken* invoices in the
corpus. Anything strict enough to reject INV-1009 turns a reasoned rejection into a
stack trace, so most of this file is deliberately feeding the models bad documents
and asserting they accept them.
"""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from galatiq.models import (
    ApprovalDecision,
    Finding,
    FindingCode,
    Invoice,
    LineItem,
    Outcome,
    PaymentResult,
    PaymentStatus,
    Severity,
)


class TestMoneyFields:
    """Decimal or nothing, on every money field."""

    def test_accepts_decimal(self):
        item = LineItem(raw_name="WidgetA", quantity=1, unit_price=Decimal("250.00"))
        assert item.unit_price == Decimal("250.00")

    def test_accepts_string_forms(self):
        """Documents state amounts as text, and it arrives with the formatting on."""
        item = LineItem(raw_name="WidgetA", quantity=1, unit_price="$9,975.00")
        assert item.unit_price == Decimal("9975.00")

    def test_rejects_float_on_line_item(self):
        """The guard the whole money design rests on.

        Pydantic accepts a float for a Decimal field and converts it without
        complaint, which reintroduces the representation error money.py exists to
        prevent.
        """
        with pytest.raises(ValidationError):
            LineItem(raw_name="WidgetA", quantity=1, unit_price=250.00)

    @pytest.mark.parametrize(
        "field", ["subtotal", "tax_amount", "total", "tax_rate"]
    )
    def test_rejects_float_on_every_invoice_money_field(self, field):
        with pytest.raises(ValidationError):
            Invoice(invoice_number="INV-1001", **{field: 250.00})

    def test_optional_money_treats_absence_as_absence(self):
        """Absent is not zero. INV-1003 states no subtotal at all."""
        invoice = Invoice(invoice_number="INV-1003", subtotal=None, tax_amount="")
        assert invoice.subtotal is None
        assert invoice.tax_amount is None

    def test_precision_is_not_lost(self):
        invoice = Invoice(invoice_number="INV-1013", total="22562.80")
        assert invoice.total == Decimal("22562.80")


class TestLineItem:
    def test_raw_name_is_preserved_alongside_canonical(self):
        """INV-1010 says "WidgetA (rush order)"; the catalog says "WidgetA".

        Both have to survive: the canonical name drives the stock check, the raw name
        is what a human reads when tracing the decision.
        """
        item = LineItem(
            raw_name="WidgetA (rush order)",
            item="WidgetA",
            quantity=2,
            unit_price="250.00",
        )
        assert item.raw_name == "WidgetA (rush order)"
        assert item.item == "WidgetA"

    def test_canonical_name_defaults_to_none(self):
        """Unresolved until the normalizer runs — and None is what becomes an
        UNKNOWN_ITEM finding when it never resolves."""
        item = LineItem(raw_name="SuperGizmo", quantity=1, unit_price="100.00")
        assert item.item is None

    def test_negative_quantity_is_allowed(self):
        """INV-1009 has quantity -5. Rejecting it here means never reporting it."""
        item = LineItem(raw_name="WidgetA", quantity=-5, unit_price="250.00")
        assert item.quantity == -5

    def test_stated_amount_is_kept_as_stated(self):
        """INV-1013 states a per-line amount as well as a unit price.

        I store it rather than recomputing, so check_math has both numbers to
        compare rather than one number and an assumption.
        """
        item = LineItem(
            raw_name="WidgetA",
            quantity=5,
            unit_price="240.00",
            stated_amount="1200.00",
            note="Volume discount",
        )
        assert item.stated_amount == Decimal("1200.00")
        assert item.note == "Volume discount"


class TestBrokenInvoicesConstruct:
    """The point of the ticket: bad documents have to be representable."""

    def test_invoice_1009_shape(self):
        """Empty vendor, null due date, negative quantity, and a stated subtotal of
        1000.00 against line items summing to -250.00.

        Every one of those would be rejected by a schema written defensively. All of
        them have to survive, because each becomes a finding downstream.
        """
        invoice = Invoice(
            invoice_number="INV-1009",
            vendor="",
            issue_date_raw="2026-01-15",
            due_date_raw=None,
            line_items=[
                LineItem(raw_name="WidgetA", quantity=-5, unit_price="250.00"),
                LineItem(raw_name="WidgetB", quantity=2, unit_price="500.00"),
            ],
            subtotal="1000.00",
            tax_rate="0.0",
            tax_amount="0.00",
            total="-250.00",
            currency="USD",
            payment_terms="",
        )

        assert invoice.vendor == ""
        assert invoice.due_date is None
        assert invoice.line_items[0].quantity == -5

        # The model stores the contradiction rather than resolving it. Computed
        # -250.00, stated 1000.00 -- check_math's problem, not the schema's.
        computed = sum(li.quantity * li.unit_price for li in invoice.line_items)
        assert computed == Decimal("-250.00")
        assert invoice.subtotal == Decimal("1000.00")

    def test_invoice_1013_shape(self):
        """Eight lines, repeated items, and a $50 gap at the total.

        Subtotal and tax both reconcile exactly; only the final total is off. The
        model has to hold that rather than "correcting" it.
        """
        lines = [
            ("WidgetA", 15, "250.00", "3750.00"),
            ("WidgetB", 10, "500.00", "5000.00"),
            ("GadgetX", 5, "750.00", "3750.00"),
            ("WidgetA", 5, "240.00", "1200.00"),
            ("WidgetB", 8, "480.00", "3840.00"),
            ("GadgetX", 3, "750.00", "2250.00"),
            ("WidgetA", 2, "250.00", "500.00"),
            ("GadgetX", 1, "750.00", "750.00"),
        ]
        invoice = Invoice(
            invoice_number="INV-1013",
            vendor="Atlas Industrial Supply",
            line_items=[
                LineItem(raw_name=n, quantity=q, unit_price=p, stated_amount=a)
                for n, q, p, a in lines
            ],
            subtotal="21040.00",
            tax_rate="0.07",
            tax_amount="1472.80",
            total="22562.80",
            currency="USD",
        )

        assert len(invoice.line_items) == 8
        assert invoice.subtotal + invoice.tax_amount == Decimal("22512.80")
        assert invoice.total - (invoice.subtotal + invoice.tax_amount) == Decimal("50.00")

    def test_invoice_1003_shape(self):
        """No subtotal, no tax, an unparseable due date, and injection text.

        The notes field is load-bearing rather than incidental: the fraud check
        cannot score language the model never stored.
        """
        invoice = Invoice(
            invoice_number="INV-1003",
            vendor="Fraudster LLC",
            issue_date_raw="2026-01-20",
            due_date_raw="yesterday",
            line_items=[
                LineItem(raw_name="FakeItem", quantity=100, unit_price="1,000.00")
            ],
            total="$100,000.00",
            payment_terms="Immediate",
            notes="URGENT - Pay immediately to avoid penalties!!! Wire transfer preferred.",
        )

        assert invoice.subtotal is None
        assert invoice.tax_amount is None
        assert invoice.due_date_raw == "yesterday"
        assert invoice.due_date is None
        assert "Wire transfer" in invoice.notes
        assert invoice.total == Decimal("100000.00")

    def test_invoice_with_no_line_items(self):
        """INV-1008 is email prose with no table at all."""
        invoice = Invoice(invoice_number="INV-1008", vendor="Office Supplies Co")
        assert invoice.line_items == []


class TestInvoiceDefaults:
    def test_currency_has_no_default(self):
        """Assuming USD when a document is silent is a judgement about the world.

        INV-1014 is in EUR, and a silent USD default is exactly how that would slip
        past the $10,000 threshold rule unnoticed.
        """
        assert Invoice(invoice_number="INV-1001").currency is None

    def test_dates_are_not_auto_parsed(self):
        """The model has no hidden behaviour — filling the parsed field is the
        extractor's job, and doing it here would bury a failure mode in a schema."""
        invoice = Invoice(invoice_number="INV-1001", due_date_raw="2026-02-15")
        assert invoice.due_date is None

    def test_parsed_dates_are_accepted_when_supplied(self):
        invoice = Invoice(
            invoice_number="INV-1001",
            due_date_raw="2026-02-15",
            due_date=date(2026, 2, 15),
        )
        assert invoice.due_date == date(2026, 2, 15)

    def test_provenance_fields(self):
        """INV-1011/1012/1013 each exist in two formats whose contents differ, so
        "which invoice" is not a specific enough answer for an audit trail."""
        invoice = Invoice(
            invoice_number="INV-1013",
            source_path="data/invoices/invoice_1013.pdf",
            source_format="pdf",
        )
        assert invoice.source_format == "pdf"

    def test_unknown_field_is_rejected(self):
        """extra="forbid" — an unexpected key from the LLM is drift, and drift is
        what the extraction retry loop exists to catch."""
        with pytest.raises(ValidationError):
            Invoice(invoice_number="INV-1001", grand_total="500.00")


class TestFinding:
    def test_construction(self):
        finding = Finding(
            code=FindingCode.STOCK_EXCEEDED,
            severity=Severity.CRITICAL,
            message="GadgetX: 9 requested, 5 in stock",
            evidence="lines 3, 6, 8 aggregate to 9",
        )
        assert finding.code == "STOCK_EXCEEDED"
        assert finding.severity == Severity.CRITICAL

    def test_code_must_be_known(self):
        """A free-string code would let a typo produce a finding the policy engine
        silently never matches — a failure with no symptom."""
        with pytest.raises(ValidationError):
            Finding(
                code="STOKC_EXCEEDED",
                severity=Severity.CRITICAL,
                message="typo",
            )

    def test_severity_must_be_known(self):
        with pytest.raises(ValidationError):
            Finding(code=FindingCode.UNKNOWN_ITEM, severity="BAD", message="x")


class TestApprovalDecision:
    def test_construction(self):
        decision = ApprovalDecision(
            outcome=Outcome.REJECTED,
            rationale="Aggregate quantities exceed stock on all three items.",
            policy_refs=["R2", "R3"],
            risk_score=90,
        )
        assert decision.outcome == "REJECTED"
        assert "R3" in decision.policy_refs

    def test_held_for_review_is_a_real_outcome(self):
        decision = ApprovalDecision(
            outcome=Outcome.HELD_FOR_REVIEW, rationale="Needs VP sign-off."
        )
        assert decision.outcome == Outcome.HELD_FOR_REVIEW

    @pytest.mark.parametrize("score", [-1, 101])
    def test_risk_score_is_bounded(self, score):
        """A score outside 0-100 is a malformed response, not a strong opinion."""
        with pytest.raises(ValidationError):
            ApprovalDecision(outcome=Outcome.APPROVED, rationale="x", risk_score=score)

    def test_defaults(self):
        decision = ApprovalDecision(outcome=Outcome.APPROVED, rationale="Clean.")
        assert decision.policy_refs == []
        assert decision.risk_score == 0


class TestPaymentResult:
    def test_paid(self):
        result = PaymentResult(
            invoice_number="INV-1001",
            status=PaymentStatus.PAID,
            amount="5000.00",
            currency="USD",
            vendor="Widgets Inc.",
            provider="xai",
            model="grok-4.5",
            paid_at="2026-08-08T12:00:00+00:00",
        )
        assert result.status == PaymentStatus.PAID
        assert result.amount == Decimal("5000.00")

    def test_already_paid_is_a_success_not_a_failure(self):
        """The idempotent no-op that stops INV-1004_revised paying twice."""
        result = PaymentResult(
            invoice_number="INV-1004",
            status=PaymentStatus.ALREADY_PAID,
            message="Ledger already holds a payment for this invoice number.",
        )
        assert result.status == PaymentStatus.ALREADY_PAID

    def test_not_attempted_is_the_normal_rejected_case(self):
        result = PaymentResult(
            invoice_number="INV-1003", status=PaymentStatus.NOT_ATTEMPTED
        )
        assert result.amount is None

    def test_rejects_float_amount(self):
        with pytest.raises(ValidationError):
            PaymentResult(
                invoice_number="INV-1001", status=PaymentStatus.PAID, amount=5000.00
            )


class TestSerialization:
    """The checkpointer serializes state to SQLite, so a round trip has to be exact."""

    def test_invoice_round_trip_preserves_precision(self):
        original = Invoice(
            invoice_number="INV-1013",
            vendor="Atlas Industrial Supply",
            line_items=[
                LineItem(raw_name="WidgetA", quantity=15, unit_price="250.00"),
            ],
            subtotal="21040.00",
            tax_amount="1472.80",
            total="22562.80",
            currency="USD",
        )

        restored = Invoice.model_validate_json(original.model_dump_json())

        assert restored == original
        assert restored.total == Decimal("22562.80")
        assert isinstance(restored.total, Decimal)

    def test_negative_amounts_round_trip(self):
        original = Invoice(invoice_number="INV-1009", total="-250.00")
        restored = Invoice.model_validate_json(original.model_dump_json())
        assert restored.total == Decimal("-250.00")

    def test_finding_round_trip(self):
        original = Finding(
            code=FindingCode.MATH_MISMATCH,
            severity=Severity.CRITICAL,
            message="Stated total exceeds subtotal + tax by 50.00",
            evidence="21040.00 + 1472.80 = 22512.80, stated 22562.80",
        )
        restored = Finding.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_decision_round_trip(self):
        original = ApprovalDecision(
            outcome=Outcome.HELD_FOR_REVIEW,
            rationale="Near-threshold amount with an unverified vendor.",
            policy_refs=["R1", "R9"],
            risk_score=55,
        )
        restored = ApprovalDecision.model_validate_json(original.model_dump_json())
        assert restored == original


class TestArbitraryInvoices:
    """Presence is what varies across invoices nobody anticipated.

    The models were permissive about values and strict about presence, which is our
    own principle applied inconsistently: for the corpus, presence is reliable; for an
    arbitrary document it is exactly the thing that is missing. A required field turns
    the least parseable documents -- the ones a human most needs told about -- into
    the ones that crash.
    """

    def test_invoice_without_a_number_constructs(self):
        """A scanned image, a truncated file, a document in an unfamiliar layout.

        This has to reach a decision as a DATA_INTEGRITY rejection naming what was
        missing, not as an exception.
        """
        invoice = Invoice(vendor="Unknown Vendor Ltd.", total="500.00")

        assert invoice.invoice_number is None
        assert invoice.total == Decimal("500.00")

    def test_line_item_without_quantity_or_price_constructs(self):
        item = LineItem(raw_name="WidgetA")

        assert item.quantity is None
        assert item.unit_price is None

    def test_unparseable_quantity_is_preserved_raw(self):
        """"a dozen" is not an int, and discarding it loses the evidence a
        DATA_INTEGRITY finding would quote."""
        item = LineItem(raw_name="WidgetA", quantity=None, quantity_raw="a dozen")

        assert item.quantity is None
        assert item.quantity_raw == "a dozen"

    def test_line_stating_only_an_extended_amount(self):
        """No unit price to give. check_math reconciles whichever numbers exist."""
        item = LineItem(raw_name="Consulting", stated_amount="1200.00")

        assert item.unit_price is None
        assert item.stated_amount == Decimal("1200.00")

    def test_unmapped_source_fields_are_preserved(self):
        """A PO number, a shipping line, a department code -- none map onto a
        modelled field. Dropping them silently means a human reading the audit trail
        cannot see what the system saw and ignored.
        """
        invoice = Invoice(
            invoice_number="INV-9001",
            extra={"po_number": "PO-20260115", "department": "Facilities"},
        )

        assert invoice.extra["po_number"] == "PO-20260115"

    def test_extra_defaults_to_empty(self):
        assert Invoice(invoice_number="INV-1001").extra == {}

    def test_an_almost_empty_invoice_still_constructs(self):
        """The floor: whatever was readable, nothing else.

        Every field below is missing, and none of that is an error -- it is a set of
        findings waiting to be raised.
        """
        invoice = Invoice(source_path="data/adversarial/invoice_A002.bin")

        assert invoice.invoice_number is None
        assert invoice.vendor == ""
        assert invoice.line_items == []
        assert invoice.total is None

    def test_ingestion_finding_codes_exist(self):
        """Load-time problems report through the same channel as validation ones."""
        for code in (
            FindingCode.UNREADABLE_DOCUMENT,
            FindingCode.UNSUPPORTED_FORMAT,
            FindingCode.HINT_DISAGREEMENT,
            FindingCode.EXTRACTION_UNCERTAIN,
        ):
            assert Finding(
                code=code, severity=Severity.WARN, message="x"
            ).code == code
