"""Tests for the graph: routing, the two cycles, and end-to-end runs.

Routing functions are tested by calling them with plain dicts — no graph, no model, no
I/O. That is the point of putting the budgets there: a limit expressed as an integer
comparison can be proven exhaustively in microseconds, and nothing inside a document can
argue with it.

The end-to-end tests run real corpus files through a scripted `FakeLLM`, so the loader,
the nodes, the reducer and the routing all execute for real while the only simulated
part is the model.
"""

import tempfile
from decimal import Decimal
from pathlib import Path

from galatiq.agents import Critique, Discrepancy
from galatiq.agents.approval_critic import ApprovalCritique
from galatiq.store.db import connect, init_db
from galatiq.store.seed import seed_inventory
from galatiq.config import (
    MAX_CRITIC_ATTEMPTS,
    MAX_REFLECT_ATTEMPTS,
    MAX_SCHEMA_ATTEMPTS,
    PROJECT_ROOT,
)
from galatiq.graph import (
    after_approval_critic,
    after_critic,
    after_extract,
    build_graph,
    run_document,
    thread_for,
)
from galatiq.llm import LLMResponseError
from galatiq.models import (
    ApprovalDecision,
    FindingCode,
    Invoice,
    LineItem,
    Outcome,
    PaymentStatus,
    Severity,
)
from galatiq.state import initial_state

from conftest import FakeLLM

INVOICES = PROJECT_ROOT / "data" / "invoices"
ADVERSARIAL = PROJECT_ROOT / "data" / "adversarial"


def sound_critique():
    return Critique(verdict="PARSE_SOUND", reasoning="Transcription matches.")


def misparse(**kwargs):
    return Critique(verdict="MISPARSE_SUSPECTED", reasoning="Quantity misread.", **kwargs)


# A seeded database for the whole module. Built once at import rather than per test,
# because the checks only ever read it -- and never the developer's real one, which is
# the sort of thing you only get wrong once.
_DB = Path(tempfile.mkdtemp(prefix="galatiq-graph-")) / "invoices.db"
_setup = connect(_DB)
init_db(_setup)
seed_inventory(_setup)
_setup.close()


def run(client, path):
    """Invoke the graph without a checkpointer — faster, and durability has its own test."""
    graph = build_graph(client, connect_db=lambda: connect(_DB))
    return graph.invoke(initial_state(str(path)))


def codes(state):
    return [f.code for f in state["findings"]]


def extraction_findings(state):
    """Only what the extraction phase produced.

    The graph now runs the checks too, so a bare `findings == []` would assert that a
    real invoice has nothing wrong with it — a different and much stronger claim than
    these tests mean to make.
    """
    phase = {
        FindingCode.DOC_INCONSISTENT,
        FindingCode.HINT_DISAGREEMENT,
        FindingCode.NEEDS_HUMAN_REVIEW,
        FindingCode.EXTRACTION_UNCERTAIN,
    }
    return [f for f in state["findings"] if f.code in phase]


# ---------------------------------------------------------------------------
# Routing — plain dicts, no graph
# ---------------------------------------------------------------------------


class TestAfterExtract:
    def test_no_invoice_retries(self):
        assert after_extract({"invoice": None}) == "extract"

    def test_an_invoice_moves_to_the_audit(self, sample_invoice):
        assert after_extract({"invoice": sample_invoice}) == "extract_critic"


class TestAfterCritic:
    """Three verdicts, two routes. The collapsing is the design."""

    def test_sound_finishes(self, sound):
        assert after_critic({"critique": sound, "critic_attempts": 0}) == "finalize"

    def test_inconsistent_finishes(self, inconsistent):
        """The document is the problem, and reading it again cannot fix it.

        This is the branch that stops INV-1009 from consuming its whole budget.
        """
        assert (
            after_critic({"critique": inconsistent, "critic_attempts": 0}) == "finalize"
        )

    def test_misparse_retries_while_budget_remains(self):
        assert after_critic({"critique": misparse(), "critic_attempts": 0}) == "extract"

    def test_misparse_stops_at_the_budget(self):
        state = {"critique": misparse(), "critic_attempts": MAX_CRITIC_ATTEMPTS}
        assert after_critic(state) == "finalize"

    def test_no_critique_finishes(self):
        """The critic was skipped — unreadable document, or a failed extraction."""
        assert after_critic({"critique": None, "critic_attempts": 0}) == "finalize"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_a_clean_invoice(self, sample_invoice, sound):
        client = FakeLLM(sample_invoice, sound)

        state = run(client, INVOICES / "invoice_1001.txt")

        assert state["invoice"].invoice_number == "INV-1001"
        assert client.extraction_calls == 2  # one extraction, one audit
        assert extraction_findings(state) == []

    def test_the_document_is_actually_read_from_disk(self, sample_invoice, sound):
        client = FakeLLM(sample_invoice, sound)

        state = run(client, INVOICES / "invoice_1001.txt")

        assert "Widgets Inc." in state["raw_text"]

    def test_a_structured_document_gets_a_hint(self, sample_invoice, sound):
        client = FakeLLM(sample_invoice, sound)

        state = run(client, INVOICES / "invoice_1006.csv")

        assert state["structural_hint"]["invoice_number"] == "INV-1006"
        assert "INV-1006" in client.calls[0].prompt


class TestSchemaRetryLoop:
    def test_a_malformed_response_is_retried_with_the_error(self, sample_invoice, sound):
        client = FakeLLM(
            LLMResponseError("bad", detail="total: input should be a valid string"),
            sample_invoice,
            sound,
        )

        state = run(client, INVOICES / "invoice_1001.txt")

        assert state["invoice"].invoice_number == "INV-1001"
        assert state["schema_attempts"] == 1
        assert "total: input should be a valid string" in client.calls[1].prompt

    def test_the_budget_is_enforced(self):
        """Two attempts, then stop. Without the counter this loops forever, and each
        turn of it is an API call."""
        client = FakeLLM(
            *[LLMResponseError("bad", detail="broken") for _ in range(MAX_SCHEMA_ATTEMPTS)]
        )

        state = run(client, INVOICES / "invoice_1001.txt")

        assert client.extraction_calls == MAX_SCHEMA_ATTEMPTS
        assert state["schema_attempts"] == MAX_SCHEMA_ATTEMPTS

    def test_exhaustion_still_produces_an_invoice(self):
        """The document has to reach a decision. "We could not read this" is one a
        human can act on."""
        client = FakeLLM(
            *[LLMResponseError("bad", detail="broken") for _ in range(MAX_SCHEMA_ATTEMPTS)]
        )

        state = run(client, INVOICES / "invoice_1001.txt")

        assert isinstance(state["invoice"], Invoice)
        assert state["invoice"].invoice_number is None

    def test_exhaustion_is_reported(self):
        client = FakeLLM(
            *[LLMResponseError("bad", detail="broken") for _ in range(MAX_SCHEMA_ATTEMPTS)]
        )

        state = run(client, INVOICES / "invoice_1001.txt")
        codes = [f.code for f in state["findings"]]

        assert FindingCode.NEEDS_HUMAN_REVIEW in codes

    def test_the_critic_is_not_asked_to_audit_a_failed_extraction(self):
        """Auditing a structure the extractor never produced costs a call to confirm
        what is already known."""
        client = FakeLLM(
            *[LLMResponseError("bad", detail="broken") for _ in range(MAX_SCHEMA_ATTEMPTS)]
        )

        run(client, INVOICES / "invoice_1001.txt")

        assert client.extraction_calls == MAX_SCHEMA_ATTEMPTS


class TestCriticLoop:
    def test_a_suspected_misparse_re_extracts(self, sample_invoice, sound):
        client = FakeLLM(
            sample_invoice,
            misparse(
                discrepancies=[
                    Discrepancy(
                        field="line_items.0.quantity",
                        transcribed="5",
                        document_says="10",
                    )
                ]
            ),
            sample_invoice,
            sound,
        )

        state = run(client, INVOICES / "invoice_1001.txt")

        assert client.extraction_calls == 4
        assert state["critic_attempts"] == 1
        assert "line_items.0.quantity" in client.calls[2].prompt

    def test_the_budget_is_enforced(self, sample_invoice):
        """Each turn is two API calls, so an unbounded loop is expensive fast."""
        script = []
        for _ in range(MAX_CRITIC_ATTEMPTS + 1):
            script.extend([sample_invoice, misparse()])

        client = FakeLLM(*script)
        state = run(client, INVOICES / "invoice_1001.txt")

        assert state["critic_attempts"] == MAX_CRITIC_ATTEMPTS

    def test_persistent_suspicion_is_reported(self, sample_invoice):
        """Neither trusted nor discarded — flagged as uncertain for a human."""
        script = []
        for _ in range(MAX_CRITIC_ATTEMPTS + 1):
            script.extend([sample_invoice, misparse()])

        state = run(FakeLLM(*script), INVOICES / "invoice_1001.txt")
        codes = [f.code for f in state["findings"]]

        assert FindingCode.EXTRACTION_UNCERTAIN in codes

    def test_a_sound_verdict_costs_no_budget(self, sample_invoice, sound):
        client = FakeLLM(sample_invoice, sound)

        state = run(client, INVOICES / "invoice_1001.txt")

        assert state["critic_attempts"] == 0


class TestDocumentInconsistent:
    """INV-1009: the case the three-way verdict exists for."""

    def test_no_re_extraction_happens(self, sample_invoice, inconsistent):
        """The extraction is right and the document is wrong. Re-reading returns the
        same values, because they are the correct values."""
        client = FakeLLM(sample_invoice, inconsistent)

        state = run(client, INVOICES / "invoice_1009.json")

        assert client.extraction_calls == 2
        assert state["critic_attempts"] == 0

    def test_the_inconsistency_travels_downstream(self, sample_invoice, inconsistent):
        client = FakeLLM(sample_invoice, inconsistent)

        state = run(client, INVOICES / "invoice_1009.json")
        codes = [f.code for f in state["findings"]]

        assert FindingCode.DOC_INCONSISTENT in codes
        assert FindingCode.NEEDS_HUMAN_REVIEW not in codes


class TestUnreadableDocuments:
    def test_no_model_calls_at_all(self):
        client = FakeLLM()

        state = run(client, ADVERSARIAL / "invoice_A002.bin")

        assert client.extraction_calls == 0
        assert isinstance(state["invoice"], Invoice)

    def test_load_findings_survive_to_the_end(self):
        """The loader already explained why the document is unreadable. That
        explanation is what reaches the reviewer."""
        state = run(FakeLLM(), ADVERSARIAL / "invoice_A002.bin")
        codes = [f.code for f in state["findings"]]

        assert FindingCode.UNREADABLE_DOCUMENT in codes
        assert any(f.severity == Severity.CRITICAL for f in state["findings"])

    def test_an_unknown_format_still_reaches_the_model(self, sample_invoice, sound):
        """A .yaml invoice has no parser and does not need one."""
        client = FakeLLM(sample_invoice, sound)

        state = run(client, ADVERSARIAL / "invoice_A001.yaml")

        assert client.extraction_calls == 2
        assert FindingCode.UNSUPPORTED_FORMAT in [f.code for f in state["findings"]]


class TestHintCrossCheck:
    def test_a_dropped_line_item_is_caught(self, sound):
        """The reason the structural parsers still exist.

        INV-1006 has two line items. A model that returns one produces a plausible,
        self-consistent, wrong invoice — and the parser is the only thing that would
        notice.
        """
        one_line = Invoice(
            invoice_number="INV-1006",
            line_items=[LineItem(raw_name="WidgetB", quantity=3, unit_price="500.00")],
            subtotal="2750.00",
            total="2750.00",
        )
        client = FakeLLM(one_line, sound)

        state = run(client, INVOICES / "invoice_1006.csv")
        codes = [f.code for f in state["findings"]]

        assert FindingCode.HINT_DISAGREEMENT in codes

    def test_agreement_is_silent(self, sound):
        agreeing = Invoice(
            invoice_number="INV-1006",
            line_items=[
                LineItem(raw_name="WidgetA", quantity=5, unit_price="250.00"),
                LineItem(raw_name="WidgetB", quantity=3, unit_price="500.00"),
            ],
            subtotal="2750.00",
            total="2750.00",
        )

        state = run(FakeLLM(agreeing, sound), INVOICES / "invoice_1006.csv")

        assert extraction_findings(state) == []


class TestOcrDamage:
    """INV-1012, end to end. The case that broke the first live run."""

    def test_an_unreadable_amount_does_not_crash_the_graph(self, sound):
        """The model transcribes "$3,500.O0" faithfully — which is what we asked for.

        Before the fix, decimal.InvalidOperation escaped through pydantic and took the
        node down: an exception where a finding belongs.
        """
        damaged = Invoice(invoice_number="INV-1012", total_raw="$3,500.O0")
        client = FakeLLM(damaged, sound)

        state = run(client, INVOICES / "invoice_1012.txt")

        assert state["invoice"] is not None

    def test_the_damage_is_reported_with_the_original_text(self, sound):
        """Not silently repaired. A retry that rewrote it as 3500.00 would produce a
        payment nobody could trace back to what the document actually said."""
        damaged = Invoice(invoice_number="INV-1012", total_raw="$3,500.O0")

        state = run(FakeLLM(damaged, sound), INVOICES / "invoice_1012.txt")
        unreadable = [
            f
            for f in state["findings"]
            if f.code == FindingCode.DATA_INTEGRITY and "$3,500.O0" in f.evidence
        ]

        assert len(unreadable) == 1
        assert state["invoice"].total is None
        assert state["invoice"].total_raw == "$3,500.O0"

    def test_a_percentage_tax_rate_is_understood(self, sound):
        """"Tax (6%)" transcribes as "6%", which the money parser rejected — four
        invoices failed on this in the first live run."""
        invoice = Invoice(invoice_number="INV-1007", tax_rate_raw="6%")

        state = run(FakeLLM(invoice, sound), INVOICES / "invoice_1007.csv")

        assert state["invoice"].tax_rate == Decimal("0.06")
        assert not [
            f
            for f in state["findings"]
            if f.code == FindingCode.DATA_INTEGRITY and "6%" in f.evidence
        ]


class TestFindingsAreNotDuplicated:
    def test_a_retried_loop_reports_once(self, sound):
        """`findings` concatenates, so a node running three times contributes three
        copies. An invoice whose audit trail says the same thing three times reads as
        three problems, which is why findings are emitted in `finalize`.
        """
        one_line = Invoice(
            invoice_number="INV-1006",
            line_items=[LineItem(raw_name="WidgetB", quantity=3, unit_price="500.00")],
        )
        client = FakeLLM(one_line, misparse(), one_line, sound)

        state = run(client, INVOICES / "invoice_1006.csv")
        disagreements = [
            f for f in state["findings"] if f.code == FindingCode.HINT_DISAGREEMENT
        ]

        assert len(disagreements) == 1


class TestValidationFanOut:
    """The seven checks running concurrently, end to end."""

    def test_findings_from_multiple_checks_merge_without_conflict(self, sound):
        """Seven nodes writing the same state key at once.

        Without `Annotated[list[Finding], operator.add]` this raises InvalidUpdateError.
        That one annotation is the difference between a fan-out and a crash.
        """
        broken = Invoice(
            invoice_number="INV-1009",
            vendor="",
            line_items=[LineItem(raw_name="WidgetA", quantity=-5, unit_price="250.00")],
            subtotal="1000.00",
            total="-250.00",
            currency="USD",
        )

        state = run(FakeLLM(broken, sound), INVOICES / "invoice_1009.json")
        found = set(codes(state))

        assert FindingCode.DATA_INTEGRITY in found
        assert FindingCode.MATH_MISMATCH in found

    def test_the_context_snapshot_lands_in_state(self, sample_invoice, sound):
        """So the audit trail answers "what stock did we see when we rejected this?"
        rather than "what does stock say now"."""
        state = run(FakeLLM(sample_invoice, sound), INVOICES / "invoice_1001.txt")

        assert state["check_context"]["stock"]["WidgetA"] == 15
        assert "today" in state["check_context"]

    def test_normalisation_happens_before_the_checks(self, sound):
        """The stock check aggregates by canonical name, so it cannot run until the
        names are canonical."""
        invoice = Invoice(
            invoice_number="INV-1010",
            vendor="Acme",
            total="3000.00",
            line_items=[
                LineItem(raw_name="WidgetA (rush order)", quantity=10, unit_price="250.00"),
                LineItem(raw_name="Widget A", quantity=2, unit_price="250.00"),
            ],
        )

        state = run(FakeLLM(invoice, sound), INVOICES / "invoice_1010.txt")

        assert [li.item for li in state["invoice"].line_items] == ["WidgetA", "WidgetA"]

    def test_aggregate_stock_breach_is_caught(self, sound):
        """INV-1013 end to end: each line passes, the totals do not."""
        invoice = Invoice(
            invoice_number="INV-1013",
            vendor="Atlas Industrial Supply",
            total="22562.80",
            currency="USD",
            line_items=[
                LineItem(raw_name=n, quantity=q, unit_price=p)
                for n, q, p in [
                    ("WidgetA", 15, "250.00"), ("WidgetA", 5, "240.00"),
                    ("WidgetA", 2, "250.00"),
                ]
            ],
        )

        state = run(FakeLLM(invoice, sound), INVOICES / "invoice_1013.json")
        breaches = [f for f in state["findings"] if f.code == FindingCode.STOCK_EXCEEDED]

        assert len(breaches) == 1
        assert "22 requested across 3 lines" in breaches[0].message

    def test_merged_findings_are_deduplicated_and_sorted(self, sound):
        broken = Invoice(invoice_number="INV-X", vendor="", line_items=[])

        state = run(FakeLLM(broken, sound), INVOICES / "invoice_1001.txt")
        merged = state["merged_findings"]

        assert merged
        assert [f.severity for f in merged] == sorted(
            (f.severity for f in merged),
            key=lambda s: {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.INFO: 2}[s],
        )

    def test_the_raw_findings_survive_alongside_the_merged_ones(self, sound):
        """A checkpoint should record what each check actually said, not just the
        tidied summary."""
        broken = Invoice(invoice_number="INV-X", vendor="", line_items=[])

        state = run(FakeLLM(broken, sound), INVOICES / "invoice_1001.txt")

        assert len(state["findings"]) >= len(state["merged_findings"])

    def test_an_unreadable_document_still_reaches_the_checks(self):
        """No model calls, and the integrity check still reports what is missing."""
        client = FakeLLM()

        state = run(client, ADVERSARIAL / "invoice_A002.bin")

        assert client.extraction_calls == 0
        assert FindingCode.DATA_INTEGRITY in codes(state)


class TestAfterApprovalCritic:
    """Routing for the second loop and the three terminals — plain dicts, no graph."""

    def _decision(self, outcome):
        return ApprovalDecision(outcome=outcome, rationale="x")

    def test_approved_pays(self):
        state = {"decision": self._decision(Outcome.APPROVED), "reflect_attempts": 0}
        assert after_approval_critic(state) == "pay"

    def test_rejected_rejects(self):
        state = {"decision": self._decision(Outcome.REJECTED), "reflect_attempts": 0}
        assert after_approval_critic(state) == "reject"

    def test_held_holds(self):
        state = {
            "decision": self._decision(Outcome.HELD_FOR_REVIEW),
            "reflect_attempts": 0,
        }
        assert after_approval_critic(state) == "hold"

    def test_a_missed_signal_reconsiders(self):
        state = {
            "decision": self._decision(Outcome.APPROVED),
            "approval_critique": ApprovalCritique(
                verdict="MISSED_SIGNALS", reasoning="x", missed=["y"]
            ),
            "reflect_attempts": 0,
        }
        assert after_approval_critic(state) == "approve"

    def test_the_budget_stops_reconsidering(self):
        state = {
            "decision": self._decision(Outcome.APPROVED),
            "approval_critique": ApprovalCritique(
                verdict="MISSED_SIGNALS", reasoning="x", missed=["y"]
            ),
            "reflect_attempts": MAX_REFLECT_ATTEMPTS,
        }
        assert after_approval_critic(state) == "pay"

    def test_no_decision_holds(self):
        """Whatever went wrong, a human should look at it."""
        assert after_approval_critic({"decision": None}) == "hold"


class TestApprovalAndPayment:
    """End to end, through to a terminal."""

    def _run(self, invoice, client=None, findings_client=None):
        return run(client or FakeLLM(invoice, sound_critique()), INVOICES / "invoice_1001.txt")

    def test_a_clean_small_invoice_is_paid(self):
        invoice = Invoice(
            invoice_number="INV-PAY-1", vendor="Widgets Inc.",
            total="2500.00", currency="USD",
            line_items=[LineItem(raw_name="WidgetA", quantity=10, unit_price="250.00")],
            due_date_raw="2099-01-01",
        )

        state = self._run(invoice)

        assert state["decision"].outcome == Outcome.APPROVED
        assert state["payment"].status == PaymentStatus.PAID

    def test_a_large_clean_invoice_is_held(self):
        """Correctness decides approve-vs-reject; size decides automatic-vs-human."""
        invoice = Invoice(
            invoice_number="INV-PAY-2", vendor="Widgets Inc.",
            total="50000.00", currency="USD",
            line_items=[LineItem(raw_name="WidgetA", quantity=10, unit_price="250.00")],
            due_date_raw="2099-01-01",
        )

        state = self._run(invoice)

        assert state["decision"].outcome == Outcome.HELD_FOR_REVIEW
        assert state["payment"].status == PaymentStatus.NOT_ATTEMPTED
        assert "Over $10,000" in state["decision"].rationale

    def test_a_broken_invoice_is_rejected_with_reasoning(self):
        invoice = Invoice(
            invoice_number="INV-PAY-3", vendor="Acme",
            total="5000.00", currency="USD",
            line_items=[LineItem(raw_name="WidgetA", quantity=999, unit_price="250.00")],
            due_date_raw="2099-01-01",
        )

        state = self._run(invoice)

        assert state["decision"].outcome == Outcome.REJECTED
        assert state["payment"].status == PaymentStatus.NOT_ATTEMPTED
        assert state["decision"].rationale
        assert state["decision"].policy_refs

    def test_an_unknown_item_is_rejected(self):
        """INV-1016's WidgetC. The normalizer refuses to guess, so it stays unknown."""
        invoice = Invoice(
            invoice_number="INV-PAY-4", vendor="Widgets Inc.",
            total="1050.00", currency="USD",
            line_items=[LineItem(raw_name="WidgetC", quantity=3, unit_price="350.00")],
            due_date_raw="2099-01-01",
        )

        state = self._run(invoice)

        assert state["decision"].outcome == Outcome.REJECTED
        assert FindingCode.UNKNOWN_ITEM in codes(state)

    def test_the_same_invoice_twice_pays_once(self):
        """TP4, through the whole graph."""
        invoice = Invoice(
            invoice_number="INV-PAY-DUP", vendor="Widgets Inc.",
            total="1000.00", currency="USD",
            line_items=[LineItem(raw_name="WidgetA", quantity=4, unit_price="250.00")],
            due_date_raw="2099-01-01",
        )

        first = self._run(invoice)
        second = self._run(invoice)

        assert first["payment"].status == PaymentStatus.PAID
        # The duplicate check sees the ledger row on the second pass and rejects before
        # payment is reached, which is the reason rather than the mechanism -- the
        # UNIQUE constraint would have stopped it regardless.
        assert second["payment"].status == PaymentStatus.NOT_ATTEMPTED
        assert second["decision"].outcome == Outcome.REJECTED

    def test_every_terminal_records_a_payment_result(self):
        """No path reaches END without saying what happened to the money."""
        for total, expected in [
            ("2500.00", PaymentStatus.PAID),
            ("50000.00", PaymentStatus.NOT_ATTEMPTED),
        ]:
            invoice = Invoice(
                invoice_number=f"INV-TERM-{total}", vendor="Widgets Inc.",
                total=total, currency="USD",
                line_items=[
                    LineItem(raw_name="WidgetA", quantity=10, unit_price="250.00")
                ],
                due_date_raw="2099-01-01",
            )
            assert self._run(invoice)["payment"].status == expected


class TestCheckpointing:
    def test_state_is_durable_and_replayable(self, tmp_path, sample_invoice, sound):
        """The audit trail, for the cost of one argument.

        Snapshots after every node mean "what did the system know before it rejected
        this" is a question with an exact answer.
        """
        client = FakeLLM(sample_invoice, sound)
        path = str(INVOICES / "invoice_1001.txt")

        state = run_document(client, path, audit_db=tmp_path / "audit.db")

        assert state["invoice"].invoice_number == "INV-1001"
        assert (tmp_path / "audit.db").exists()

    def test_each_document_gets_its_own_timeline(self):
        """The twins are genuinely different documents and should not share a history."""
        pdf = thread_for("data/invoices/invoice_1011.pdf")
        txt = thread_for("data/invoices/invoice_1011.txt")

        assert pdf["configurable"]["thread_id"] != txt["configurable"]["thread_id"]
