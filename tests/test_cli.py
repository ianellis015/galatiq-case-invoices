"""Tests for the command-line layer.

Thin on purpose. The runner and the review queue have their own tests; what is left here
is the wiring — argument parsing, error paths, output shape, and exit codes.

Exit codes matter more than they look. They are how this gets used from a script, and a
batch that silently exits 0 with fourteen rejections is a batch nobody notices.
"""

import json
from datetime import date

import pytest
from typer.testing import CliRunner

from galatiq import cli
from galatiq.cli.runner import BatchResult
from galatiq.config import PROJECT_ROOT
from galatiq.models import Outcome, PaymentStatus, RunRecord

runner = CliRunner()
INVOICES = PROJECT_ROOT / "data" / "invoices"


class FakeClient:
    provider = "fake"
    model = "fake-model-1"


def record(name="invoice_1001.txt", outcome=Outcome.APPROVED, **kwargs):
    return RunRecord(
        source_path=f"data/invoices/{name}",
        invoice_number=kwargs.pop("invoice_number", "INV-1001"),
        outcome=outcome,
        usd_total=kwargs.pop("usd_total", "5000.00"),
        **kwargs,
    )


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the model client and the batch, leaving the CLI wiring under test."""
    monkeypatch.setattr(cli, "get_client", lambda: FakeClient())

    captured = {}

    def fake_batch(paths, options):
        captured["paths"] = paths
        captured["options"] = options
        return BatchResult(records=captured.get("records", [record()]))

    monkeypatch.setattr(cli, "process_batch", fake_batch)
    return captured


class TestTheDocumentedCommand:
    def test_the_brief_s_flag_spelling_works(self, stubbed):
        """`--invoice_path`, with an underscore, because that is what the brief says.

        A documented command that does not work is worse than no documentation.
        """
        result = runner.invoke(
            cli.app,
            [f"--invoice_path={INVOICES / 'invoice_1001.txt'}", "--no-interactive"],
        )

        assert result.exit_code == 0

    def test_the_hyphenated_spelling_works_too(self, stubbed):
        """Nobody's fingers type an underscore first."""
        result = runner.invoke(
            cli.app,
            [f"--invoice-path={INVOICES / 'invoice_1001.txt'}", "--no-interactive"],
        )

        assert result.exit_code == 0

    def test_a_directory(self, stubbed):
        result = runner.invoke(
            cli.app, [f"--invoice_path={INVOICES}", "--no-interactive"]
        )

        assert result.exit_code in (0, 1)
        assert len(stubbed["paths"]) == 20

    def test_a_glob(self, stubbed):
        result = runner.invoke(
            cli.app, [f"--invoice_path={INVOICES}/*.csv", "--no-interactive"]
        )

        assert result.exit_code in (0, 1)
        assert len(stubbed["paths"]) == 3


class TestErrorPaths:
    def test_a_missing_path_exits_cleanly(self, stubbed):
        result = runner.invoke(
            cli.app, ["--invoice_path=/nowhere/at/all", "--no-interactive"]
        )

        assert result.exit_code == 2
        assert "no such file" in result.output.lower()

    def test_a_bad_date_is_not_a_traceback(self, stubbed):
        result = runner.invoke(
            cli.app,
            [
                f"--invoice_path={INVOICES / 'invoice_1001.txt'}",
                "--as-of=not-a-date",
                "--no-interactive",
            ],
        )

        assert result.exit_code != 0

    def test_a_missing_key_says_what_to_set(self, monkeypatch):
        from galatiq.llm import LLMConfigError

        def no_key():
            raise LLMConfigError("No xAI API key. Set XAI_API_KEY in .env.")

        monkeypatch.setattr(cli, "get_client", no_key)

        result = runner.invoke(
            cli.app,
            [f"--invoice_path={INVOICES / 'invoice_1001.txt'}", "--no-interactive"],
        )

        assert result.exit_code == 2
        assert "XAI_API_KEY" in result.output


class TestAsOf:
    def test_it_reaches_the_run_options(self, stubbed):
        runner.invoke(
            cli.app,
            [
                f"--invoice_path={INVOICES / 'invoice_1001.txt'}",
                "--as-of=2026-02-01",
                "--no-interactive",
            ],
        )

        assert stubbed["options"].as_of == date(2026, 2, 1)

    def test_it_defaults_to_today(self, stubbed):
        runner.invoke(
            cli.app,
            [f"--invoice_path={INVOICES / 'invoice_1001.txt'}", "--no-interactive"],
        )

        assert stubbed["options"].as_of == date.today()


class TestJsonOutput:
    def test_it_is_valid_json(self, stubbed):
        result = runner.invoke(
            cli.app,
            [
                f"--invoice_path={INVOICES / 'invoice_1001.txt'}",
                "--json",
                "--no-interactive",
            ],
        )
        payload = json.loads(result.output)

        assert payload[0]["invoice_number"] == "INV-1001"
        assert payload[0]["outcome"] == "APPROVED"

    def test_it_suppresses_the_banner(self, stubbed):
        """Machine-readable means machine-readable. A banner on stdout makes the output
        unparseable for the caller that asked for JSON."""
        result = runner.invoke(
            cli.app,
            [
                f"--invoice_path={INVOICES / 'invoice_1001.txt'}",
                "--json",
                "--no-interactive",
            ],
        )

        assert "galatiq" not in result.output.split("\n")[0]


class TestExitCodes:
    """How this gets used from a script."""

    def test_all_approved_is_zero(self, stubbed):
        stubbed["records"] = [record()]

        result = runner.invoke(
            cli.app,
            [f"--invoice_path={INVOICES / 'invoice_1001.txt'}", "--no-interactive"],
        )

        assert result.exit_code == 0

    def test_anything_needing_attention_is_one(self, stubbed):
        """A batch with rejections worked -- catching bad invoices is the job -- but a
        script should still be able to tell."""
        stubbed["records"] = [record(outcome=Outcome.REJECTED)]

        result = runner.invoke(
            cli.app,
            [f"--invoice_path={INVOICES / 'invoice_1001.txt'}", "--no-interactive"],
        )

        assert result.exit_code == 1

    def test_a_failure_to_run_is_two(self, stubbed):
        stubbed["records"] = [record(outcome=None, error="boom")]

        result = runner.invoke(
            cli.app,
            [f"--invoice_path={INVOICES / 'invoice_1001.txt'}", "--no-interactive"],
        )

        assert result.exit_code == 2


class TestBatchReport:
    def test_the_summary_counts_documents_and_unique_invoices(self, stubbed):
        """INV-1011 arrives twice. "20 invoices" would be wrong and "16" would hide
        work that was done."""
        stubbed["records"] = [
            record("invoice_1011.pdf", invoice_number="INV-1011"),
            record("invoice_1011.txt", invoice_number="INV-1011"),
        ]

        result = runner.invoke(
            cli.app, [f"--invoice_path={INVOICES}", "--no-interactive"]
        )

        assert "Documents" in result.output
        assert "Unique invoices" in result.output

    def test_an_approval_that_paid_nothing_says_so(self, stubbed):
        """Under concurrency two documents for one invoice can both be approved. A row
        reading plain APPROVED for the second would be a report lying about a payment.
        """
        stubbed["records"] = [
            record("invoice_1004.json", payment_status=PaymentStatus.PAID),
            record(
                "invoice_1004_revised.json", payment_status=PaymentStatus.ALREADY_PAID
            ),
        ]

        result = runner.invoke(
            cli.app, [f"--invoice_path={INVOICES}", "--no-interactive"]
        )

        assert "already paid" in result.output
        assert "Duplicate payments avoided" in result.output

    def test_a_rejection_with_no_findings_still_states_a_reason(self, stubbed):
        """INV-1004 passes every deterministic check and is refused on the approver's
        judgement about a fictional vendor address. The row used to print blank — a
        rejection with no stated reason, in a tool whose claim is that every decision
        is explicable."""
        stubbed["records"] = [
            record(
                "invoice_1004.json",
                outcome=Outcome.REJECTED,
                invoice_number="INV-1004",
                findings=[],
                concerns=["Vendor address is a famous fictional address"],
            )
        ]

        result = runner.invoke(
            cli.app, [f"--invoice_path={INVOICES}", "--no-interactive"]
        )

        # Truncated to the terminal width, so the assertion is on the opening of the
        # concern rather than the whole of it.
        assert "Vendor address" in result.output


class TestPreventedTotal:
    """The headline business number, and the one worth being conservative about."""

    def test_one_invoice_arriving_twice_is_counted_once(self, stubbed):
        """INV-1013 is rejected as JSON and as a PDF, but only one payment could ever
        have happened — the ledger's UNIQUE constraint guarantees it. Counting both
        inflated the figure by $22,562.80 on the provided corpus."""
        stubbed["records"] = [
            record(
                "invoice_1013.json",
                outcome=Outcome.REJECTED,
                invoice_number="INV-1013",
                usd_total="22562.80",
            ),
            record(
                "invoice_1013.pdf",
                outcome=Outcome.REJECTED,
                invoice_number="INV-1013",
                usd_total="22562.80",
            ),
        ]

        result = runner.invoke(
            cli.app, [f"--invoice_path={INVOICES}", "--no-interactive"]
        )

        assert "$22,562.80" in result.output
        assert "$45,125.60" not in result.output

    def test_a_negative_total_does_not_subtract(self, stubbed):
        """INV-1009 states a total of -$250. Preventing it did not save minus two
        hundred and fifty dollars."""
        stubbed["records"] = [
            record(
                "invoice_1002.txt",
                outcome=Outcome.REJECTED,
                invoice_number="INV-1002",
                usd_total="15000.00",
            ),
            record(
                "invoice_1009.json",
                outcome=Outcome.REJECTED,
                invoice_number="INV-1009",
                usd_total="-250.00",
            ),
        ]

        result = runner.invoke(
            cli.app, [f"--invoice_path={INVOICES}", "--no-interactive"]
        )

        assert "$15,000.00" in result.output
        assert "$14,750.00" not in result.output
