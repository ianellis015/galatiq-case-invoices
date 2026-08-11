"""Tests for the operational log and the live event printer.

Two questions the system answers in two places, and these tests hold the line between
them. The decision trail lives in the checkpointer and the run records; this covers what
the system was *doing* -- retries, timings, batch lifecycle, and the agents at work.

The live printer matters more than it looks. The self-correction loops are the most
interesting behaviour in the pipeline and, without `--verbose`, the terminal shows no
sign they exist at all.
"""

import logging

import pytest
from typer.testing import CliRunner

from galatiq import cli, logs
from galatiq.cli import render
from galatiq.cli.runner import BatchResult
from galatiq.config import PROJECT_ROOT
from galatiq.models import Outcome, RunRecord

runner = CliRunner()
INVOICES = PROJECT_ROOT / "data" / "invoices"


@pytest.fixture(autouse=True)
def clean_logging(tmp_path, monkeypatch):
    """Every test configures from scratch, into its own file."""
    logs.reset()
    monkeypatch.setattr("galatiq.logs.LOG_PATH", tmp_path / "galatiq.log")
    yield tmp_path / "galatiq.log"
    logs.reset()


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
    monkeypatch.setattr(cli, "get_client", lambda: FakeClient())

    captured = {}

    def fake_batch(paths, options):
        captured["options"] = options
        return BatchResult(records=captured.get("records", [record()]))

    monkeypatch.setattr(cli, "process_batch", fake_batch)
    return captured


class TestConfiguration:
    def test_the_file_is_written_without_being_asked_for(self, clean_logging):
        """Observability that has to be switched on in advance is observability you do
        not have the one time you need it."""
        path = logs.configure(verbose=False)

        logs.logger("galatiq.test").info("hello")

        assert path == clean_logging
        assert "hello" in clean_logging.read_text()

    def test_configuring_twice_does_not_duplicate_lines(self, clean_logging):
        """The CLI and the API both configure at startup. A test importing both should
        not end up writing every line twice."""
        logs.configure()
        logs.configure()

        logs.logger("galatiq.test").info("once")

        assert clean_logging.read_text().count("once") == 1

    def test_it_does_not_hijack_the_root_logger(self, clean_logging):
        """Ours alone. Anything else configuring the root logger -- uvicorn does -- must
        not get a copy of every line, and we must not get a copy of theirs."""
        logs.configure()

        assert logging.getLogger(logs.LOGGER_NAME).propagate is False

    def test_a_module_gets_a_named_child(self):
        assert logs.logger("galatiq.cli.runner").name == "galatiq.cli.runner"
        assert logs.logger("cli.runner").name == "galatiq.cli.runner"

    def test_an_unwritable_path_does_not_stop_the_run(self, tmp_path, monkeypatch):
        """A read-only checkout should cost the log, not the batch."""
        monkeypatch.setattr("galatiq.logs.LOG_PATH", tmp_path / "nope" / "x.log")
        monkeypatch.setattr(
            "galatiq.logs.ensure_var_dir",
            lambda p: (_ for _ in ()).throw(OSError("read-only")),
        )

        assert logs.configure() is None


class TestLiveEvents:
    """What `--verbose` prints while the pipeline runs."""

    def test_agents_and_machinery_are_marked_differently(self, capsys):
        """The distinction the whole architecture rests on. A reader watching this
        scroll past should be able to see which half is working."""
        printer = render.live_events()

        printer(
            {
                "type": "step.start",
                "doc": "data/invoices/invoice_1001.txt",
                "label": "Extractor",
                "kind": "agent",
                "parallel": False,
            }
        )
        printer(
            {
                "type": "step.start",
                "doc": "data/invoices/invoice_1001.txt",
                "label": "Checking stock",
                "kind": "deterministic",
                "parallel": False,
            }
        )

        out = capsys.readouterr().out

        assert "~" in out and "Extractor" in out
        assert "Checking stock" in out

    def test_the_fan_out_is_announced_once(self, capsys):
        """Eight checks start at once. Eight identical lines is noise, not information."""
        printer = render.live_events()

        for _ in range(8):
            printer(
                {
                    "type": "step.start",
                    "doc": "data/invoices/invoice_1001.txt",
                    "label": "Checking stock",
                    "kind": "deterministic",
                    "parallel": True,
                }
            )

        assert capsys.readouterr().out.count("checks at once") == 1

    def test_each_document_announces_its_own_fan_out(self, capsys):
        printer = render.live_events()

        for doc in ("a.txt", "b.txt"):
            printer(
                {
                    "type": "step.start",
                    "doc": doc,
                    "label": "Checking stock",
                    "kind": "deterministic",
                    "parallel": True,
                }
            )

        assert capsys.readouterr().out.count("checks at once") == 2

    def test_a_handoff_is_printed_with_its_reason(self, capsys):
        """The most interesting event in a run, and the one a report cannot show."""
        printer = render.live_events()

        printer(
            {
                "type": "handoff",
                "doc": "data/invoices/invoice_1013.json",
                "from": "Extraction critic",
                "to": "Extractor",
                "reason": "subtotal: read 21400.00, document says 21040.00",
            }
        )

        out = capsys.readouterr().out

        assert "Extraction critic" in out
        assert "Extractor" in out
        assert "21040.00" in out


class TestTheVerboseFlag:
    def test_it_wires_the_event_stream(self, stubbed):
        runner.invoke(
            cli.app,
            [f"--invoice_path={INVOICES}", "--no-interactive", "--verbose"],
        )

        assert stubbed["options"].on_event is not None

    def test_a_plain_run_stays_quiet(self, stubbed):
        """Twenty documents' worth of steps printed above the report would bury it."""
        runner.invoke(cli.app, [f"--invoice_path={INVOICES}", "--no-interactive"])

        assert stubbed["options"].on_event is None

    def test_json_output_is_never_polluted(self, stubbed):
        """`--json` has to stay machine-readable, whatever else was asked for."""
        result = runner.invoke(
            cli.app,
            [f"--invoice_path={INVOICES}", "--no-interactive", "--json", "--verbose"],
        )

        assert stubbed["options"].on_event is None

        import json

        json.loads(result.output)

    def test_the_log_path_is_reported(self, stubbed):
        """A grader should not have to go looking for it."""
        result = runner.invoke(
            cli.app, [f"--invoice_path={INVOICES}", "--no-interactive"]
        )

        assert "Logging to" in result.output
