"""Tests for the HTTP layer.

Against a fake client, like everything else — no key, no network.

The point being defended here is that the API is *additive*. It reaches the same
decisions the CLI reaches, because it calls the same runner; it adds a way to watch and
a record of what happened, and it changes nothing about how anything is decided.
"""

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from galatiq import api
from galatiq.api.batches import Batch, BatchManager
from galatiq.api.events import PARALLEL_CHECKS, STEPS, DocumentTracker, describe
from galatiq.cli.runner import RunOptions
from galatiq.config import PROJECT_ROOT
from galatiq.models import Invoice, LineItem, Outcome

from conftest import AutoLLM

INVOICES = PROJECT_ROOT / "data" / "invoices"
client = TestClient(api.app)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the API at a temporary database and a fake model."""
    from galatiq.store.db import connect, init_db
    from galatiq.store.seed import seed_inventory

    db = tmp_path / "invoices.db"
    conn = connect(db)
    init_db(conn)
    seed_inventory(conn)
    conn.close()

    monkeypatch.setattr("galatiq.config.DB_PATH", db)
    monkeypatch.setattr("galatiq.store.db.DB_PATH", db)
    # Resetting deletes the checkpointer's database, so this one has to point at the
    # temp directory too, or the tests take the real one with them.
    monkeypatch.setattr(api, "AUDIT_DB_PATH", tmp_path / "audit.db")

    def options(as_of=None, concurrency=4):
        return RunOptions(
            client=AutoLLM(),
            connect_db=lambda: connect(db),
            audit_db=tmp_path / "audit.db",
            as_of=date(2026, 2, 15),
            interactive=True,
            concurrency=concurrency,
        )

    monkeypatch.setattr(api, "_options", options)
    return db


class TestStepDescriptions:
    """Every node needs a name a person can read."""

    def test_agents_are_marked_as_agents(self):
        for node in ("extract", "extract_critic", "normalize", "approve", "approval_critic"):
            assert describe(node).kind == "agent"

    def test_machinery_is_not(self):
        """The distinction the whole architecture rests on. A UI that rendered the
        eight checks the same way as the extractor would misrepresent the system."""
        for node in ("load", "check_stock", "merge_findings", "pay"):
            assert describe(node).kind == "deterministic"

    def test_labels_are_readable(self):
        assert describe("extract_critic").label == "Extraction critic"
        assert describe("check_stock").label == "Checking stock"

    def test_an_unknown_node_still_gets_a_label(self):
        """A node added later should not render as a blank."""
        assert describe("some_future_node").label == "Some future node"

    def test_the_parallel_checks_are_identified(self):
        """Tied to the registry rather than to a number, so adding a check cannot leave
        the interface describing a fan-out it no longer draws."""
        from galatiq.checks import all_checks

        assert PARALLEL_CHECKS == frozenset(all_checks())
        assert all(name.startswith("check_") for name in PARALLEL_CHECKS)


class TestHandoffDetection:
    """A backwards edge is one agent sending work to another."""

    def test_a_critic_sending_work_back_is_a_handoff(self):
        tracker = DocumentTracker(doc="x.txt")
        tracker.on_task_result("extract_critic", {})

        events = tracker.on_task("extract")

        assert events[0]["type"] == "handoff"
        assert events[0]["from"] == "Extraction critic"
        assert events[0]["to"] == "Extractor"

    def test_forward_progress_is_not_a_handoff(self):
        tracker = DocumentTracker(doc="x.txt")
        tracker.on_task_result("extract", {})

        assert all(e["type"] != "handoff" for e in tracker.on_task("extract_critic"))

    def test_the_reason_quotes_the_critic_when_it_gave_one(self):
        """"An agent sent it back" is a fact. "The critic thought line 3's quantity was
        misread" is the thing worth watching."""
        from galatiq.agents import Critique, Discrepancy

        tracker = DocumentTracker(doc="x.txt")
        tracker.on_task_result(
            "extract_critic",
            {
                "critique": Critique(
                    verdict="MISPARSE_SUSPECTED",
                    reasoning="Quantity misread.",
                    discrepancies=[
                        Discrepancy(
                            field="line_items.0.quantity",
                            transcribed="5",
                            document_says="15",
                        )
                    ],
                )
            },
        )

        reason = tracker.on_task("extract")[0]["reason"]

        assert "line_items.0.quantity" in reason
        assert "15" in reason

    def test_it_falls_back_to_a_plain_description(self):
        tracker = DocumentTracker(doc="x.txt")
        tracker.on_task_result("approval_critic", {})

        assert tracker.on_task("approve")[0]["reason"]


class TestEventBuffer:
    """A client that connects late must not miss the beginning."""

    def test_a_late_subscriber_sees_everything(self):
        batch = Batch(id="b1", total=2)
        batch.append({"type": "batch.start"})
        batch.append({"type": "step.start", "step": "extract"})
        batch.finish()

        assert len(list(batch.follow())) == 2

    def test_following_stops_when_the_batch_ends(self):
        batch = Batch(id="b2", total=0)
        batch.finish()

        assert list(batch.follow()) == []

    def test_several_watchers_each_see_the_whole_run(self):
        batch = Batch(id="b3", total=1)
        batch.append({"type": "one"})
        batch.append({"type": "two"})
        batch.finish()

        assert len(list(batch.follow())) == len(list(batch.follow())) == 2


class TestHealth:
    def test_it_reports_the_model(self, monkeypatch):
        """A frontend rendering a "Run batch" button against a service with no
        credentials is a button that fails on click for no visible reason."""
        monkeypatch.setattr(api, "get_client", lambda: AutoLLM())

        body = client.get("/api/health").json()

        assert body["ok"] is True
        assert body["model"] == "fake-model-1"

    def test_a_missing_key_is_reported_rather_than_raised(self, monkeypatch):
        from galatiq.llm import LLMConfigError

        def no_key():
            raise LLMConfigError("Set XAI_API_KEY.")

        monkeypatch.setattr(api, "get_client", no_key)
        body = client.get("/api/health").json()

        assert body["ok"] is False
        assert "XAI_API_KEY" in body["detail"]


class TestRunningABatch:
    def test_it_returns_immediately_with_an_id(self, workspace):
        """The work takes minutes. A request that blocks for minutes is one the browser
        gives up on."""
        response = client.post(
            "/api/runs", json={"invoice_path": str(INVOICES / "invoice_1001.txt")}
        )

        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert response.json()["batch_id"]

    def test_a_missing_path_is_a_404(self, workspace):
        response = client.post("/api/runs", json={"invoice_path": "/nowhere"})
        assert response.status_code == 404

    def test_the_stream_reports_the_run(self, workspace):
        started = client.post(
            "/api/runs", json={"invoice_path": str(INVOICES / "invoice_1001.txt")}
        ).json()

        with client.stream("GET", f"/api/stream/{started['batch_id']}") as response:
            events = [
                json.loads(line[6:])
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

        types = {e["type"] for e in events}

        assert "batch.start" in types
        assert "step.start" in types
        assert "document.end" in types
        assert "batch.end" in types

    def test_agents_are_distinguishable_in_the_stream(self, workspace):
        started = client.post(
            "/api/runs", json={"invoice_path": str(INVOICES / "invoice_1001.txt")}
        ).json()

        with client.stream("GET", f"/api/stream/{started['batch_id']}") as response:
            events = [
                json.loads(line[6:])
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

        kinds = {e["kind"] for e in events if e["type"] == "step.start"}
        assert kinds == {"agent", "deterministic"}

    def test_an_unknown_batch_is_a_404(self):
        assert client.get("/api/stream/nonexistent").status_code == 404


class TestReadingResults:
    def _run(self, workspace, path=INVOICES / "invoice_1001.txt"):
        started = client.post("/api/runs", json={"invoice_path": str(path)}).json()
        with client.stream("GET", f"/api/stream/{started['batch_id']}") as response:
            list(response.iter_lines())
        return started["batch_id"]

    def test_records_persist_after_the_batch(self, workspace):
        """A browser refresh must not lose the run."""
        self._run(workspace)

        records = client.get("/api/runs").json()

        assert len(records) == 1
        assert records[0]["outcome"]
        assert "id" in records[0]

    def test_one_record_carries_its_findings(self, workspace):
        self._run(workspace)
        run_id = client.get("/api/runs").json()[0]["id"]

        record = client.get(f"/api/runs/{run_id}").json()

        assert "findings" in record
        assert "rationale" in record

    def test_an_unknown_run_is_a_404(self, workspace):
        assert client.get("/api/runs/999999").status_code == 404

    def test_the_summary_matches_the_records(self, workspace):
        self._run(workspace)

        summary = client.get("/api/summary").json()

        assert summary["documents"] == 1
        assert summary["approved"] + summary["held"] + summary["rejected"] == 1


class TestReview:
    def _held(self, workspace):
        """Run a large clean invoice, which the rules hold for a human."""
        big = Invoice(
            invoice_number="INV-BIG",
            vendor="Widgets Inc.",
            total="50000.00",
            currency="USD",
            due_date_raw="2026-03-01",
            line_items=[LineItem(raw_name="WidgetA", quantity=10, unit_price="250.00")],
        )
        original = api._options

        def options(as_of=None, concurrency=4):
            opts = original(as_of, concurrency)
            opts.client = AutoLLM(invoice=big)
            return opts

        api._options = options
        try:
            started = client.post(
                "/api/runs", json={"invoice_path": str(INVOICES / "invoice_1001.txt")}
            ).json()
            with client.stream("GET", f"/api/stream/{started['batch_id']}") as response:
                list(response.iter_lines())
            return client.get("/api/runs").json()[0]
        finally:
            api._options = original

    def test_a_held_invoice_is_marked_as_awaiting_review(self, workspace):
        record = self._held(workspace)

        assert record["awaiting_review"] is True
        assert record["outcome"] == Outcome.HELD_FOR_REVIEW

    def test_approving_pays_it(self, workspace):
        record = self._held(workspace)

        updated = client.post(
            f"/api/runs/{record['id']}/review", json={"verdict": "approve"}
        ).json()

        assert updated["outcome"] == Outcome.APPROVED
        assert updated["payment_status"] == "PAID"

    def test_denying_rejects_it(self, workspace):
        record = self._held(workspace)

        updated = client.post(
            f"/api/runs/{record['id']}/review", json={"verdict": "deny"}
        ).json()

        assert updated["outcome"] == Outcome.REJECTED

    def test_the_decision_survives_a_refresh(self, workspace):
        """The whole reason records are persisted."""
        record = self._held(workspace)
        client.post(f"/api/runs/{record['id']}/review", json={"verdict": "approve"})

        assert client.get(f"/api/runs/{record['id']}").json()["outcome"] == (
            Outcome.APPROVED
        )

    def test_a_nonsense_verdict_is_rejected(self, workspace):
        record = self._held(workspace)

        response = client.post(
            f"/api/runs/{record['id']}/review", json={"verdict": "maybe"}
        )

        assert response.status_code == 400

    def test_reviewing_something_not_held_is_a_conflict(self, workspace):
        started = client.post(
            "/api/runs", json={"invoice_path": str(INVOICES / "invoice_1001.txt")}
        ).json()
        with client.stream("GET", f"/api/stream/{started['batch_id']}") as response:
            list(response.iter_lines())
        record = client.get("/api/runs").json()[0]

        response = client.post(
            f"/api/runs/{record['id']}/review", json={"verdict": "approve"}
        )

        assert response.status_code == 409


class TestDocuments:
    """The grid needs its tiles before anything has run."""

    def test_it_discovers_rather_than_lists(self, workspace, tmp_path):
        """Nothing about any particular set of files is baked in. Point it elsewhere
        and that is what appears."""
        folder = tmp_path / "elsewhere"
        folder.mkdir()
        (folder / "invoice_x.txt").write_text("INVOICE")
        (folder / "invoice_y.csv").write_text("field,value")

        body = client.get(f"/api/documents?path={folder}").json()

        assert {d["name"] for d in body["documents"]} == {
            "invoice_x.txt",
            "invoice_y.csv",
        }

    def test_it_reports_format_and_size(self, workspace, tmp_path):
        folder = tmp_path / "one"
        folder.mkdir()
        (folder / "invoice_a.json").write_text('{"invoice_number": "INV-1"}')

        doc = client.get(f"/api/documents?path={folder}").json()["documents"][0]

        assert doc["format"] == "json"
        assert doc["bytes"] > 0

    def test_an_empty_folder_is_not_an_error(self, workspace, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()

        body = client.get(f"/api/documents?path={folder}").json()

        assert body["documents"] == []

    def test_a_missing_folder_is_a_404(self, workspace):
        assert client.get("/api/documents?path=/nowhere/at/all").status_code == 404


class TestReset:
    def test_it_clears_payments_and_runs(self, workspace):
        started = client.post(
            "/api/runs",
            json={"invoice_path": str(INVOICES / "invoice_1001.txt"), "reset": False},
        ).json()
        with client.stream("GET", f"/api/stream/{started['batch_id']}") as response:
            list(response.iter_lines())

        assert client.get("/api/runs").json()

        client.post("/api/reset")

        assert client.get("/api/runs").json() == []

    def test_inventory_returns_to_its_seeded_position(self, workspace):
        from galatiq.store.db import connect
        from galatiq.store.repository import get_all_stock

        conn = connect(workspace)
        conn.execute("UPDATE inventory SET stock = 0")
        conn.commit()
        conn.close()

        client.post("/api/reset")

        conn = connect(workspace)
        assert get_all_stock(conn)["WidgetA"] == 15
        conn.close()

    def test_running_resets_by_default(self, workspace):
        """Otherwise the second demo run rejects everything as a duplicate of the
        first -- correct, and a terrible first impression."""
        first = client.post(
            "/api/runs", json={"invoice_path": str(INVOICES / "invoice_1001.txt")}
        ).json()
        with client.stream("GET", f"/api/stream/{first['batch_id']}") as response:
            list(response.iter_lines())

        second = client.post(
            "/api/runs", json={"invoice_path": str(INVOICES / "invoice_1001.txt")}
        ).json()
        with client.stream("GET", f"/api/stream/{second['batch_id']}") as response:
            list(response.iter_lines())

        records = client.get("/api/runs").json()

        assert len(records) == 1
        assert records[0]["outcome"] == Outcome.APPROVED

    def test_it_clears_the_checkpointer_too(self, workspace, tmp_path):
        """Threads are keyed by source path, so a checkpoint that outlives a reset is a
        run the same document resumes into instead of starting over."""
        audit = tmp_path / "audit.db"
        audit.write_bytes(b"")

        client.post("/api/reset")

        assert not audit.exists()


class TestStartup:
    """State lasts as long as the process, and not a moment longer."""

    def test_booting_clears_previous_decisions(self, workspace):
        started = client.post(
            "/api/runs",
            json={"invoice_path": str(INVOICES / "invoice_1001.txt"), "reset": False},
        ).json()
        with client.stream("GET", f"/api/stream/{started['batch_id']}") as response:
            list(response.iter_lines())

        assert client.get("/api/runs").json()

        # Entering the client is what runs the lifespan -- the same thing uvicorn does
        # when it boots.
        with TestClient(api.app):
            pass

        assert client.get("/api/runs").json() == []
