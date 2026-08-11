"""Running a batch in the background, and letting browsers watch.

A batch takes a couple of minutes. The request that starts it returns immediately and the
work continues on a thread, which is the only way a browser gets to see anything happen
rather than staring at a pending request.

**Events are buffered, not queued.** Every batch keeps the full list of what it has
emitted. A client connecting a second after the batch started -- which is every client --
replays what it missed and then follows live. A queue would have dropped those first
events into a socket that did not exist yet, and the first documents would be invisible.

Buffering also means several browsers can watch the same batch, and a reconnection after a
dropped connection resumes rather than restarts.

In memory, deliberately. This is one process serving one person a demo; a durable event
log would be infrastructure with no reader. The *decisions* are persisted -- those matter.
"""

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from galatiq.cli.runner import RunOptions, prepare_checkpointer, process_document
from galatiq.loaders import discover
from galatiq.store.db import connect
from galatiq.store.repository import save_run

# How long a waiting client blocks before checking again. Short enough that a finished
# batch closes promptly, long enough not to spin.
_POLL_SECONDS = 0.25


@dataclass
class Batch:
    """One run of the pipeline over a set of documents."""

    id: str
    total: int
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    error: str | None = None

    _condition: threading.Condition = field(
        default_factory=threading.Condition, repr=False
    )

    def append(self, event: dict[str, Any]) -> None:
        """Record an event and wake anyone waiting.

        Called from worker threads, which is why it holds the lock -- eight documents
        run at once and they all report here.
        """
        with self._condition:
            self.events.append(event)
            self._condition.notify_all()

    def finish(self, error: str | None = None) -> None:
        with self._condition:
            self.done = True
            self.error = error
            self._condition.notify_all()

    def follow(self, start: int = 0) -> Iterator[dict[str, Any]]:
        """Replay from `start`, then follow live until the batch ends.

        Yields history first so a late connection sees the whole run, then blocks for
        new events rather than polling in a tight loop.
        """
        index = start

        while True:
            with self._condition:
                while index >= len(self.events) and not self.done:
                    self._condition.wait(_POLL_SECONDS)

                pending = self.events[index:]
                index = len(self.events)
                finished = self.done and not pending

            yield from pending

            if finished:
                return


class BatchManager:
    """Starts batches and keeps the recent ones around to be watched."""

    def __init__(self) -> None:
        self._batches: dict[str, Batch] = {}
        self._lock = threading.Lock()

    def get(self, batch_id: str) -> Batch | None:
        with self._lock:
            return self._batches.get(batch_id)

    def start(self, invoice_path: str, options: RunOptions) -> Batch:
        """Begin processing, and return immediately.

        The caller gets a batch id straight away so the browser can attach to the stream
        before any real work has happened.
        """
        paths = discover(invoice_path)
        batch = Batch(id=uuid.uuid4().hex[:12], total=len(paths))

        with self._lock:
            self._batches[batch.id] = batch

        thread = threading.Thread(
            target=self._run, args=(batch, paths, options), daemon=True
        )
        thread.start()

        return batch

    def _run(self, batch: Batch, paths: list[Path], options: RunOptions) -> None:
        """Process every document, reporting as it goes.

        Concurrency is the runner's, unchanged. This adds an event sink and a database
        write per document; it does not decide anything.
        """
        from concurrent.futures import ThreadPoolExecutor

        batch.append(
            {
                "type": "batch.start",
                "batch": batch.id,
                "total": len(paths),
                "documents": [str(p) for p in paths],
            }
        )

        try:
            options.on_event = batch.append

            # The pool below opens one checkpointer connection per worker, and an
            # sqlite file that does not exist yet is created by whichever connection
            # gets there first. Creating it here means the eight that follow open
            # something that is already a database.
            prepare_checkpointer(options)

            def one(path: Path) -> None:
                record = process_document(path, options)

                conn = options.connect_db() if options.connect_db else connect()
                try:
                    save_run(conn, batch.id, record)
                finally:
                    conn.close()

                batch.append(
                    {
                        "type": "document.end",
                        "doc": str(path),
                        "invoice": record.invoice_number,
                        "outcome": str(record.outcome) if record.outcome else None,
                        "risk": record.risk_score,
                        "awaiting_review": record.awaiting_review,
                        "vendor": record.vendor,
                        "usd_total": (
                            str(record.usd_total) if record.usd_total else None
                        ),
                        "latency_ms": record.latency_ms,
                    }
                )

            with ThreadPoolExecutor(max_workers=options.concurrency) as pool:
                list(pool.map(one, paths))

            batch.append({"type": "batch.end", "batch": batch.id})
            batch.finish()

        except Exception as exc:  # noqa: BLE001 - a failed batch must still close
            # The stream has to end, or every watching browser hangs. Reporting the
            # failure is better than a connection that quietly never closes.
            batch.append({"type": "batch.error", "message": f"{type(exc).__name__}: {exc}"})
            batch.finish(error=str(exc))


manager = BatchManager()
