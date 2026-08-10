"""The command line.

    python main.py --invoice_path=data/invoices/invoice_1001.txt
    python main.py --invoice_path=data/invoices/
    python main.py --invoice_path='data/invoices/*.csv' --as-of 2026-02-01

The flag is spelled `--invoice_path` with an underscore because that is what the brief
specifies, and a documented command that does not work is worse than no documentation.
`--invoice-path` is accepted too, since a hyphen is what anyone would try first.
"""

import json
import time
from datetime import date
from pathlib import Path
from typing import Optional

import typer

from galatiq.cli import render
from galatiq.cli.review import review_queue
from galatiq.cli.runner import RunOptions, process_batch
from galatiq.config import DEFAULT_CONCURRENCY
from galatiq.llm import LLMConfigError, get_client
from galatiq.loaders import discover
from galatiq.models import Outcome

app = typer.Typer(
    add_completion=False,
    help="Process invoices: ingest, validate, approve, pay.",
)


@app.command()
def run(
    invoice_path: str = typer.Option(
        ...,
        "--invoice_path",
        "--invoice-path",
        help="A file, a directory, or a glob.",
    ),
    as_of: Optional[str] = typer.Option(
        None,
        "--as-of",
        help=(
            "Reference date for due-date checks (YYYY-MM-DD). "
            "The sample corpus is dated January 2026."
        ),
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Review held invoices at the end of the run.",
    ),
    concurrency: int = typer.Option(
        DEFAULT_CONCURRENCY, "--concurrency", "-j", help="Documents processed at once."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit run records as JSON instead of a report."
    ),
) -> None:
    """Process one invoice, or a directory of them."""
    reference = date.fromisoformat(as_of) if as_of else date.today()

    try:
        client = get_client()
    except LLMConfigError as exc:
        render.console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)

    try:
        paths = discover(invoice_path)
    except FileNotFoundError as exc:
        render.console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)

    if not paths:
        render.console.print(f"[yellow]No documents found at {invoice_path}[/yellow]")
        raise typer.Exit(code=1)

    options = RunOptions(
        client=client,
        as_of=reference,
        interactive=interactive,
        concurrency=concurrency,
    )

    if not as_json:
        render.banner(client.provider, client.model, reference)
        render.console.print()

    started = time.monotonic()
    result = process_batch(paths, options)
    elapsed = time.monotonic() - started

    if as_json:
        typer.echo(
            json.dumps(
                [json.loads(r.model_dump_json()) for r in result.records], indent=2
            )
        )
        raise typer.Exit(code=_exit_code(result))

    _report(result, paths, elapsed)

    if interactive and result.held:
        review_queue(result.held, options)

    raise typer.Exit(code=_exit_code(result))


def _report(result, paths: list[Path], elapsed: float) -> None:
    """Full detail for one document, one line each for many."""
    single = len(paths) == 1

    if single:
        record = result.records[0]
        if record.error:
            render.console.print(f"[red]{record.error}[/red]")
            return

        # The invoice is re-read from the record rather than kept in memory, so what is
        # shown is exactly what was recorded -- the same thing `--json` would emit.
        render.findings_table(record.findings)
        render.console.print()
        render.decision_panel(record)
        return

    for record in result.records:
        render.batch_row(record)

    render.console.print()
    render.summary(result, elapsed)


def _exit_code(result) -> int:
    """0 clean, 1 if anything needs attention, 2 for a failure to run.

    A batch containing rejections is a batch that worked -- catching bad invoices is the
    job. But a non-zero code is what makes this usable from a script, so anything other
    than "all approved" says so.
    """
    if any(r.error for r in result.records):
        return 2
    if any(r.outcome is not Outcome.APPROVED for r in result.records):
        return 1
    return 0


__all__ = ["app", "run"]
