"""Working through the invoices that need a person.

The queue is deliberately at the end rather than mid-run. A batch of twenty where three
need a signature should not stop on the third while somebody is at lunch -- so the run
completes, and the held ones are worked through afterwards with all the context still
attached.

Resuming costs nothing but attention. The graph picks up inside the hold node exactly
where it paused, with the invoice, the findings, the decision and its reasoning intact.
No model is called and nothing is recomputed.
"""

from rich.prompt import Prompt

from galatiq.cli import render
from galatiq.cli.runner import RunOptions, resume_document
from galatiq.models import RunRecord


def review_queue(held: list[RunRecord], options: RunOptions) -> list[RunRecord]:
    """Present each held invoice and take a decision. Returns the resumed records."""
    if not held:
        return []

    render.console.print()
    render.console.rule(f"[yellow]{len(held)} invoice(s) held for review[/yellow]")

    resumed: list[RunRecord] = []

    for index, record in enumerate(held, start=1):
        render.console.print()
        render.console.print(
            f"[dim]{index} of {len(held)}[/dim]  "
            f"[bold]{record.invoice_number or record.source_path}[/bold]  "
            f"{record.vendor or ''}"
        )

        if record.usd_total is not None:
            render.console.print(f"[dim]Amount:[/dim] ${record.usd_total:,.2f}")

        render.findings_table(record.findings)
        render.decision_panel(record)

        verdict = Prompt.ask(
            "  [bold]Approve this payment?[/bold]",
            choices=["approve", "deny", "skip"],
            default="skip",
        )

        if verdict == "skip":
            # Left suspended. The checkpointer holds its state, so the same queue can be
            # picked up on a later run -- which is the point of a durable pause rather
            # than an in-memory one.
            render.console.print("  [dim]Left for later.[/dim]")
            resumed.append(record)
            continue

        updated = resume_document(record.source_path, verdict, options)
        resumed.append(updated)

        render.console.print(
            f"  [green]Approved and paid.[/green]"
            if verdict == "approve"
            else "  [red]Denied. Rejection recorded.[/red]"
        )

    return resumed
