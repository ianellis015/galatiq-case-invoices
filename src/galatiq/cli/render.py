"""Putting a decision on screen.

Two audiences, and they want opposite things. Someone running a single invoice wants
everything: what was read, what was found, what was decided and why. Someone running
twenty wants one line each and a total.

Every panel leads with the outcome and the reason, because that is the question being
asked. The evidence is underneath for the reader who wants to check.

`rich` drops colour automatically when output is redirected, so a piped run is plain text
without anything here having to care.
"""

from decimal import Decimal

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from galatiq.models import (
    Finding,
    Invoice,
    Outcome,
    PaymentStatus,
    RunRecord,
    Severity,
)

console = Console()

_OUTCOME_STYLE = {
    Outcome.APPROVED: "bold green",
    Outcome.REJECTED: "bold red",
    Outcome.HELD_FOR_REVIEW: "bold yellow",
}

_SEVERITY_STYLE = {
    Severity.CRITICAL: "red",
    Severity.WARN: "yellow",
    Severity.INFO: "dim",
}

# Five days, in seconds. The baseline the brief gives for the current manual process, and
# the number every latency claim is measured against.
_MANUAL_BASELINE_SECONDS = 5 * 24 * 60 * 60


def banner(provider: str | None, model: str | None, as_of) -> None:
    """What is about to run, and with what.

    The model is named up front because it is recorded against every decision. Anyone
    reading a payment months later should be able to tie it back to the thing that made
    it, and that starts with it being visible at the time.
    """
    console.print(
        Text.assemble(
            ("galatiq", "bold"),
            ("  ·  ", "dim"),
            (f"{provider or 'unknown'}/{model or 'unknown'}", "cyan"),
            ("  ·  ", "dim"),
            (f"as of {as_of}", "dim"),
        )
    )


def invoice_panel(invoice: Invoice) -> None:
    """What the system read, before what it concluded."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()

    table.add_row("Invoice", invoice.invoice_number or "[red]none found[/red]")
    table.add_row("Vendor", invoice.vendor or "[red]none named[/red]")
    if invoice.due_date_raw:
        table.add_row("Due", invoice.due_date_raw)
    table.add_row(
        "Total",
        f"{invoice.total_raw or invoice.total or '—'} {invoice.currency or ''}".strip(),
    )

    if invoice.line_items:
        lines = Table(box=None, pad_edge=False, show_edge=False)
        lines.add_column("Item")
        lines.add_column("Qty", justify="right")
        lines.add_column("Unit", justify="right")

        for line in invoice.line_items:
            # The raw name is shown when it differs from the canonical one, because a
            # reviewer checking against the document needs to see what the document said.
            name = line.item or f"[red]{line.raw_name}[/red]"
            if line.item and line.item != line.raw_name:
                name = f"{line.item} [dim]({line.raw_name})[/dim]"
            lines.add_row(
                name,
                str(line.quantity if line.quantity is not None else line.quantity_raw or "—"),
                str(line.unit_price_raw or line.unit_price or "—"),
            )
        table.add_row("Items", lines)

    console.print(Panel(table, title="Extracted", border_style="dim", expand=False))


def findings_table(findings: list[Finding]) -> None:
    """Everything that was noticed, worst first."""
    if not findings:
        console.print("[green]No findings.[/green]")
        return

    table = Table(show_header=True, header_style="dim", expand=True)
    table.add_column("", width=8)
    table.add_column("Finding")
    table.add_column("Evidence", style="dim", overflow="fold")

    for finding in findings:
        table.add_row(
            Text(finding.severity, style=_SEVERITY_STYLE[finding.severity]),
            f"[bold]{finding.code}[/bold]\n{finding.message}",
            finding.evidence or "",
        )

    console.print(table)


def decision_panel(record: RunRecord) -> None:
    """The answer, and why."""
    outcome = record.outcome or Outcome.HELD_FOR_REVIEW
    style = _OUTCOME_STYLE[outcome]

    body = Table.grid(padding=(0, 2))
    body.add_column(style="dim", justify="right")
    body.add_column()

    body.add_row("Risk", _risk_bar(record.risk_score))
    if record.policy_refs:
        body.add_row("Rules", ", ".join(record.policy_refs))
    if record.payment_status:
        body.add_row("Payment", str(record.payment_status))

    console.print(
        Panel(
            Group(Text(record.rationale or "No rationale recorded."), "", body),
            title=Text(str(outcome), style=style),
            border_style=style,
            expand=False,
        )
    )


def _risk_bar(score: int) -> str:
    filled = round(score / 10)
    colour = "red" if score >= 60 else "yellow" if score >= 30 else "green"
    return f"[{colour}]{'█' * filled}{'░' * (10 - filled)}[/{colour}] {score}"


def batch_row(record: RunRecord) -> None:
    """One line per document, printed as it completes."""
    name = record.source_path.rsplit("/", 1)[-1]

    if record.error:
        console.print(f"  [red]ERROR[/red]     {name}  [dim]{record.error}[/dim]")
        return

    outcome = record.outcome or Outcome.HELD_FOR_REVIEW
    style = _OUTCOME_STYLE[outcome]
    reason = record.findings[0].message if record.findings else ""

    # An approval that did not release money has to say so. Under concurrency two
    # documents describing the same invoice can both be approved -- they snapshot the
    # ledger before either has paid -- and the UNIQUE constraint then lets exactly one
    # through. The money is safe either way; a row reading plain "APPROVED" for the
    # second one would be a report that lies about a payment.
    if outcome is Outcome.APPROVED and record.payment_status is PaymentStatus.ALREADY_PAID:
        console.print(
            f"  [{style}]{str(outcome):<9}[/{style}] {name:<26} "
            f"[yellow]already paid — no second payment made[/yellow]"
        )
        return

    console.print(
        f"  [{style}]{str(outcome):<9}[/{style}] {name:<26} "
        f"[dim]{reason[:60]}[/dim]"
    )


def summary(result, elapsed_seconds: float) -> None:
    """The business case, from measurements rather than estimates."""
    records = result.records
    by_outcome = {o: [r for r in records if r.outcome is o] for o in Outcome}
    failed = [r for r in records if r.error]

    prevented = sum(
        (r.usd_total or Decimal("0") for r in by_outcome[Outcome.REJECTED]),
        Decimal("0"),
    )

    table = Table.grid(padding=(0, 3))
    table.add_column(style="dim", justify="right")
    table.add_column()

    table.add_row("Documents", str(len(records)))
    table.add_row("Unique invoices", str(result.unique_invoices))
    table.add_row("", "")
    table.add_row("Approved", f"[green]{len(by_outcome[Outcome.APPROVED])}[/green]")
    table.add_row(
        "Held for review", f"[yellow]{len(by_outcome[Outcome.HELD_FOR_REVIEW])}[/yellow]"
    )
    table.add_row("Rejected", f"[red]{len(by_outcome[Outcome.REJECTED])}[/red]")
    if failed:
        table.add_row("Failed", f"[red]{len(failed)}[/red]")

    duplicates = [
        r for r in records if r.payment_status is PaymentStatus.ALREADY_PAID
    ]
    if duplicates:
        table.add_row(
            "Duplicate payments avoided", f"[yellow]{len(duplicates)}[/yellow]"
        )

    if prevented:
        table.add_row("", "")
        # Rejected only. A held invoice is pending a human, not prevented -- counting it
        # would inflate the number, and an inflated number is the kind of thing a reader
        # checks and then stops trusting the rest of the page.
        table.add_row("Bad payments prevented", f"[bold]${prevented:,.2f}[/bold]")

    if records:
        per_invoice = elapsed_seconds / len(records)
        speedup = int(_MANUAL_BASELINE_SECONDS / per_invoice) if per_invoice else 0
        table.add_row("", "")
        table.add_row("Elapsed", f"{elapsed_seconds:.1f}s")
        table.add_row(
            "Per document",
            f"{per_invoice:.1f}s  [dim]vs a 5-day manual turnaround "
            f"(~{speedup:,}x)[/dim]",
        )

    console.print(Panel(table, title="Summary", border_style="dim", expand=False))
