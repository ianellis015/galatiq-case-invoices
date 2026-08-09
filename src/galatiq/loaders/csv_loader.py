"""CSV invoices -- two different formats wearing the same extension.

The only parser here with real logic, because CSV is where the corpus hides its
structural traps. Both produce a *wrong answer* rather than an error, which is what
makes them worth explicit machinery:

**Vertical (INV-1006)** -- a `field,value` sheet where `item`, `quantity` and
`unit_price` repeat as interleaved triples. The obvious `dict(rows)` keeps the last
value for each duplicated key, silently discarding WidgetA and leaving an invoice
whose line items no longer sum to its own subtotal. Nothing raises. The invoice
quietly becomes a different invoice.

**Row-per-item (INV-1007, INV-1015)** -- one row per line item, then trailing summary
rows:

    ,,,,,,Subtotal:,14750.00
    ,,,,,,Tax (6%):,885.00
    ,,,,,,Total:,15525.00

Treat those as data and the invoice grows three phantom line items with no name, no
quantity and no price. INV-1015 -- documented as clean -- has them too, so this is the
normal shape of a row-per-item sheet rather than a quirk of one file.

Both are recognized by their header row. A sheet matching neither produces no hint and
falls through to text extraction, which is the correct answer for a dialect nobody
anticipated: this parser knows two shapes, and says so rather than guessing.
"""

import csv
import io
from pathlib import Path
from typing import Any

from galatiq.loaders.base import (
    LoadedDocument,
    hint_is_useful,
    hint_unavailable,
    read_text_safely,
)

# A vertical sheet announces itself: two columns, named field and value.
_VERTICAL_HEADER = ("field", "value")

# Keys belonging to the line item currently being built rather than to the invoice.
_LINE_ITEM_KEYS = frozenset({"item", "quantity", "unit_price", "amount", "note"})

# Source spellings mapped onto the canonical key set. Deliberately short: this is a
# fast path for shapes we recognize, not an attempt to anticipate every dialect.
# Anything unmatched produces no hint and takes the text route.
_ALIASES = {
    "qty": "quantity",
    "price": "unit_price",
    "unit_cost": "unit_price",
    "line_total": "amount",
    "tax": "tax_amount",
    "invoice": "invoice_number",
}


def load_csv(path: Path) -> LoadedDocument:
    """Read a .csv invoice as text, pre-parsing it when the header is recognized."""
    text, findings = read_text_safely(path)
    hint: dict[str, Any] | None = None

    if text:
        try:
            candidate = _parse(text)
        except Exception as exc:  # noqa: BLE001 - a hint is optional; text is not
            findings.append(hint_unavailable(path, f"{type(exc).__name__}: {exc}"))
        else:
            if hint_is_useful(candidate):
                hint = candidate
            else:
                findings.append(
                    hint_unavailable(path, "unrecognized CSV layout")
                )

    return LoadedDocument(
        source_path=str(path),
        source_format="csv",
        raw_text=text,
        structural_hint=hint,
        hint_source="csv" if hint else None,
        findings=findings,
    )


def _parse(text: str) -> dict[str, Any] | None:
    """Dispatch on the header row. Returns None for an empty sheet."""
    rows = [
        row
        for row in csv.reader(io.StringIO(text))
        if any(cell.strip() for cell in row)
    ]

    if not rows:
        return None

    header = [_normalize_key(cell) for cell in rows[0]]

    if tuple(header[:2]) == _VERTICAL_HEADER:
        return _parse_vertical(rows[1:])
    return _parse_row_per_item(header, rows[1:])


def _normalize_key(cell: str) -> str:
    """"Invoice Number" -> "invoice_number", "Qty" -> "quantity"."""
    key = cell.strip().rstrip(":").strip().lower().replace(" ", "_")
    return _ALIASES.get(key, key)


def _parse_vertical(rows: list[list[str]]) -> dict[str, Any]:
    """Walk `field,value` pairs in order, grouping repeated keys into line items.

    Order is the entire mechanism, which is why this is a loop and not a dict
    comprehension. Each `item` row opens a new line item, and the `quantity` and
    `unit_price` rows following it belong to that item. Position in the file is the
    only thing distinguishing WidgetA's quantity from WidgetB's -- the keys are
    identical.
    """
    data: dict[str, Any] = {}
    line_items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for row in rows:
        if len(row) < 2:
            continue

        key = _normalize_key(row[0])
        value = row[1].strip()

        if key == "item":
            if current is not None:
                line_items.append(current)
            current = {"item": value}
        elif key in _LINE_ITEM_KEYS and current is not None:
            current[key] = value
        else:
            data[key] = value

    if current is not None:
        line_items.append(current)

    data["line_items"] = line_items
    return data


def _parse_row_per_item(header: list[str], rows: list[list[str]]) -> dict[str, Any]:
    """One row per line item, with trailing summary rows excluded.

    The test for a line item is a populated invoice number. Summary rows leave it
    blank, which is a more reliable signal than counting columns or assuming the
    footer is exactly three rows.

    A sheet with no invoice-number column at all yields no line items, `hint_is_useful`
    rejects it, and the document takes the text route -- which is the right outcome
    for a layout this parser does not understand.
    """
    data: dict[str, Any] = {}
    line_items: list[dict[str, Any]] = []

    for row in rows:
        cells = dict(zip(header, (cell.strip() for cell in row)))

        if cells.get("invoice_number"):
            # Invoice-level fields repeat on every row; the first wins. Later
            # disagreement is a document problem for the checks to report rather
            # than something to silently reconcile here.
            for key in ("invoice_number", "vendor", "date", "due_date", "currency"):
                if cells.get(key) and key not in data:
                    data[key] = cells[key]

            line_items.append(
                {
                    key: cells[key]
                    for key in ("item", "quantity", "unit_price", "amount")
                    if cells.get(key)
                }
            )
        else:
            _apply_summary_row(row, data)

    data["line_items"] = line_items
    return data


def _apply_summary_row(row: list[str], data: dict[str, Any]) -> None:
    """Read a trailing `Subtotal:` / `Tax (6%):` / `Total:` row into invoice totals.

    Located by scanning for the last two populated cells rather than by column index,
    since the label sits wherever the sheet's author left it.

    The tax label is kept verbatim as `tax_label`. "Tax (6%)" carries a rate that
    check_math can use, but deriving it is interpretation -- this parser records what
    the document said and stops there.
    """
    populated = [cell.strip() for cell in row if cell.strip()]
    if len(populated) < 2:
        return

    label, value = populated[-2], populated[-1]
    normalized = label.rstrip(":").strip().lower()

    # "subtotal" is tested first: "total" is a suffix of it, not a prefix, but the
    # ordering makes the intent obvious to anyone editing this later.
    if normalized.startswith("subtotal"):
        data["subtotal"] = value
    elif normalized.startswith("tax"):
        data["tax_amount"] = value
        data["tax_label"] = label
    elif normalized.startswith("total"):
        data["total"] = value
