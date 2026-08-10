"""Resolving what a document called an item to what the catalog calls it.

Documents name the same product differently: "WidgetA", "Widget A", "WidgetA (rush
order)". The stock check needs one name to aggregate against, and the catalog is the
authority for what that name is.

**This deliberately does not use fuzzy matching, and that is the whole design.**

`difflib.SequenceMatcher("WidgetC", "WidgetA")` scores about 0.857. Any cutoff low enough
to accept "Widget A" also accepts "WidgetC" -- and INV-1016's WidgetC would silently
become WidgetA, an item that is in stock, and the invoice would be approved. A wrong
match produces a payment; a missed match produces a rejection someone can appeal. The
failure modes are not symmetric, so the matching is conservative.

Instead: differences in *formatting* are resolved, differences in *content* never are.
Strip the qualifiers a human would ignore, remove punctuation and spacing, lowercase, and
require an exact match on what is left.

    WidgetA (rush order)  ->  widgeta   ->  WidgetA
    Widget A              ->  widgeta   ->  WidgetA
    Gadget X              ->  gadgetx   ->  GadgetX
    WidgetC               ->  widgetc   ->  no match, correctly

The model is a fallback for names that survive that and still do not match -- "WIDGET_A
DELUXE", "Item: WidgetA" -- and its instructions say plainly that a different product
code is a different product.
"""

import re

from galatiq.agents.prompts import normalization_messages
from galatiq.llm import LLMClient, LLMResponseError
from galatiq.models import Invoice, LineItem
from pydantic import BaseModel, ConfigDict, Field

# Trailing qualifiers a human reads past: "(rush order)", "[backordered]", "- expedited".
_QUALIFIER = re.compile(r"\s*[\(\[].*?[\)\]]\s*|\s+-\s+.*$")

# Everything that is not a letter or digit. Spacing, punctuation, underscores.
_NOISE = re.compile(r"[^a-z0-9]+")


class ItemMatch(BaseModel):
    """One resolution attempt, with its reasoning."""

    model_config = ConfigDict(extra="forbid")

    raw_name: str
    item: str | None
    reasoning: str


class ItemMatches(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: list[ItemMatch] = Field(default_factory=list)


def canonical_key(name: str) -> str:
    """Reduce a name to the part that identifies the product.

    Qualifiers first, then punctuation and spacing. "WidgetA (rush order)" becomes
    "widgeta"; strip in the other order and it becomes "widgetarushorder", which matches
    nothing.
    """
    without_qualifier = _QUALIFIER.sub("", name or "").strip()
    return _NOISE.sub("", without_qualifier.lower())


def build_index(catalog: dict[str, int] | list[str]) -> dict[str, str]:
    """Map canonical keys to catalog names."""
    names = catalog if isinstance(catalog, list) else list(catalog)
    return {canonical_key(name): name for name in names}


def resolve_deterministically(raw_name: str, index: dict[str, str]) -> str | None:
    """Exact match on the canonical key, or nothing."""
    return index.get(canonical_key(raw_name))


def normalize_invoice(
    invoice: Invoice,
    catalog: dict[str, int] | list[str],
    *,
    client: LLMClient | None = None,
) -> Invoice:
    """Fill `LineItem.item` for every line the catalog recognises.

    Deterministic resolution first, for every line. The model is asked only about names
    that survive it, and only once per invoice -- a single call covering every unresolved
    name rather than one per line.

    A name that resolves to nothing keeps `item` as None, which the stock check reports
    as UNKNOWN_ITEM. That is the correct outcome for INV-1008's "SuperGizmo" and
    INV-1016's "WidgetC", and the reason this function would rather return None than
    guess.
    """
    index = build_index(catalog)

    resolved = [
        line.model_copy(update={"item": resolve_deterministically(line.raw_name, index)})
        for line in invoice.line_items
    ]

    unresolved = sorted(
        {line.raw_name for line in resolved if line.item is None and line.raw_name}
    )

    if unresolved and client is not None:
        matches = _ask_model(client, unresolved, sorted(index.values()))
        resolved = [_apply_match(line, matches) for line in resolved]

    return invoice.model_copy(update={"line_items": resolved})


def _ask_model(
    client: LLMClient, unresolved: list[str], catalog: list[str]
) -> dict[str, str]:
    """Ask about the names deterministic matching could not place.

    A failed call is not a failed invoice. An unreachable model leaves the names
    unresolved, which is the same outcome as a genuine non-match -- the invoice is
    reported as containing unknown items and a human looks at it. Degrading toward
    caution is the right direction for something that authorises payments.
    """
    try:
        result = client.complete(
            normalization_messages(unresolved, catalog), ItemMatches
        )
    except LLMResponseError:
        return {}

    return {
        match.raw_name: match.item
        for match in result.value.matches
        # Only a name that is actually in the catalog. A model returning "WidgetD"
        # would otherwise invent an inventory item, and the catalog is the authority
        # for what exists.
        if match.item in catalog
    }


def _apply_match(line: LineItem, matches: dict[str, str]) -> LineItem:
    if line.item is not None:
        return line

    matched = matches.get(line.raw_name)
    return line.model_copy(update={"item": matched}) if matched else line
