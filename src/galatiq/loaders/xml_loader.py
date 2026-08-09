"""XML invoices.

Fitted to INV-1014's schema, and openly so: it assumes <header> and <totals> are
containers and that a line item's name lives in <name>. A vendor using UBL or cXML
produces no useful hint from this parser.

That is a bounded weakness rather than a hole, because a hint is optional. An
unrecognized XML schema falls through to text extraction -- the same path an email
invoice takes -- with an INFO finding saying the cross-check was unavailable. The
generalization mechanism is the model, not more parser cases.
"""

from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from galatiq.loaders.base import (
    LoadedDocument,
    hint_is_useful,
    hint_unavailable,
    read_text_safely,
)

# Sections whose children are invoice-level fields. The grouping is presentational,
# so it is discarded and the children are lifted to the top level.
_FLATTENED_SECTIONS = {"header", "totals"}

# The XML calls a line item's name <name>; everywhere else in the corpus it is `item`.
_LINE_ITEM_ALIASES = {"name": "item"}


def load_xml(path: Path) -> LoadedDocument:
    """Read an .xml invoice as text, pre-parsing it when the schema is recognized."""
    text, findings = read_text_safely(path)
    hint: dict[str, Any] | None = None

    if text:
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError as exc:
            findings.append(hint_unavailable(path, f"invalid XML: {exc}"))
        else:
            candidate = _to_flat_dict(root)
            if hint_is_useful(candidate):
                hint = candidate
            else:
                findings.append(
                    hint_unavailable(path, "unrecognized XML schema")
                )

    return LoadedDocument(
        source_path=str(path),
        source_format="xml",
        raw_text=text,
        structural_hint=hint,
        hint_source="xml" if hint else None,
        findings=findings,
    )


def _to_flat_dict(root: ElementTree.Element) -> dict[str, Any]:
    """Unwrap the presentational sections into the canonical flat shape."""
    data: dict[str, Any] = {}

    for child in root:
        if child.tag in _FLATTENED_SECTIONS:
            for field in child:
                data[field.tag] = _text(field)
        elif child.tag == "line_items":
            data["line_items"] = [_line_item(item) for item in child]
        else:
            data[child.tag] = _text(child)

    return data


def _line_item(element: ElementTree.Element) -> dict[str, Any]:
    """One <item> element, with its child tags as keys."""
    return {
        _LINE_ITEM_ALIASES.get(field.tag, field.tag): _text(field)
        for field in element
    }


def _text(element: ElementTree.Element) -> str | None:
    """Element text, or None for an empty element.

    An empty element means the document stated nothing, which differs from stating an
    empty string -- and the difference decides whether a missing field becomes a
    finding.
    """
    return element.text.strip() if element.text is not None else None
