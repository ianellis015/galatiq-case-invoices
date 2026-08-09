"""JSON invoices.

The most explicit documents in the corpus, and the shape the other structural parsers
aim for. What comes out is a *hint*, not a bypass -- the extractor still reads the
text, and the hint gives the critic an independent reading to compare against.

Only one transformation happens: the nested vendor object is flattened, because
INV-1014's XML gives a bare vendor string and one downstream shape is better than two.
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from galatiq.loaders.base import (
    LoadedDocument,
    hint_is_useful,
    hint_unavailable,
    read_text_safely,
)


def load_json(path: Path) -> LoadedDocument:
    """Read a .json invoice as text, and pre-parse it when the JSON is valid.

    `parse_float=Decimal` is load-bearing. Without it json turns 250.00 into a Python
    float, and floats are rejected at the model boundary -- so every JSON invoice
    would fail to build an `Invoice`. Converting later is too late: the precision is
    gone the moment the float exists.

    Malformed JSON is a finding, not an exception. A vendor sending a truncated file
    should get a reasoned rejection, and the extractor can often still read what is
    there.
    """
    text, findings = read_text_safely(path)
    hint: dict[str, Any] | None = None

    if text:
        try:
            raw = json.loads(text, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            findings.append(hint_unavailable(path, f"invalid JSON: {exc.msg}"))
        else:
            if isinstance(raw, dict):
                candidate = _flatten_vendor(raw)
                if hint_is_useful(candidate):
                    hint = candidate
                else:
                    findings.append(
                        hint_unavailable(path, "no invoice number or line items found")
                    )
            else:
                findings.append(
                    hint_unavailable(path, "top-level value is not an object")
                )

    return LoadedDocument(
        source_path=str(path),
        source_format="json",
        raw_text=text,
        structural_hint=hint,
        hint_source="json" if hint else None,
        findings=findings,
    )


def _flatten_vendor(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn {"vendor": {"name": ..., "address": ...}} into two flat keys.

    Every JSON invoice nests the vendor; INV-1014's XML does not.

    INV-1009's vendor name is the empty string and its address is null. Both survive
    as-is -- an empty vendor is a DATA_INTEGRITY finding later, and it can only be
    reported if it is faithfully recorded now.
    """
    data = dict(raw)
    vendor = data.pop("vendor", None)

    if isinstance(vendor, dict):
        data["vendor"] = vendor.get("name")
        data["vendor_address"] = vendor.get("address")
    elif vendor is not None:
        data["vendor"] = vendor

    return data
