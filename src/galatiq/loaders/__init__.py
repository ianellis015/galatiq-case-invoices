"""Reading invoice files off disk.

    from galatiq.loaders import load, load_many, discover

    doc = load("data/invoices/invoice_1006.csv")     # one file
    docs = load_many("data/invoices")                # a directory
    paths = discover("data/invoices/*.csv")          # a glob

Every document comes back with `raw_text`, whatever its format. Formats with a
dedicated parser also come back with a `structural_hint` -- an independent second
reading for the extractor to use as context and the critic to check against.

`load()` raises only when there is no file to read. Format problems, decode failures
and unparseable content all come back as findings on the document, because a batch of
twenty invoices should not stop for one bad file, and a human needs to be told which
file and why.
"""

import glob
from pathlib import Path

from galatiq.loaders.base import (
    CANONICAL_FIELDS,
    LoadedDocument,
    hint_is_useful,
    read_text_safely,
)
from galatiq.loaders.csv_loader import load_csv
from galatiq.loaders.fallback import load_unknown
from galatiq.loaders.json_loader import load_json
from galatiq.loaders.pdf import load_pdf
from galatiq.loaders.text import load_text
from galatiq.loaders.xml_loader import load_xml

# Extensions with a dedicated loader. Everything else goes to `load_unknown`, which
# reads it as text -- so this map is a list of optimizations, not a list of
# capabilities.
_LOADERS = {
    ".txt": load_text,
    ".pdf": load_pdf,
    ".json": load_json,
    ".xml": load_xml,
    ".csv": load_csv,
}

KNOWN_EXTENSIONS = frozenset(_LOADERS)


def load(path: str | Path) -> LoadedDocument:
    """Read one invoice file, whatever its format.

    Raises only FileNotFoundError, and only for a path that is not a file. An
    unrecognized extension is not an error -- it is a document with no structural
    cross-check, which the extractor handles the same way it handles a plain text
    invoice.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")

    loader = _LOADERS.get(path.suffix.lower(), load_unknown)
    return loader(path)


def discover(path: str | Path) -> list[Path]:
    """Find invoice files from a file path, a directory, or a glob pattern.

    Returns every match, sorted by name so batch ordering is deterministic.

    Extension is not a filter. Since any text file is processable, filtering by
    extension here would reintroduce exactly the brittleness the fallback removes --
    a directory of .yaml invoices would come back empty rather than being read.
    Hidden files are skipped; a file that turns out to be unreadable produces a
    finding when it is loaded, which is where that belongs.

    Directories are not walked recursively -- an invoice inbox is flat, and recursing
    would sweep up whatever happens to be nested underneath it.

    Note a directory of the provided corpus yields **twenty** documents, not sixteen.
    INV-1011, 1012 and 1013 each exist as a pair whose contents genuinely differ, so
    both members are real documents. Recognising that two describe the same invoice
    is the dedupe check's job, and the ledger's UNIQUE constraint means a pair can
    only ever produce one payment.
    """
    raw = str(path)

    # Glob first: a pattern is not a path that happens to be missing. stdlib glob
    # handles absolute and relative patterns identically, which Path.glob does not.
    if any(char in raw for char in "*?["):
        return sorted(
            p for p in (Path(match) for match in glob.glob(raw)) if p.is_file()
        )

    target = Path(path)

    if target.is_dir():
        return sorted(
            p
            for p in target.iterdir()
            if p.is_file() and not p.name.startswith(".")
        )

    if target.is_file():
        return [target]

    raise FileNotFoundError(f"no such file or directory: {target}")


def load_many(path: str | Path) -> list[LoadedDocument]:
    """Load every invoice file found at a path, directory, or glob.

    No document can fail the batch. A file that cannot be read comes back carrying a
    CRITICAL finding and goes on to be held for a human, which is the same treatment
    an invoice with bad arithmetic gets -- one reporting channel, not two.
    """
    return [load(p) for p in discover(path)]


__all__ = [
    "load",
    "load_many",
    "discover",
    "LoadedDocument",
    "KNOWN_EXTENSIONS",
    "CANONICAL_FIELDS",
    "hint_is_useful",
    "read_text_safely",
]
