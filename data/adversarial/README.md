# Adversarial fixtures

These files are **mine**, not part of the assessment's provided corpus. They live in
their own directory so it stays obvious which documents were given and which I wrote.

Each one exists to exercise a code path the provided corpus cannot reach. All 20 files
in `data/invoices/` are formats the system has a parser for, so nothing in there tests
what happens when a document arrives in a shape nobody anticipated — which, for a
workflow whose invoices arrive by email from arbitrary vendors, is the case that
matters most.

| File | What it proves |
|---|---|
| `invoice_A001.yaml` | A format with no parser is still read, extracted and decided on. There is no YAML loader and there is not going to be one — the file takes the same text route a `.txt` invoice takes, and the only difference is a WARN noting the structural cross-check was unavailable. |
| `invoice_A002.bin` | A genuinely unreadable file produces a CRITICAL `UNREADABLE_DOCUMENT` finding naming the file, and is held for a human. It does not stop the batch and it does not become an invoice with no content. |

The governing rule both serve:

> No input crashes the pipeline. Every input produces a decision with reasoning.

More adversarial invoices — a markdown table, a forwarded email, a CSV with reordered
columns and `VAT` instead of `Tax` — land in the testing phase, where they can be run
end to end rather than only through the loader.
