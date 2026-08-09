"""galatiq — an auditable invoice processing pipeline.

I structured the package around the four stages of the workflow, with the
deterministic and model-driven parts kept in separate namespaces:

    store/      persistence: inventory catalog, payment ledger
    loaders/    format parsing (txt, pdf, json, xml, csv)
    agents/     the LLM nodes (extractor, critics, normalizer, approver)
    checks/     deterministic validation (stock, math, dates, ...)
    policy/     the approval rules, as configuration
    graph.py    the LangGraph wiring

My governing principle throughout: the LLM handles ambiguity, deterministic code
handles correctness. Nothing in `store/` or `checks/` ever calls a model, and I
enforce that by keeping them in packages that have no client dependency to reach for.
"""

__version__ = "0.1.0"
