"""The agents: nodes that call a language model.

Two so far, and they are deliberately separate agents rather than one prompted twice:

  * **Extractor** -- a document becomes an `Invoice`. Transcription, not judgement.
  * **Extraction critic** -- did the extractor misread? Adversarial re-read.

A generator and its reviewer sharing a prompt is one agent talking to itself, and it
inherits the mistake it is meant to catch.

Still to come: the normalizer (item names to catalog entries), the approver, and the
approval critic. Payment is deliberately not an agent -- it is a tool call behind a
deterministic edge, and no model is given discretion over releasing funds.
"""

from galatiq.agents.extract_critic import (
    Critique,
    Discrepancy,
    critique_extraction,
    findings_for,
)
from galatiq.agents.extractor import ExtractionOutcome, extract_invoice
from galatiq.agents.prompts import (
    CRITIC_SYSTEM,
    EXTRACTOR_SYSTEM,
    critique_messages,
    extraction_messages,
    neutralise_sentinels,
    wrap_document,
)

__all__ = [
    "extract_invoice",
    "ExtractionOutcome",
    "critique_extraction",
    "findings_for",
    "Critique",
    "Discrepancy",
    "extraction_messages",
    "critique_messages",
    "wrap_document",
    "neutralise_sentinels",
    "EXTRACTOR_SYSTEM",
    "CRITIC_SYSTEM",
]
