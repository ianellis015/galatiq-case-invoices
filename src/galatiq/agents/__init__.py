"""The agents: nodes that call a language model.

Five of them, and I kept each one a separate agent with its own prompt rather than one
agent prompted five ways:

  * **Extractor** -- a document becomes an `Invoice`. Transcription, not judgement.
  * **Extraction critic** -- did the extractor misread? Adversarial re-read.
  * **Normalizer** -- the names a document used become the names the catalog uses.
  * **Approver** -- findings become a decision, with reasons a human can read.
  * **Approval critic** -- what did the approver miss? A second pair of eyes.

The pairing is the point. A generator and its reviewer sharing a prompt is one agent
talking to itself, and it inherits the mistake it is meant to catch.

Payment is deliberately not among them. It is a tool call behind a deterministic edge,
and I give no model discretion over releasing funds.
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
