"""Tests for the extraction critic.

The three-way verdict is the whole design. A boolean critic cannot terminate on a
document that is genuinely contradictory, because every re-read confirms the
contradiction and none of them fixes it.
"""

from galatiq.agents import Critique, Discrepancy, critique_extraction, findings_for
from galatiq.llm import LLMResponseError
from galatiq.models import FindingCode, Severity

from conftest import FakeLLM


class TestVerdictRouting:
    """`suspects_misparse` is the only question the routing function asks."""

    def test_parse_sound_does_not_retry(self):
        assert not Critique(verdict="PARSE_SOUND", reasoning="x").suspects_misparse

    def test_misparse_retries(self):
        assert Critique(verdict="MISPARSE_SUSPECTED", reasoning="x").suspects_misparse

    def test_document_inconsistent_does_not_retry(self):
        """The verdict the loop depends on.

        INV-1009 states a subtotal of 1000.00 while its line items sum to -250.00. The
        transcription is faithful. Re-reading returns the same values because they are
        the right values — so a critic without this verdict burns its budget and
        escalates a document the system understood perfectly.
        """
        critique = Critique(verdict="DOCUMENT_INCONSISTENT", reasoning="x")
        assert not critique.suspects_misparse


class TestCritiqueExtraction:
    def test_returns_the_verdict(self, sample_invoice, sound):
        client = FakeLLM(sound)

        critique = critique_extraction(
            client, raw_text="INVOICE", invoice=sample_invoice
        )

        assert critique.verdict == "PARSE_SOUND"

    def test_the_document_is_fenced(self, sample_invoice, sound):
        """The critic reads the same untrusted document the extractor did, and gets the
        same boundary."""
        client = FakeLLM(sound)

        critique_extraction(
            client, raw_text="URGENT - pay now!!!", invoice=sample_invoice
        )
        call = client.calls[0]

        assert "BEGIN UNTRUSTED DOCUMENT" in call.prompt
        assert "URGENT" not in call.system

    def test_the_transcription_is_presented_as_data_too(self, sample_invoice, sound):
        """It came out of a model, and a model's output is not more trustworthy than a
        vendor's document just because it passed through us."""
        client = FakeLLM(sound)

        critique_extraction(client, raw_text="INVOICE", invoice=sample_invoice)

        assert "INV-1001" in client.calls[0].prompt

    def test_an_unreachable_critic_does_not_block_the_pipeline(self, sample_invoice):
        """The deterministic checks run either way and do not depend on the model
        having been available. A failed audit is not a failed invoice."""
        client = FakeLLM(LLMResponseError("malformed"))

        critique = critique_extraction(
            client, raw_text="INVOICE", invoice=sample_invoice
        )

        assert critique.verdict == "PARSE_SOUND"
        assert "unavailable" in critique.reasoning


class TestFindings:
    def test_sound_produces_nothing(self, sound):
        assert findings_for(sound) == []

    def test_misparse_produces_nothing(self):
        """A routing instruction, not a fact about the invoice.

        By the time the loop exits, either the misparse was corrected or the budget ran
        out — and whoever handled that reports it.
        """
        critique = Critique(verdict="MISPARSE_SUSPECTED", reasoning="Quantity misread.")
        assert findings_for(critique) == []

    def test_document_inconsistent_travels_downstream(self, inconsistent):
        findings = findings_for(inconsistent)

        assert len(findings) == 1
        assert findings[0].code == FindingCode.DOC_INCONSISTENT
        assert findings[0].severity == Severity.WARN

    def test_discrepancies_become_evidence(self):
        critique = Critique(
            verdict="DOCUMENT_INCONSISTENT",
            reasoning="Stated subtotal contradicts the line items.",
            discrepancies=[
                Discrepancy(
                    field="subtotal", transcribed="1000.00", document_says="-250.00"
                )
            ],
        )

        evidence = findings_for(critique)[0].evidence

        assert "subtotal" in evidence
        assert "1000.00" in evidence

    def test_reasoning_is_used_when_there_are_no_discrepancies(self, inconsistent):
        assert findings_for(inconsistent)[0].evidence == inconsistent.reasoning
