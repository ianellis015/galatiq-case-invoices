"""Tests for date parsing.

The interesting cases are the failures. An unparseable date is a normal property of a
real invoice, not an error, and how it fails decides whether the system can tell "no due
date was stated" from "a due date was stated and it is not a date".
"""

from datetime import date

import pytest

from galatiq.dates import is_parseable, parse_date


class TestFormatsInTheCorpus:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("2026-01-24", date(2026, 1, 24)),      # JSON, XML, most CSVs
            ("01/28/2026", date(2026, 1, 28)),      # INV-1007
            ("26-Jan-2026", date(2026, 1, 26)),     # INV-1012's PDF
            ("26 Jan 2026", date(2026, 1, 26)),
            ("Jan 26, 2026", date(2026, 1, 26)),
            ("January 26, 2026", date(2026, 1, 26)),
            ("2026/01/24", date(2026, 1, 24)),
            ("24.01.2026", date(2026, 1, 24)),
        ],
    )
    def test_parses(self, text, expected):
        assert parse_date(text) == expected

    def test_whitespace_is_tolerated(self):
        assert parse_date("  2026-01-24  ") == date(2026, 1, 24)

    def test_iso_with_a_time_component(self):
        assert parse_date("2026-01-24T09:30:00") == date(2026, 1, 24)


class TestUnparseable:
    """None, not an exception. The invoice still has to reach a decision."""

    def test_yesterday(self):
        """INV-1003. A word, not a date -- and the reason that invoice is rejected."""
        assert parse_date("yesterday") is None

    def test_null(self):
        """INV-1009's due date."""
        assert parse_date(None) is None

    def test_empty_and_whitespace(self):
        assert parse_date("") is None
        assert parse_date("   ") is None

    def test_ocr_damage_is_not_repaired(self):
        """INV-1012 has "2O26" -- a letter O where a zero belongs.

        Returning None keeps the damage visible. Guessing "2026" would mean the system
        silently rewrote a date on a document it moves money against.
        """
        assert parse_date("26-Jan-2O26") is None

    def test_nonsense(self):
        assert parse_date("not a date at all") is None
        assert parse_date("2026-13-45") is None


class TestAmbiguity:
    def test_us_convention_is_assumed(self):
        """01/28/2026 can only be January 28 -- there is no month 28.

        03/04/2026 genuinely cannot be resolved from the string, and US convention is
        assumed because the corpus is a US manufacturer's accounts payable. Recorded in
        the README rather than left implicit, since guessing the other way would shift
        a due date by months.
        """
        assert parse_date("03/04/2026") == date(2026, 3, 4)


class TestIsParseable:
    def test_reads_as_a_question(self):
        assert is_parseable("2026-01-24")
        assert not is_parseable("yesterday")
        assert not is_parseable(None)
