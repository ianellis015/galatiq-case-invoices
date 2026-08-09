"""Tests for the strict-mode schema transform.

This module is the most likely thing to silently break extraction: if the schema is
wrong the provider either rejects the request or -- worse -- stops enforcing the shape
and responses drift without anything failing. So the transform's output is pinned
here rather than trusted.

Pure functions, no network.
"""

from pydantic import BaseModel, Field

from galatiq.llm import AGENT_EXCLUDED_FIELDS, build_strict_schema, response_format
from galatiq.models import ApprovalDecision, Invoice, LineItem


class Nested(BaseModel):
    label: str
    amount: int | None = None


class Sample(BaseModel):
    name: str
    count: int = 0
    child: Nested | None = None
    tags: list[str] = Field(default_factory=list)


class TestEveryPropertyIsRequired:
    """Strict mode requires it, and our models are almost entirely optional.

    Presence is what varies across arbitrary invoices, so nearly every field carries a
    default -- which pydantic reads as "not required" and strict mode rejects.
    Optionality is expressed by allowing null, not by omitting the key.
    """

    def test_defaults_do_not_make_a_field_optional(self):
        schema = build_strict_schema(Sample)
        assert set(schema["required"]) == set(schema["properties"])

    def test_invoice_requires_all_of_its_properties(self):
        schema = build_strict_schema(Invoice)
        assert set(schema["required"]) == set(schema["properties"])

    def test_nested_models_too(self):
        schema = build_strict_schema(Invoice)
        line_item = schema["$defs"]["LineItem"]
        assert set(line_item["required"]) == set(line_item["properties"])


class TestAdditionalProperties:
    def test_false_at_the_root(self):
        assert build_strict_schema(Sample)["additionalProperties"] is False

    def test_false_on_nested_definitions(self):
        schema = build_strict_schema(Invoice)
        assert schema["$defs"]["LineItem"]["additionalProperties"] is False


class TestUnsupportedKeywords:
    """Constraints are stripped from the schema, not from validation.

    Pydantic re-applies every one of them when the response is parsed on the way back
    in. The schema constrains the model's output shape; we check the values.
    """

    def test_decimal_pattern_is_gone(self):
        """Pydantic's Decimal schema carries a regex strict mode will not accept."""
        schema = build_strict_schema(Invoice)
        assert "pattern" not in str(schema)

    def test_date_format_is_gone(self):
        schema = build_strict_schema(Invoice)
        assert schema["properties"]["issue_date"] == {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "title": "Issue Date",
        }

    def test_numeric_bounds_are_gone(self):
        """ApprovalDecision.risk_score is ge=0 le=100."""
        schema = build_strict_schema(ApprovalDecision)
        risk = schema["properties"]["risk_score"]

        assert "minimum" not in risk
        assert "maximum" not in risk
        assert risk["type"] == "integer"

    def test_defaults_are_gone(self):
        assert "default" not in str(build_strict_schema(Sample))

    def test_bounds_are_still_enforced_on_the_way_back_in(self):
        """The constraint left the schema, not the system."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ApprovalDecision.model_validate_json(
                '{"outcome": "APPROVED", "rationale": "x", '
                '"policy_refs": [], "risk_score": 500}'
            )


class TestMoneyIsTranscribed:
    """Money fields are strings in the schema, deliberately.

    Pydantic's default is a union of number, pattern-constrained string, and null. Two
    problems: the pattern is rejected by strict mode, and a numeric money field invites
    the model to *interpret* an amount. INV-1012's "$3,500.O0" should come back as
    those characters so the OCR damage stays visible.
    """

    def test_required_money_is_a_string(self):
        schema = build_strict_schema(LineItem)
        assert schema["properties"]["unit_price"]["type"] == ["string", "null"]

    def test_optional_money_is_string_or_null(self):
        schema = build_strict_schema(Invoice)
        assert schema["properties"]["total"]["type"] == ["string", "null"]

    def test_tax_rate_too(self):
        schema = build_strict_schema(Invoice)
        assert schema["properties"]["tax_rate"]["type"] == ["string", "null"]


class TestExclusions:
    def test_agent_excluded_fields_are_dropped(self):
        """Provenance is ours -- we know which file we opened. `extra` is decided when
        a document is mapped onto the model. Offering either invites invention."""
        schema = build_strict_schema(Invoice, exclude=AGENT_EXCLUDED_FIELDS)

        for field in AGENT_EXCLUDED_FIELDS:
            assert field not in schema["properties"]
            assert field not in schema["required"]

    def test_open_dict_would_otherwise_be_unrepresentable(self):
        """`extra` is dict[str, Any], and strict mode cannot express an open object.

        Excluding it is the correct shape rather than a workaround.
        """
        assert "extra" in Invoice.model_json_schema()["properties"]
        assert "extra" not in build_strict_schema(
            Invoice, exclude=AGENT_EXCLUDED_FIELDS
        )["properties"]

    def test_exclusion_is_optional(self):
        assert "source_path" in build_strict_schema(Invoice)["properties"]


class TestResponseFormat:
    def test_shape(self):
        payload = response_format(Invoice, exclude=AGENT_EXCLUDED_FIELDS)

        assert payload["type"] == "json_schema"
        assert payload["json_schema"]["name"] == "Invoice"
        assert "schema" in payload["json_schema"]

    def test_strict_is_on(self):
        """The difference between a schema the provider suggests and one it enforces.

        Without it the typed-output guarantee is advisory.
        """
        assert response_format(Invoice)["json_schema"]["strict"] is True


class TestEnumsSurvive:
    def test_enum_values_are_preserved(self):
        """The model has to be told which outcomes exist -- that is the constraint
        doing real work."""
        schema = build_strict_schema(ApprovalDecision)
        outcome = schema["properties"]["outcome"]
        target = schema["$defs"][outcome["$ref"].split("/")[-1]]

        assert set(target["enum"]) == {"APPROVED", "REJECTED", "HELD_FOR_REVIEW"}
