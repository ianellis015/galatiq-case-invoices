"""Turning a pydantic model into a schema the API will enforce.

Structured output only constrains a response if the provider accepts the schema, and
strict mode is fussier than pydantic's default output in three ways:

  1. **Every property must appear in `required`.** Pydantic omits fields that have
     defaults -- which, in our models, is nearly all of them, since presence is
     exactly what varies across arbitrary invoices. Optionality is expressed by
     allowing null, not by leaving a field out.
  2. **`additionalProperties: false` on every object.** Our models already set
     `extra="forbid"`, so pydantic emits this at the top level, but nested objects
     and `$defs` need it too.
  3. **A limited keyword set.** `pattern`, `format`, `minimum` and friends are
     rejected. Pydantic emits `format: date` for date fields and a `pattern` for
     Decimal, both of which have to go.

None of this is negotiable at the call site, so it happens here, once, and is pinned
by tests. This module is the most likely thing to silently break extraction, which is
why it is a pure function with no network anywhere near it.
"""

from typing import Any

from pydantic import BaseModel

# Keywords strict structured output does not accept. Stripped recursively.
#
# These are all validation constraints rather than shape, so removing them loses
# nothing that matters: pydantic re-applies every one of them when the response is
# validated on the way back in. The schema's job is to constrain the model's output
# shape; ours is to check the values.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "pattern",
        "format",
        "default",
        "examples",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def build_strict_schema(
    model: type[BaseModel],
    *,
    exclude: frozenset[str] | None = None,
) -> dict[str, Any]:
    """A JSON schema for `model`, adjusted for strict structured output.

    `exclude` drops top-level fields the model should not be asked to fill.
    Provenance (`source_path`, `source_format`) and `extra` are ours to set, and
    offering them to the model only invites it to invent values for them.

    `extra` has a second problem: it is an open dict, and strict mode has no way to
    express one. Excluding it is not a workaround but the correct shape -- the model
    reports what a document says, and deciding which of those fields we do not model
    happens afterwards.
    """
    schema = model.model_json_schema()

    if exclude:
        _drop_properties(schema, exclude)

    return _make_strict(schema)


def response_format(
    model: type[BaseModel],
    *,
    exclude: frozenset[str] | None = None,
) -> dict[str, Any]:
    """The `response_format` payload for a chat completion call.

    `strict: true` is the difference between a schema the provider *suggests* to the
    model and one it *enforces*. Without it a response can come back shaped however
    the model felt like shaping it, and the whole typed-output guarantee is advisory.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": build_strict_schema(model, exclude=exclude),
        },
    }


def _drop_properties(schema: dict[str, Any], names: frozenset[str]) -> None:
    """Remove top-level properties, in place."""
    properties = schema.get("properties", {})
    for name in names:
        properties.pop(name, None)

    if "required" in schema:
        schema["required"] = [
            name for name in schema["required"] if name not in names
        ]


def _make_strict(node: Any) -> Any:
    """Walk the schema applying the three strict-mode rules.

    Recurses through `properties`, `items`, `anyOf`, `allOf` and `$defs`, so nested
    models (`LineItem` inside `Invoice`) get the same treatment as the root. `$ref`
    is left intact -- strict mode supports definitions, and inlining them would break
    on any model that referenced itself.
    """
    if isinstance(node, list):
        return [_make_strict(item) for item in node]

    if not isinstance(node, dict):
        return node

    result: dict[str, Any] = {
        key: _make_strict(value)
        for key, value in node.items()
        if key not in _UNSUPPORTED_KEYWORDS
    }

    if result.get("type") == "object" or "properties" in result:
        result["additionalProperties"] = False

        # Every property required. Optionality lives in the type (null is allowed),
        # not in whether the key is present -- which is what strict mode demands and
        # also what makes a response easier to reason about: the shape is always the
        # same, and absence is expressed as a value.
        properties = result.get("properties", {})
        if properties:
            result["required"] = list(properties)

    return result
