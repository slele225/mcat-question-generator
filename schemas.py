"""
JSON schema definitions for MCAT question generation.
Imported directly from prompt_templates.py to maintain single source of truth.
"""

from prompt_templates import (
    SCIENCE_ITEM_SCHEMA,
    SCIENCE_BATCH_SCHEMA,
    CARS_QUESTION_SCHEMA,
    CARS_SET_SCHEMA,
    SCIENCE_VALIDATION_SCHEMA,
    CARS_VALIDATION_SCHEMA,
    SCIENCE_REPAIR_OUTPUT_SCHEMA,
    CARS_REPAIR_OUTPUT_SCHEMA,
    schema_to_pretty_json,
)

# Re-export for convenience
__all__ = [
    "SCIENCE_ITEM_SCHEMA",
    "SCIENCE_BATCH_SCHEMA",
    "CARS_QUESTION_SCHEMA",
    "CARS_SET_SCHEMA",
    "SCIENCE_VALIDATION_SCHEMA",
    "CARS_VALIDATION_SCHEMA",
    "SCIENCE_REPAIR_OUTPUT_SCHEMA",
    "CARS_REPAIR_OUTPUT_SCHEMA",
    "schema_to_pretty_json",
]


def validate_against_schema(obj, schema, path=""):
    """
    Simple schema validator.
    
    Args:
        obj: The object to validate
        schema: The schema dict (keys are required fields, values are type hints)
        path: Current path for error messages
    
    Returns:
        (is_valid, errors) tuple
    """
    errors = []
    
    if not isinstance(schema, dict):
        # Leaf node - check type
        if schema == "string" and not isinstance(obj, str):
            errors.append(f"{path}: expected string, got {type(obj).__name__}")
        elif schema == "number" and not isinstance(obj, (int, float)):
            errors.append(f"{path}: expected number, got {type(obj).__name__}")
        elif schema == "integer" and not isinstance(obj, int):
            errors.append(f"{path}: expected integer, got {type(obj).__name__}")
        elif schema == "boolean" and not isinstance(obj, bool):
            errors.append(f"{path}: expected boolean, got {type(obj).__name__}")
        elif isinstance(schema, str) and "|" in schema:
            # Enum-like: "A|B|C"
            allowed = [s.strip() for s in schema.split("|")]
            if obj not in allowed:
                errors.append(f"{path}: expected one of {allowed}, got {obj}")
        elif isinstance(schema, list):
            # List type
            if not isinstance(obj, list):
                errors.append(f"{path}: expected list, got {type(obj).__name__}")
            else:
                for i, item in enumerate(obj):
                    sub_errors = validate_against_schema(item, schema[0], f"{path}[{i}]")
                    errors.extend(sub_errors)
        return len(errors) == 0, errors
    
    # Dict node - check all keys
    if not isinstance(obj, dict):
        errors.append(f"{path}: expected dict, got {type(obj).__name__}")
        return False, errors
    
    # Check required keys
    for key, key_schema in schema.items():
        if key not in obj:
            errors.append(f"{path}: missing required key '{key}'")
        else:
            sub_errors = validate_against_schema(obj[key], key_schema, f"{path}.{key}")
            errors.extend(sub_errors)
    
    return len(errors) == 0, errors