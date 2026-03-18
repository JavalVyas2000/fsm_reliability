from __future__ import annotations

import ast
import json
import re
from typing import Optional

from pydantic import BaseModel, ValidationError, field_validator


JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
LIST_PATTERN = re.compile(r"\[[^\[\]]*\]")


class PathResponse(BaseModel):
    path: list[int]

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value):
        """
        Accept both:
        - [3, 4]
        - ["3", "4"]

        and convert numeric strings to integers.
        """
        if not isinstance(value, list):
            raise ValueError("Path must be a list.")

        cleaned: list[int] = []
        for item in value:
            if isinstance(item, bool):
                raise ValueError("Boolean values are not allowed in path.")

            if isinstance(item, int):
                cleaned.append(item)
                continue

            if isinstance(item, str):
                stripped = item.strip()
                if stripped.lstrip("-").isdigit():
                    cleaned.append(int(stripped))
                    continue

            raise ValueError("Path must contain integers or numeric strings only.")

        return cleaned


class ParsedPathResult(BaseModel):
    path: Optional[list[int]]
    parse_success: int
    parse_mode: str  # "json", "list", "none"


def _validate_int_list(obj) -> Optional[list[int]]:
    """
    Accept both raw ints and numeric strings, and coerce to int.
    """
    if not isinstance(obj, list):
        return None

    cleaned: list[int] = []
    for item in obj:
        if isinstance(item, bool):
            return None

        if isinstance(item, int):
            cleaned.append(item)
            continue

        if isinstance(item, str):
            stripped = item.strip()
            if stripped.lstrip("-").isdigit():
                cleaned.append(int(stripped))
                continue

        return None

    return cleaned


def extract_first_json_object(text: str) -> Optional[str]:
    if text is None:
        return None

    text = text.strip()
    if not text:
        return None

    match = JSON_OBJECT_PATTERN.search(text)
    if not match:
        return None

    return match.group(0)


def extract_first_list(text: str) -> Optional[str]:
    if text is None:
        return None

    text = text.strip()
    if not text:
        return None

    match = LIST_PATTERN.search(text)
    if not match:
        return None

    return match.group(0)


def parse_path_from_text(text: str) -> ParsedPathResult:
    """
    Try:
    1. strict JSON: {"path": [0, 2, 5]} or {"path": ["0", "2", "5"]}
    2. fallback plain list: [0, 2, 5] or ["0", "2", "5"]
    """
    json_str = extract_first_json_object(text)
    if json_str is not None:
        try:
            payload = json.loads(json_str)
            parsed = PathResponse.model_validate(payload)
            return ParsedPathResult(
                path=parsed.path,
                parse_success=1,
                parse_mode="json",
            )
        except (json.JSONDecodeError, ValidationError):
            pass

    list_str = extract_first_list(text)
    if list_str is not None:
        try:
            parsed = ast.literal_eval(list_str)
            validated = _validate_int_list(parsed)
            if validated is not None:
                return ParsedPathResult(
                    path=validated,
                    parse_success=1,
                    parse_mode="list",
                )
        except (SyntaxError, ValueError):
            pass

    return ParsedPathResult(
        path=None,
        parse_success=0,
        parse_mode="none",
    )