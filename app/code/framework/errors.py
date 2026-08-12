"""Record terminal computation failures for container-level reporting."""

import json
import os
import traceback
from typing import Any, Dict, List

TERMINAL_ERROR_FILE_NAME = ".neuroflame_error.json"
ERROR_ENVELOPE_KEY = "__neuroflame_error__"
ERROR_SCHEMA_VERSION = 1
ERROR_ORIGIN_SITE = "site"
ERROR_ORIGIN_CENTRAL = "central"
_VALID_ERROR_ORIGINS = {ERROR_ORIGIN_SITE, ERROR_ORIGIN_CENTRAL}


def clear_terminal_error(output_dir: str) -> None:
    """Remove a terminal-error marker left by an earlier run."""
    error_path = os.path.join(output_dir, TERMINAL_ERROR_FILE_NAME)
    try:
        os.remove(error_path)
    except FileNotFoundError:
        pass


def build_error_envelope(origin: str, stage: str, scope: str) -> Dict[str, Any]:
    """Build safe failure provenance for transport through an NVFlare Shareable."""
    if origin not in _VALID_ERROR_ORIGINS:
        raise ValueError(f"Invalid error origin: {origin!r}")
    return {
        "schema_version": ERROR_SCHEMA_VERSION,
        "origin": origin,
        "stage": stage,
        "scope": scope,
    }


def parse_error_envelope(value: Any) -> Dict[str, Any] | None:
    """Validate a transported NeuroFLAME error envelope."""
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") != ERROR_SCHEMA_VERSION:
        return None
    if value.get("origin") not in _VALID_ERROR_ORIGINS:
        return None
    if not isinstance(value.get("stage"), str) or not value["stage"]:
        return None
    if not isinstance(value.get("scope"), str) or not value["scope"]:
        return None
    return {
        "schema_version": ERROR_SCHEMA_VERSION,
        "origin": value["origin"],
        "stage": value["stage"],
        "scope": value["scope"],
    }


def record_terminal_error(
    output_dir: str,
    scope: str,
    error: Exception,
    *,
    origin: str,
    stage: str,
) -> None:
    """Persist an exception and traceback without replacing the original error."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        error_path = os.path.join(output_dir, TERMINAL_ERROR_FILE_NAME)
        with open(error_path, "w", encoding="utf-8") as error_file:
            json.dump(
                {
                    **build_error_envelope(origin, stage, scope),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                error_file,
                indent=2,
            )
    except Exception:
        # Error reporting must not replace the computation exception.
        pass


def find_terminal_errors(root_dir: str) -> List[Dict[str, Any]]:
    """Find and parse every terminal-error marker below a directory."""
    errors = []
    if not os.path.isdir(root_dir):
        return errors

    for directory, _subdirectories, file_names in os.walk(root_dir):
        if TERMINAL_ERROR_FILE_NAME not in file_names:
            continue
        error_path = os.path.join(directory, TERMINAL_ERROR_FILE_NAME)
        try:
            with open(error_path, encoding="utf-8") as error_file:
                error = json.load(error_file)
            if not isinstance(error, dict):
                raise TypeError("Terminal error marker must contain a JSON object")
        except Exception as read_error:
            error = {
                "scope": os.path.relpath(directory, root_dir),
                "error_type": type(read_error).__name__,
                "message": f"Could not read terminal error marker: {read_error}",
                "traceback": "",
            }
        error["path"] = error_path
        errors.append(error)

    return sorted(errors, key=lambda error: error["path"])


def raise_for_terminal_errors(root_dir: str) -> None:
    """Raise one combined error when terminal markers exist below a directory."""
    errors = find_terminal_errors(root_dir)
    if not errors:
        return

    details = []
    for error in errors:
        summary = (
            f"[{error.get('scope', 'computation')}] "
            f"{error.get('error_type', 'Error')}: {error.get('message', '')}"
        )
        error_traceback = error.get("traceback")
        details.append(f"{summary}\n{error_traceback}" if error_traceback else summary)
    raise RuntimeError("Terminal computation failure:\n" + "\n".join(details))
