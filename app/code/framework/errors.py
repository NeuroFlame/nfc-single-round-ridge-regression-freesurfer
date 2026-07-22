import json
import os
import traceback
from typing import Any, Dict, List


TERMINAL_ERROR_FILE_NAME = ".neuroflame_error.json"


def clear_terminal_error(output_dir: str) -> None:
    error_path = os.path.join(output_dir, TERMINAL_ERROR_FILE_NAME)
    try:
        os.remove(error_path)
    except FileNotFoundError:
        pass


def record_terminal_error(output_dir: str, scope: str, error: Exception) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        error_path = os.path.join(output_dir, TERMINAL_ERROR_FILE_NAME)
        with open(error_path, "w", encoding="utf-8") as error_file:
            json.dump(
                {
                    "scope": scope,
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
