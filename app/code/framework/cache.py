import os
import shutil
from typing import Any, Dict, Optional

import json

from .serialization import DEFAULT_MAX_INLINE_ARRAY_BYTES, deserialize_value, serialize_value


class JsonStateStore:
    def __init__(
        self,
        base_dir: str,
        codecs: Dict[type, Any] = None,
        max_inline_array_bytes: int = DEFAULT_MAX_INLINE_ARRAY_BYTES,
    ):
        self._state_dir = os.path.join(base_dir, "_temp_state")
        self._state_file_path = os.path.join(self._state_dir, "local_state.json")
        self._codecs = codecs or {}
        self._max_inline_array_bytes = max_inline_array_bytes
        os.makedirs(self._state_dir, exist_ok=True)
        self._stored_state: Optional[Any] = None

        if os.path.exists(self._state_file_path):
            with open(self._state_file_path, "r", encoding="utf-8") as state_file:
                self._stored_state = json.load(state_file)

    def load_state(self, state_type: Optional[type] = None) -> Optional[Any]:
        if self._stored_state is None:
            return None
        return deserialize_value(
            self._stored_state,
            state_type,
            self._codecs,
            max_inline_array_bytes=self._max_inline_array_bytes,
        )

    def save_state(self, state: Any) -> None:
        self._stored_state = serialize_value(
            state,
            self._codecs,
            max_inline_array_bytes=self._max_inline_array_bytes,
        )
        with open(self._state_file_path, "w", encoding="utf-8") as state_file:
            json.dump(self._stored_state, state_file)

    def remove_state(self) -> None:
        self._stored_state = None
        shutil.rmtree(self._state_dir, ignore_errors=True)

JsonCacheStore = JsonStateStore
