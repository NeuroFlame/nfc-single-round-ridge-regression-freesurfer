"""Define future-facing references to externally transferred artifacts."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ArtifactRef:
    """Describe a file-like artifact without embedding its content."""

    path: str
    kind: str = "file"
    media_type: Optional[str] = None
