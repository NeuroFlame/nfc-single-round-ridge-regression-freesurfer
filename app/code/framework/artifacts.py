from dataclasses import dataclass
from typing import Optional


@dataclass
class ArtifactRef:
    path: str
    kind: str = "file"
    media_type: Optional[str] = None
