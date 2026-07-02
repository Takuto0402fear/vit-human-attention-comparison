"""
Common data format (adapter layer output schema).

Every data loader must produce List[GazeRecord].  Downstream modules depend
*only* on this schema, so swapping data sources requires only a new loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


@dataclass
class Fixation:
    """A single fixation point."""
    x: float                        # horizontal coordinate
    y: float                        # vertical coordinate
    duration: Optional[float] = None  # milliseconds (optional)


@dataclass
class GazeRecord:
    """One subject's fixation data for one image."""
    subject_id: str                 # e.g. "S001"
    group: str                      # "healthy" | "schizophrenia"
    image_id: str                   # must match the image fed to the ViT
    fixations: List[Fixation]       # fixation sequence on this image
    coord_system: str               # e.g. "image_topleft_pixel"
    image_size: Tuple[int, int]     # (width, height) in pixels

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_VALID_GROUPS = {"healthy", "schizophrenia"}
_VALID_COORD_SYSTEMS = {"image_topleft_pixel", "normalized"}


def validate_record(rec: GazeRecord) -> None:
    """Raise ValueError if the record violates the schema."""
    if rec.group not in _VALID_GROUPS:
        raise ValueError(f"Invalid group '{rec.group}'; expected one of {_VALID_GROUPS}")
    if rec.coord_system not in _VALID_COORD_SYSTEMS:
        raise ValueError(
            f"Invalid coord_system '{rec.coord_system}'; "
            f"expected one of {_VALID_COORD_SYSTEMS}"
        )
    if not rec.fixations:
        raise ValueError(f"Record {rec.subject_id}/{rec.image_id} has no fixations")
    w, h = rec.image_size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image_size {rec.image_size}")

    if rec.coord_system == "image_topleft_pixel":
        for f in rec.fixations:
            if not (0 <= f.x < w and 0 <= f.y < h):
                raise ValueError(
                    f"Fixation ({f.x}, {f.y}) out of bounds for "
                    f"image_size ({w}, {h})"
                )
    elif rec.coord_system == "normalized":
        for f in rec.fixations:
            if not (0.0 <= f.x <= 1.0 and 0.0 <= f.y <= 1.0):
                raise ValueError(
                    f"Normalized fixation ({f.x}, {f.y}) out of [0,1] range"
                )
