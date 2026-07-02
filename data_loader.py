"""
Adapter layer: data source -> List[GazeRecord].

Implementations
---------------
- "dummy"   : synthetic centre-biased fixations (for pipeline testing)
- "healthy" : stub (raises NotImplementedError until real data arrives)
"""
from __future__ import annotations

import warnings
from typing import List

import numpy as np

from config import Config
from schema import Fixation, GazeRecord, validate_record


# ======================================================================
# Public API
# ======================================================================

def load_gaze(source: str, config: Config) -> List[GazeRecord]:
    """
    Load gaze data from *source* and return a list of validated GazeRecords.

    Parameters
    ----------
    source : str
        "dummy"   - generate synthetic data (both groups).
        "healthy" - placeholder for real healthy-subject data.
    config : Config
        Pipeline configuration.
    """
    loaders = {
        "dummy": _generate_dummy,
        "healthy": _load_healthy_stub,
    }
    if source not in loaders:
        raise ValueError(f"Unknown source '{source}'. Available: {list(loaders)}")
    records = loaders[source](config)
    for r in records:
        validate_record(r)
    return records


# ======================================================================
# Dummy generator
# ======================================================================

def _generate_dummy(config: Config) -> List[GazeRecord]:
    """Centre-biased random fixations for two groups."""
    rng = np.random.RandomState(config.seed)
    w, h = config.image_size
    records: List[GazeRecord] = []

    image_ids = [f"img_{i:04d}" for i in range(config.dummy_n_images)]

    for group in config.groups:
        for subj_idx in range(config.dummy_n_subjects_per_group):
            sid = f"{group[0].upper()}{subj_idx + 1:03d}"  # H001, S001 ...
            for img_id in image_ids:
                n_fix = rng.randint(*config.dummy_fixations_range)
                fixations = []
                for _ in range(n_fix):
                    # centre-biased Gaussian, clipped to image bounds
                    fx = np.clip(rng.normal(w / 2, w / 6), 0, w - 1)
                    fy = np.clip(rng.normal(h / 2, h / 6), 0, h - 1)
                    dur = float(rng.exponential(200))  # ms
                    fixations.append(Fixation(x=float(fx), y=float(fy), duration=dur))

                records.append(GazeRecord(
                    subject_id=sid,
                    group=group,
                    image_id=img_id,
                    fixations=fixations,
                    coord_system="image_topleft_pixel",
                    image_size=(w, h),
                ))
    return records


# ======================================================================
# Healthy-subject loader (stub)
# ======================================================================

def _load_healthy_stub(config: Config) -> List[GazeRecord]:
    """Stub: replace when real data format is determined."""
    raise NotImplementedError(
        "Real healthy-subject loader is not yet implemented. "
        "Waiting for data format from Yoshida-sensei. "
        "Use source='dummy' for pipeline testing."
    )
