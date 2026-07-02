"""
Aggregate per-trial metric scores into a group x layer summary table
and export to CSV.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from config import Config
from schema import GazeRecord
from metrics import nss, auc_judd, sauc


# ======================================================================
# Core: compute all metrics for every (subject, image, layer)
# ======================================================================

def compute_all_metrics(
    records: List[GazeRecord],
    attention_maps: Dict[str, np.ndarray],
    config: Config,
) -> List[dict]:
    """
    Compute NSS, AUC-Judd, sAUC for each (subject, image, layer).

    Parameters
    ----------
    records : list of GazeRecord
    attention_maps : {image_id: ndarray (layers, H, W)}
    config : Config

    Returns
    -------
    list of dicts, each with keys:
        subject_id, group, image_id, layer, n_fixations,
        nss, auc_judd, sauc
    """
    # filter to images that have both gaze data and attention maps
    available_images = set(attention_maps.keys())
    records = [r for r in records if r.image_id in available_images]

    if config.common_images_only:
        records, common_imgs = _filter_common_images(records, config)
        print(f"[aggregation] common_images_only: {len(common_imgs)} images retained")

    # pre-collect fixation points per image (for sAUC other-image negatives)
    img_fixations: Dict[str, List[np.ndarray]] = defaultdict(list)
    for rec in records:
        pts = np.array([[f.x, f.y] for f in rec.fixations])
        img_fixations[rec.image_id].append(pts)

    # pool other-image fixations for sAUC
    all_image_ids = sorted(set(r.image_id for r in records))

    results: List[dict] = []
    for rec in records:
        attn = attention_maps[rec.image_id]  # (L, H, W)
        n_layers = attn.shape[0]
        pts = np.array([[f.x, f.y] for f in rec.fixations])

        # collect other-image fixations for sAUC
        other_pts_list = []
        for oid in all_image_ids:
            if oid == rec.image_id:
                continue
            other_pts_list.extend(img_fixations[oid])
        other_pts = (np.concatenate(other_pts_list, axis=0)
                     if other_pts_list else np.empty((0, 2)))

        for layer in range(n_layers):
            sal = attn[layer]  # (H, W)
            row = {
                "subject_id": rec.subject_id,
                "group": rec.group,
                "image_id": rec.image_id,
                "layer": layer + 1,  # 1-indexed
                "n_fixations": len(rec.fixations),
                "nss": nss(sal, pts),
                "auc_judd": auc_judd(sal, pts),
                "sauc": sauc(sal, pts, other_pts),
            }
            results.append(row)
    return results


# ======================================================================
# Group x Layer summary table
# ======================================================================

def summarise_group_layer(
    results: List[dict],
) -> List[dict]:
    """
    Aggregate results into group x layer means and SDs.

    Returns
    -------
    list of dicts with keys:
        group, layer, n_subjects, n_observations, mean_n_fixations,
        nss_mean, nss_sd, auc_judd_mean, auc_judd_sd,
        sauc_mean, sauc_sd
    """
    # bucket by (group, layer)
    buckets: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for row in results:
        buckets[(row["group"], row["layer"])].append(row)

    summary = []
    for (group, layer), rows in sorted(buckets.items()):
        nss_vals = [r["nss"] for r in rows if np.isfinite(r["nss"])]
        auc_vals = [r["auc_judd"] for r in rows if np.isfinite(r["auc_judd"])]
        sauc_vals = [r["sauc"] for r in rows if np.isfinite(r["sauc"])]
        n_fix_vals = [r["n_fixations"] for r in rows]
        subjects = set(r["subject_id"] for r in rows)
        summary.append({
            "group": group,
            "layer": layer,
            "n_subjects": len(subjects),
            "n_observations": len(rows),
            "mean_n_fixations": float(np.mean(n_fix_vals)) if n_fix_vals else 0,
            "nss_mean": float(np.mean(nss_vals)) if nss_vals else np.nan,
            "nss_sd": float(np.std(nss_vals, ddof=1)) if len(nss_vals) > 1 else np.nan,
            "auc_judd_mean": float(np.mean(auc_vals)) if auc_vals else np.nan,
            "auc_judd_sd": float(np.std(auc_vals, ddof=1)) if len(auc_vals) > 1 else np.nan,
            "sauc_mean": float(np.mean(sauc_vals)) if sauc_vals else np.nan,
            "sauc_sd": float(np.std(sauc_vals, ddof=1)) if len(sauc_vals) > 1 else np.nan,
        })
    return summary


def save_csv(summary: List[dict], path: str) -> None:
    """Write the summary table to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(summary[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[aggregation] Saved {path}")


def print_table(summary: List[dict]) -> None:
    """Pretty-print the summary table to stdout."""
    header = f"{'group':<16} {'layer':>5}  {'N_subj':>6}  {'N_obs':>5}  " \
             f"{'fix':>5}  {'NSS':>7}  {'AUC':>7}  {'sAUC':>7}"
    print("\n" + header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['group']:<16} {row['layer']:>5}  "
            f"{row['n_subjects']:>6}  {row['n_observations']:>5}  "
            f"{row['mean_n_fixations']:>5.1f}  "
            f"{row['nss_mean']:>7.3f}  "
            f"{row['auc_judd_mean']:>7.3f}  "
            f"{row['sauc_mean']:>7.3f}"
        )
    print()


# ======================================================================
# Internal helpers
# ======================================================================

def _filter_common_images(
    records: List[GazeRecord], config: Config
) -> Tuple[List[GazeRecord], set]:
    """Keep only images viewed by ALL groups."""
    from collections import defaultdict
    group_images: Dict[str, set] = defaultdict(set)
    for r in records:
        group_images[r.group].add(r.image_id)

    if not group_images:
        return records, set()

    common = set.intersection(*group_images.values())
    filtered = [r for r in records if r.image_id in common]
    return filtered, common
