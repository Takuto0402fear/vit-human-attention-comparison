"""
Phase 3: Ridge-regression readout of Phase 2's low-level visual targets
from Phase 1's RAW (un-projected) L1..L12 pooled object features. Mirrors
lib/osie_text_alignment_probe.py's design (same image_id-grouped CV
philosophy, same "CLIP is frozen, only the tiny linear readout is fit"
philosophy) but for continuous regression targets instead of binary
attribute labels: StandardScaler + Ridge(alpha=1.0), plain GroupKFold
(not stratified -- there is no class balance to stratify for a continuous
target), out-of-fold R²/Pearson r/Spearman r/MAE.
"""
from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_LAYERS = tuple(range(1, 13))
DEFAULT_TARGETS = (
    "mean_luminance", "mean_red", "mean_green", "mean_blue", "mean_saturation",
    "luminance_contrast", "edge_strength", "fine_texture",
)
GEOMETRY_CONTROLS = ("mask_area_fraction", "centroid_x", "centroid_y")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def load_lowlevel_targets(lowlevel_targets_csv: str, target_names: Sequence[str]):
    with open(lowlevel_targets_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    object_keys, image_ids, target_values, valid_mask = [], [], {t: [] for t in target_names}, []
    for r in rows:
        object_keys.append(f"{r['image_id']}_{r['object_id']}")
        image_ids.append(r["image_id"])
        is_valid = r["valid"] == "True"
        valid_mask.append(is_valid)
        for t in target_names:
            target_values[t].append(float(r[t]) if is_valid and r[t] != "" else np.nan)
    return (object_keys, np.array(image_ids), {t: np.array(v) for t, v in target_values.items()},
            np.array(valid_mask))


def align_features_to_targets(feature_object_keys: Sequence[str], target_object_keys: Sequence[str]):
    """Returns an index array mapping target-order -> feature-order, raising
    if any target object_key is missing from the feature set."""
    pos = {k: i for i, k in enumerate(feature_object_keys)}
    idx = np.empty(len(target_object_keys), dtype=int)
    for i, k in enumerate(target_object_keys):
        if k not in pos:
            raise RuntimeError(f"STOP: object_key {k} present in low-level targets but missing from features")
        idx[i] = pos[k]
    return idx


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) < 2 or np.std(y_true) == 0:
        return {"r2": float("nan"), "pearson_r": float("nan"), "spearman_r": float("nan"), "mae": float("nan")}
    r2 = float(r2_score(y_true, y_pred))
    pear = float(pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else float("nan")
    spear = float(spearmanr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else float("nan")
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"r2": r2, "pearson_r": pear, "spearman_r": spear, "mae": mae}


def run_ridge_probe(
    object_features_npz: str,
    lowlevel_targets_csv: str,
    output_dir: str,
    layers: Sequence[int] = DEFAULT_LAYERS,
    targets: Sequence[str] = DEFAULT_TARGETS,
    alpha: float = 1.0,
    max_splits: int = 5,
    seed: int = 42,
    fold_scores_filename: str = "lowlevel_fold_scores.csv",
    summary_filename: str = "lowlevel_summary.csv",
    fold_assignments_filename: str = "fold_assignments.csv",
    collect_oof: bool = False,
    oof_filename: str = "lowlevel_oof_predictions.csv",
    verbose: bool = True,
):
    """
    Returns (fold_rows, summary_rows, oof_data_or_None). oof_data is
    {(target, layer): {"image_id":..., "object_key":..., "fold_id":...,
    "y_true":..., "y_pred":...}}.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_all_seeds(seed)

    npz = np.load(object_features_npz, allow_pickle=False)
    feature_object_keys = list(npz["object_keys"])
    raw_by_layer = {l: npz[f"raw_L{l}"].astype(np.float64) for l in layers}

    target_object_keys, image_ids_all, target_values, valid_mask = load_lowlevel_targets(
        lowlevel_targets_csv, targets)
    feat_idx = align_features_to_targets(feature_object_keys, target_object_keys)

    fold_rows, summary_rows = [], []
    oof_data = {} if collect_oof else None
    fold_assignment_rows = []

    for target in targets:
        y_full = target_values[target]
        valid = valid_mask & np.isfinite(y_full)
        idx = np.where(valid)[0]
        if len(idx) < 10:
            if verbose:
                print(f"  [{target}] INSUFFICIENT_DATA (n_valid={len(idx)})")
            continue

        y = y_full[idx]
        groups = image_ids_all[idx]
        n_groups = len(set(groups))
        n_splits = min(max_splits, n_groups)
        if n_splits < 2:
            continue

        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(np.zeros((len(idx), 1)), y, groups))
        for fold_i, (tr, te) in enumerate(splits):
            if set(groups[tr]) & set(groups[te]):
                raise RuntimeError(f"STOP: image_id leaked across train/test in fold {fold_i} for target={target}")
        for fold_i, (tr, te) in enumerate(splits):
            for k in np.array(target_object_keys)[idx][te]:
                fold_assignment_rows.append({"target": target, "object_key": k, "fold": fold_i})

        if verbose:
            print(f"  [{target}] n_valid={len(idx)} n_groups={n_groups} n_splits={n_splits}")

        for l_disp in layers:
            X_full = raw_by_layer[l_disp][feat_idx]
            X = X_full[idx]
            oof_image_ids, oof_object_keys, oof_fold_ids, oof_y_true, oof_y_pred = [], [], [], [], []

            for fold_i, (tr, te) in enumerate(splits):
                pipe = make_ridge_pipeline(alpha)
                pipe.fit(X[tr], y[tr])
                y_pred = pipe.predict(X[te])
                metrics = compute_metrics(y[te], y_pred)

                fold_rows.append({
                    "target": target, "layer": l_disp, "fold": fold_i, "n_train": len(tr), "n_test": len(te),
                    "r2": metrics["r2"], "pearson_r": metrics["pearson_r"],
                    "spearman_r": metrics["spearman_r"], "mae": metrics["mae"],
                })
                if collect_oof:
                    oof_image_ids.extend(groups[te].tolist())
                    oof_object_keys.extend(np.array(target_object_keys)[idx][te].tolist())
                    oof_fold_ids.extend([fold_i] * len(te))
                    oof_y_true.extend(y[te].tolist())
                    oof_y_pred.extend(y_pred.tolist())

            if collect_oof:
                oof_data[(target, l_disp)] = {
                    "image_id": np.array(oof_image_ids), "object_key": np.array(oof_object_keys),
                    "fold_id": np.array(oof_fold_ids), "y_true": np.array(oof_y_true),
                    "y_pred": np.array(oof_y_pred),
                }

            fold_metrics = [r for r in fold_rows if r["target"] == target and r["layer"] == l_disp]
            for metric_name in ("r2", "pearson_r", "spearman_r", "mae"):
                vals = np.array([r[metric_name] for r in fold_metrics if np.isfinite(r[metric_name])])
                summary_rows.append({
                    "target": target, "layer": l_disp, "metric": metric_name,
                    "n_folds": len(fold_metrics), "n_folds_valid": len(vals),
                    "mean": float(vals.mean()) if len(vals) else float("nan"),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n_valid_objects": len(idx),
                })

    fold_fields = ["target", "layer", "fold", "n_train", "n_test", "r2", "pearson_r", "spearman_r", "mae"]
    with open(os.path.join(output_dir, fold_scores_filename), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fold_fields)
        w.writeheader()
        for r in fold_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    summary_fields = ["target", "layer", "metric", "n_folds", "n_folds_valid", "mean", "std", "n_valid_objects"]
    with open(os.path.join(output_dir, summary_filename), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    with open(os.path.join(output_dir, fold_assignments_filename), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "object_key", "fold"])
        w.writeheader()
        w.writerows(fold_assignment_rows)

    if collect_oof:
        with open(os.path.join(output_dir, oof_filename), "w", newline="", encoding="utf-8") as fh:
            fieldnames = ["target", "layer", "object_key", "image_id", "fold", "y_true", "y_pred"]
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for (target, layer), d in oof_data.items():
                for key, img, fold, yt, yp in zip(d["object_key"], d["image_id"], d["fold_id"], d["y_true"], d["y_pred"]):
                    w.writerow({"target": target, "layer": layer, "object_key": key, "image_id": img,
                                "fold": int(fold), "y_true": f"{yt:.6f}", "y_pred": f"{yp:.6f}"})

    return fold_rows, summary_rows, oof_data


def image_level_bootstrap_r2_diff(oof_a: dict, oof_b: dict, seed: int = 42, n_boot: int = 10000, alpha: float = 0.05):
    """Paired-by-object, image-level bootstrap of R2(b) - R2(a) -- mirrors
    lib.osie_text_alignment_probe.image_level_bootstrap_auprc_diff exactly,
    for the regression (pooled out-of-fold R2) case."""
    assert np.array_equal(oof_a["image_id"], oof_b["image_id"])
    assert np.allclose(oof_a["y_true"], oof_b["y_true"])
    image_ids = oof_a["image_id"]
    y_true = oof_a["y_true"]
    unique_images = sorted(set(image_ids))
    n_img = len(unique_images)
    rows_by_image = {img: np.where(image_ids == img)[0] for img in unique_images}

    def r2_for_indices(idx, pred):
        yt = y_true[idx]
        if len(idx) < 2 or np.std(yt) == 0:
            return float("nan")
        return float(r2_score(yt, pred[idx]))

    observed_a = r2_for_indices(np.arange(len(y_true)), oof_a["y_pred"])
    observed_b = r2_for_indices(np.arange(len(y_true)), oof_b["y_pred"])

    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        sampled_images = rng.choice(unique_images, size=n_img, replace=True)
        idx = np.concatenate([rows_by_image[img] for img in sampled_images])
        a = r2_for_indices(idx, oof_a["y_pred"])
        b = r2_for_indices(idx, oof_b["y_pred"])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(b - a)
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"observed_a": observed_a, "observed_b": observed_b, "observed_diff": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n_boot_valid": 0, "p_value": float("nan")}
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    frac_le0 = float((diffs <= 0).mean())
    frac_ge0 = float((diffs >= 0).mean())
    p_value = float(min(1.0, 2 * min(frac_le0, frac_ge0)))
    return {
        "observed_a": observed_a, "observed_b": observed_b, "observed_diff": observed_b - observed_a,
        "ci95_lo": float(lo), "ci95_hi": float(hi), "n_boot_valid": len(diffs), "p_value": p_value,
    }
