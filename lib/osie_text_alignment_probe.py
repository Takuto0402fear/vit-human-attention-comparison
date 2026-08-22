"""
Shared linear-probe implementation (Implementation B) for the OSIE
text-alignment pilot, used IDENTICALLY by both the 50-image pilot
(scripts/probe_osie_text_alignment.py) and the 700-image extension
(scripts/probe_osie_full700.py) -- same StandardScaler+LogisticRegression
pipeline, same StratifiedGroupKFold-by-image_id logic, same permutation
baseline, same random_state=42 throughout. Only the input object-feature
cache and output directory differ between the two callers.

CLIP itself is never touched here -- this module only ever sees already-
extracted, cached RAW (un-projected) pooled hidden-state features.
"""
from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lib.osie_text_alignment import safe_n_splits

DEFAULT_LAYERS = (4, 8, 12)
DEFAULT_ATTRIBUTES = (
    "Text", "Face", "Emotion", "Sound", "Smell", "Taste", "Touch",
    "Motion", "Operability", "Watchability", "Touched", "Gazed",
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                                    max_iter=5000, random_state=seed)),
    ])


def load_object_features(object_features_npz: str, objects_metadata_csv: str, layers: Sequence[int]):
    npz = np.load(object_features_npz, allow_pickle=False)
    object_keys = list(npz["object_keys"])
    raw_by_layer = {l: npz[f"raw_L{l}"].astype(np.float64) for l in layers}

    with open(objects_metadata_csv, encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    if len(meta_rows) != len(object_keys):
        raise RuntimeError(
            f"STOP: {objects_metadata_csv} has {len(meta_rows)} rows, "
            f"{object_features_npz} has {len(object_keys)} objects")
    for row, key in zip(meta_rows, object_keys):
        expected_key = f"{row['image_id']}_{row['object_id']}"
        if expected_key != key:
            raise RuntimeError(f"STOP: object key mismatch: metadata={expected_key} features={key}")

    image_ids = np.array([r["image_id"] for r in meta_rows])
    pos_sets = [set(r["positive_attributes"].split(";")) if r["positive_attributes"] else set()
                for r in meta_rows]
    return object_keys, raw_by_layer, image_ids, pos_sets


def build_conditions(pos_sets: List[set], attributes: Sequence[str]) -> Dict[tuple, np.ndarray]:
    n = len(pos_sets)
    conditions = {}
    for attr in attributes:
        conditions[(attr, "all_objects")] = np.ones(n, dtype=bool)
    if "Face" in attributes and "Emotion" in attributes:
        face_mask = np.array(["Face" in s for s in pos_sets])
        conditions[("Emotion", "conditional_on_face")] = face_mask
    return conditions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    if len(set(y_true.tolist())) < 2:
        return {"auprc": float("nan"), "auroc": float("nan"), "balanced_accuracy": float("nan")}
    return {
        "auprc": float(average_precision_score(y_true, y_proba)),
        "auroc": float(roc_auc_score(y_true, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def run_linear_probe(
    object_features_npz: str,
    objects_metadata_csv: str,
    output_dir: str,
    layers: Sequence[int] = DEFAULT_LAYERS,
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    max_splits: int = 5,
    n_permutation_reps: int = 20,
    seed: int = 42,
    fold_scores_filename: str = "probe_fold_scores.csv",
    summary_filename: str = "probe_summary.csv",
    permutation_filename: str = "probe_permutation_baseline.csv",
    collect_oof: bool = False,
    verbose: bool = True,
):
    """
    Runs the full Implementation-B linear-probe pipeline (identical logic
    regardless of caller/scale) and writes the 3 standard CSVs to
    `output_dir`. If collect_oof=True, also returns a dict
    {(attribute, condition, layer): {"image_id": array, "y_true": array,
    "y_proba": array}} of pooled OUT-OF-FOLD predictions (one prediction
    per object, from the fold where it was held out) for downstream
    layer-difference bootstrap analysis -- each object appears exactly
    once, so this is safe to treat as one paired dataset across layers.

    Returns (fold_rows, summary_rows, perm_rows, insufficient, oof_data_or_None).
    """
    os.makedirs(output_dir, exist_ok=True)
    set_all_seeds(seed)
    object_keys, raw_by_layer, image_ids, pos_sets = load_object_features(
        object_features_npz, objects_metadata_csv, layers)
    if verbose:
        print(f"  Loaded {len(object_keys)} objects")

    conditions = build_conditions(pos_sets, attributes)
    fold_rows, summary_rows, perm_rows, insufficient = [], [], [], []
    oof_data = {} if collect_oof else None

    for (attr, condition), obj_mask in conditions.items():
        idx = np.where(obj_mask)[0]
        y = np.array([1 if attr in pos_sets[i] else 0 for i in idx])
        groups = image_ids[idx]

        pos_images = set(groups[y == 1])
        neg_images = set(groups[y == 0])
        n_pos_img, n_neg_img = len(pos_images), len(neg_images)
        n_total_img = len(set(groups))

        n_splits = safe_n_splits(n_pos_img, n_neg_img, max_splits=max_splits)
        if n_splits is not None:
            n_splits = min(n_splits, n_total_img)
        if n_splits is None or n_splits < 2:
            insufficient.append({
                "attribute": attr, "condition": condition, "n_objects": len(idx),
                "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
                "n_positive_images": n_pos_img, "n_negative_images": n_neg_img,
                "reason": "insufficient_data",
            })
            if verbose:
                print(f"  [{attr}/{condition}] INSUFFICIENT_DATA "
                      f"(n_pos_img={n_pos_img}, n_neg_img={n_neg_img})")
            continue

        if verbose:
            print(f"  [{attr}/{condition}] n_objects={len(idx)} n_pos={int(y.sum())} "
                  f"n_neg={int((1-y).sum())} n_pos_img={n_pos_img} n_neg_img={n_neg_img} "
                  f"n_splits={n_splits}")

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(sgkf.split(np.zeros((len(idx), 1)), y, groups))
        for fold_i, (tr, te) in enumerate(splits):
            if set(groups[tr]) & set(groups[te]):
                raise RuntimeError(f"STOP: image_id leaked across train/test in fold {fold_i} for {attr}/{condition}")

        random_auprc = float(y.mean())

        object_keys_arr = np.array(object_keys)
        for l_disp in layers:
            X = raw_by_layer[l_disp][idx]
            oof_image_ids, oof_object_keys, oof_fold_ids, oof_y_true, oof_y_proba = [], [], [], [], []

            for fold_i, (tr, te) in enumerate(splits):
                pipe = make_pipeline(seed)
                pipe.fit(X[tr], y[tr])
                y_pred = pipe.predict(X[te])
                y_proba = pipe.predict_proba(X[te])[:, 1]
                metrics = compute_metrics(y[te], y_pred, y_proba)

                fold_rows.append({
                    "attribute": attr, "condition": condition, "layer": l_disp, "fold": fold_i,
                    "n_objects": len(idx), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
                    "n_positive_images": n_pos_img, "n_negative_images": n_neg_img,
                    "auprc": metrics["auprc"], "auroc": metrics["auroc"],
                    "balanced_accuracy": metrics["balanced_accuracy"], "random_auprc": random_auprc,
                })
                if collect_oof:
                    oof_image_ids.extend(groups[te].tolist())
                    oof_object_keys.extend(object_keys_arr[idx][te].tolist())
                    oof_fold_ids.extend([fold_i] * len(te))
                    oof_y_true.extend(y[te].tolist())
                    oof_y_proba.extend(y_proba.tolist())

            if collect_oof:
                oof_data[(attr, condition, l_disp)] = {
                    "image_id": np.array(oof_image_ids), "object_key": np.array(oof_object_keys),
                    "fold_id": np.array(oof_fold_ids), "y_true": np.array(oof_y_true),
                    "y_proba": np.array(oof_y_proba), "random_auprc": random_auprc,
                }

            fold_metrics = [r for r in fold_rows if r["attribute"] == attr
                            and r["condition"] == condition and r["layer"] == l_disp]
            for metric_name in ("auprc", "auroc", "balanced_accuracy"):
                vals = np.array([r[metric_name] for r in fold_metrics if np.isfinite(r[metric_name])])
                summary_rows.append({
                    "attribute": attr, "condition": condition, "layer": l_disp,
                    "metric": metric_name, "n_folds": len(fold_metrics), "n_folds_valid": len(vals),
                    "mean": float(vals.mean()) if len(vals) else float("nan"),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n_objects": len(idx), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
                    "n_positive_images": n_pos_img, "n_negative_images": n_neg_img,
                    "random_auprc": random_auprc,
                })

            # ---- permutation baseline (shuffle TRAIN labels only, per fold) ----
            perm_rng = np.random.RandomState(seed)
            perm_metrics = {"auprc": [], "auroc": [], "balanced_accuracy": []}
            for rep in range(n_permutation_reps):
                for fold_i, (tr, te) in enumerate(splits):
                    y_train_shuffled = perm_rng.permutation(y[tr])
                    pipe = make_pipeline(seed)
                    pipe.fit(X[tr], y_train_shuffled)
                    y_pred = pipe.predict(X[te])
                    y_proba = pipe.predict_proba(X[te])[:, 1]
                    metrics = compute_metrics(y[te], y_pred, y_proba)
                    for m in perm_metrics:
                        if np.isfinite(metrics[m]):
                            perm_metrics[m].append(metrics[m])

            for metric_name, vals in perm_metrics.items():
                arr = np.array(vals)
                perm_rows.append({
                    "attribute": attr, "condition": condition, "layer": l_disp,
                    "metric": metric_name, "n_repetitions": n_permutation_reps, "n_values": len(arr),
                    "mean": float(arr.mean()) if len(arr) else float("nan"),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "random_auprc": random_auprc,
                })

    fold_fields = ["attribute", "condition", "layer", "fold", "n_objects", "n_positive",
                   "n_negative", "n_positive_images", "n_negative_images", "auprc", "auroc",
                   "balanced_accuracy", "random_auprc"]
    with open(os.path.join(output_dir, fold_scores_filename), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fold_fields)
        w.writeheader()
        for r in fold_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    summary_fields = ["attribute", "condition", "layer", "metric", "n_folds", "n_folds_valid",
                      "mean", "std", "n_objects", "n_positive", "n_negative",
                      "n_positive_images", "n_negative_images", "random_auprc"]
    all_summary_rows = list(summary_rows)
    for row in insufficient:
        all_summary_rows.append({**{f: "" for f in summary_fields}, **row,
                                  "metric": "insufficient_data", "mean": "", "std": ""})
    with open(os.path.join(output_dir, summary_filename), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=summary_fields)
        w.writeheader()
        for r in all_summary_rows:
            row = {k: r.get(k, "") for k in summary_fields}
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})

    perm_fields = ["attribute", "condition", "layer", "metric", "n_repetitions", "n_values",
                   "mean", "std", "random_auprc"]
    with open(os.path.join(output_dir, permutation_filename), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=perm_fields)
        w.writeheader()
        for r in perm_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    return fold_rows, all_summary_rows, perm_rows, insufficient, oof_data


def image_level_bootstrap_auprc_diff(oof_a: dict, oof_b: dict, seed: int = 42, n_boot: int = 10000, alpha: float = 0.05):
    """
    Paired-by-image bootstrap of AUPRC(b) - AUPRC(a), where oof_a/oof_b are
    two entries from run_linear_probe's oof_data dict for the SAME
    (attribute, condition) but different layers -- each object appears in
    exactly one fold's held-out set, so both dicts share the identical
    object population; only the model (fit on that layer's features) and
    its resulting y_proba differ.
    """
    assert np.array_equal(oof_a["image_id"], oof_b["image_id"]), "STOP: OOF object order/image_ids must match across layers"
    assert np.array_equal(oof_a["y_true"], oof_b["y_true"]), "STOP: OOF true labels must match across layers"
    image_ids = oof_a["image_id"]
    y_true = oof_a["y_true"]
    unique_images = sorted(set(image_ids))
    n_img = len(unique_images)
    rows_by_image = {img: np.where(image_ids == img)[0] for img in unique_images}

    def auprc_for_indices(idx, proba):
        yt = y_true[idx]
        if len(set(yt.tolist())) < 2:
            return float("nan")
        return float(average_precision_score(yt, proba[idx]))

    observed_a = auprc_for_indices(np.arange(len(y_true)), oof_a["y_proba"])
    observed_b = auprc_for_indices(np.arange(len(y_true)), oof_b["y_proba"])

    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        sampled_images = rng.choice(unique_images, size=n_img, replace=True)
        idx = np.concatenate([rows_by_image[img] for img in sampled_images])
        a = auprc_for_indices(idx, oof_a["y_proba"])
        b = auprc_for_indices(idx, oof_b["y_proba"])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(b - a)
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"observed_a": observed_a, "observed_b": observed_b, "observed_diff": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n_boot_valid": 0}
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Bootstrap-derived two-sided p-value: 2x the smaller tail fraction
    # crossing zero (standard percentile-bootstrap significance test).
    frac_le0 = float((diffs <= 0).mean())
    frac_ge0 = float((diffs >= 0).mean())
    p_value = float(min(1.0, 2 * min(frac_le0, frac_ge0)))
    return {
        "observed_a": observed_a, "observed_b": observed_b, "observed_diff": observed_b - observed_a,
        "ci95_lo": float(lo), "ci95_hi": float(hi), "n_boot_valid": len(diffs), "p_value": p_value,
    }


# ----------------------------------------------------------------------
# Persisting fold assignments / OOF predictions (for all-12-layer runs)
# ----------------------------------------------------------------------

def save_fold_assignments_csv(oof_data: dict, path: str) -> int:
    """
    oof_data : {(attribute, condition, layer): {...}} from run_linear_probe
    (collect_oof=True). Fold assignment is layer-independent by
    construction (the SAME splits object is reused for every layer within
    an (attribute, condition)), so this saves ONE row per (attribute,
    condition, object_key) using the first layer's fold_id -- verified
    identical across layers before writing.
    """
    layers_by_ac = {}
    for (attr, cond, layer), d in oof_data.items():
        layers_by_ac.setdefault((attr, cond), {})[layer] = d

    rows = []
    for (attr, cond), by_layer in layers_by_ac.items():
        layers_sorted = sorted(by_layer.keys())
        ref = by_layer[layers_sorted[0]]
        order = np.argsort(ref["object_key"])
        ref_keys_sorted = ref["object_key"][order]
        ref_folds_sorted = ref["fold_id"][order]
        for other_l in layers_sorted[1:]:
            d = by_layer[other_l]
            o = np.argsort(d["object_key"])
            if not np.array_equal(d["object_key"][o], ref_keys_sorted) or \
               not np.array_equal(d["fold_id"][o], ref_folds_sorted):
                raise RuntimeError(
                    f"STOP: fold assignment differs across layers for {attr}/{cond} "
                    f"(layer {layers_sorted[0]} vs {other_l}) -- layers must share identical folds")
        for key, img, fold in zip(ref["object_key"], ref["image_id"], ref["fold_id"]):
            rows.append({"attribute": attr, "condition": cond, "object_key": key,
                         "image_id": img, "fold": int(fold)})

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["attribute", "condition", "object_key", "image_id", "fold"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def save_oof_predictions_csv(oof_data: dict, path: str) -> int:
    """One row per (attribute, condition, layer, object_key): out-of-fold
    y_true/y_proba, plus which fold held it out."""
    rows = []
    for (attr, cond, layer), d in oof_data.items():
        for key, img, fold, yt, yp in zip(d["object_key"], d["image_id"], d["fold_id"], d["y_true"], d["y_proba"]):
            rows.append({"attribute": attr, "condition": cond, "layer": layer, "object_key": key,
                         "image_id": img, "fold": int(fold), "y_true": int(yt), "y_proba": float(yp)})
    fieldnames = ["attribute", "condition", "layer", "object_key", "image_id", "fold", "y_true", "y_proba"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    return len(rows)


# ----------------------------------------------------------------------
# Multiple-comparison correction and peak-layer detection
# ----------------------------------------------------------------------

def benjamini_hochberg_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Standard BH step-up FDR correction. NaN p-values pass through as NaN
    and are excluded from the correction family (not counted toward m)."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    valid_mask = ~np.isnan(pvals)
    valid_p = pvals[valid_mask]
    m = len(valid_p)
    if m == 0:
        return out
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    q = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest p-value downward
    q_monotone = np.minimum.accumulate(q[::-1])[::-1]
    q_monotone = np.clip(q_monotone, 0, 1)
    corrected = np.empty(m)
    corrected[order] = q_monotone
    out[valid_mask] = corrected
    return out


def peak_layer_analysis(auprc_by_layer: Dict[int, float], diff_results: Dict[tuple, dict], all_layers: Sequence[int]):
    """
    auprc_by_layer : {layer: point-estimate AUPRC}.
    diff_results   : {(layer_b, layer_a): image_level_bootstrap_auprc_diff(...) result}
                      must include every (peak, l) pair for l in all_layers
                      (direction: layer_b=peak, layer_a=l, so diff>0 means
                      peak is higher than l).
    Returns {"peak_layer": int, "peak_auprc": float,
             "layers_not_significantly_below_peak": [int, ...] (peak's own
             tier -- CI of (peak - l) includes 0, i.e. not clearly worse),
             "near_peak_label": "L8" or "L7-L9"-style string}.
    """
    peak_layer = max(auprc_by_layer, key=auprc_by_layer.get)
    tier = [peak_layer]
    for l in all_layers:
        if l == peak_layer:
            continue
        key = (peak_layer, l)
        r = diff_results.get(key)
        if r is not None and r["ci95_lo"] <= 0 <= r["ci95_hi"]:
            tier.append(l)
    tier_sorted = sorted(tier)
    if len(tier_sorted) == 1:
        label = f"L{tier_sorted[0]}"
    elif tier_sorted == list(range(min(tier_sorted), max(tier_sorted) + 1)):
        label = f"L{min(tier_sorted)}-L{max(tier_sorted)}" if len(tier_sorted) > 1 else f"L{tier_sorted[0]}"
    else:
        label = "L" + ",L".join(str(l) for l in tier_sorted)
    return {
        "peak_layer": peak_layer, "peak_auprc": auprc_by_layer[peak_layer],
        "layers_not_significantly_below_peak": tier_sorted, "near_peak_label": label,
    }
