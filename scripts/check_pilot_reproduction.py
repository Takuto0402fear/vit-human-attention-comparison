"""
Reproducibility gate: confirms that the seed=42, 50-image pilot subset of
the full700 run (outputs/osie_attribute_grounding_full700/per_image_long.csv)
matches the original pilot run
(outputs/osie_attribute_grounding_pilot/per_image_long.csv) within floating-
point tolerance, for every (image_name, layer, attribute) row that exists in
the pilot.

This does NOT re-run any model -- it only compares two already-written CSVs.
Must be run (and PASS) before scripts/stats_osie_attribute_grounding_full700.py
or scripts/plot_osie_attribute_grounding_full700.py. If it does not pass,
this script exits with a non-zero code and the reason is written to
pilot_reproduction_check.json -- statistics/figures must not proceed.

Reads only outputs/osie_attribute_grounding_pilot/ (read-only) and
outputs/osie_attribute_grounding_full700/per_image_long.csv (read-only).
Writes only outputs/osie_attribute_grounding_full700/pilot_reproduction_check.json.

Usage (PowerShell):
    python scripts\\check_pilot_reproduction.py
"""
import sys
sys.path.insert(0, r"C:\Users\user\gaze")

import csv
import json
import math
import os

PILOT_DIR = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_pilot"
FULL700_DIR = r"C:\Users\user\gaze\outputs\osie_attribute_grounding_full700"

PILOT_LONG_CSV = os.path.join(PILOT_DIR, "per_image_long.csv")
FULL700_LONG_CSV = os.path.join(FULL700_DIR, "per_image_long.csv")
PILOT_RUN_CONFIG = os.path.join(PILOT_DIR, "run_config.json")

OUT_JSON = os.path.join(FULL700_DIR, "pilot_reproduction_check.json")

NUMERIC_FIELDS = ["attention_mass", "area_fraction", "area_normalized_enrichment"]
INT_FIELDS = ["mask_object_count", "mask_pixel_count"]
TOLERANCE = 1e-6  # abs tolerance for float comparison


def load_long_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    print("=" * 65)
    print("  Pilot reproduction check (full700 subset vs. original pilot)")
    print("=" * 65)

    if not os.path.isfile(PILOT_LONG_CSV):
        raise RuntimeError(f"STOP: {PILOT_LONG_CSV} not found")
    if not os.path.isfile(FULL700_LONG_CSV):
        raise RuntimeError(f"STOP: {FULL700_LONG_CSV} not found -- run the full700 extraction first")
    if not os.path.isfile(PILOT_RUN_CONFIG):
        raise RuntimeError(f"STOP: {PILOT_RUN_CONFIG} not found")

    with open(PILOT_RUN_CONFIG, encoding="utf-8") as f:
        pilot_run_config = json.load(f)
    pilot_image_ids = set(pilot_run_config["pilot_image_ids"])
    print(f"  Pilot image ids (seed=42, n={len(pilot_image_ids)}): "
          f"{sorted(pilot_image_ids)[:5]} ... (+{len(pilot_image_ids)-5} more)")

    pilot_rows = load_long_csv(PILOT_LONG_CSV)
    full700_rows = load_long_csv(FULL700_LONG_CSV)
    print(f"  Pilot per_image_long.csv: {len(pilot_rows)} rows")
    print(f"  Full700 per_image_long.csv: {len(full700_rows)} rows")

    def key(r):
        return (r["image_name"], int(r["layer"]), r["attribute"])

    pilot_index = {key(r): r for r in pilot_rows}
    if len(pilot_index) != len(pilot_rows):
        raise RuntimeError("STOP: duplicate keys within pilot per_image_long.csv itself")

    full700_index = {}
    for r in full700_rows:
        stem = os.path.splitext(r["image_name"])[0]
        if stem in pilot_image_ids:
            k = key(r)
            if k in full700_index:
                raise RuntimeError(f"STOP: duplicate key {k} within full700 subset")
            full700_index[k] = r

    pilot_keys = set(pilot_index.keys())
    full700_subset_keys = set(full700_index.keys())

    missing_in_full700 = sorted(pilot_keys - full700_subset_keys)
    extra_in_full700 = sorted(full700_subset_keys - pilot_keys)

    max_abs_diff = {field: 0.0 for field in NUMERIC_FIELDS}
    max_abs_diff_row = {field: None for field in NUMERIC_FIELDS}
    int_mismatches = []
    n_compared = 0

    for k in sorted(pilot_keys & full700_subset_keys):
        pr = pilot_index[k]
        fr = full700_index[k]
        n_compared += 1
        for field in NUMERIC_FIELDS:
            pv = float(pr[field])
            fv = float(fr[field])
            d = abs(pv - fv)
            if math.isnan(d):
                raise RuntimeError(f"STOP: NaN comparing {field} at {k}")
            if d > max_abs_diff[field]:
                max_abs_diff[field] = d
                max_abs_diff_row[field] = {"key": list(k), "pilot": pv, "full700": fv, "abs_diff": d}
        for field in INT_FIELDS:
            if int(pr[field]) != int(fr[field]):
                int_mismatches.append({
                    "key": list(k), "field": field,
                    "pilot": int(pr[field]), "full700": int(fr[field]),
                })

    all_max_diff = max(max_abs_diff.values()) if max_abs_diff else float("nan")
    tolerance_ok = all_max_diff <= TOLERANCE
    passed = (
        n_compared == len(pilot_keys)
        and not missing_in_full700
        and not extra_in_full700
        and not int_mismatches
        and tolerance_ok
    )

    print(f"\n  Rows compared: {n_compared} / {len(pilot_keys)} pilot rows")
    print(f"  Missing in full700 subset: {len(missing_in_full700)}")
    print(f"  Unexpected extra rows in full700 subset (shouldn't happen): {len(extra_in_full700)}")
    print(f"  Integer-field mismatches (mask_object_count/mask_pixel_count): {len(int_mismatches)}")
    for field in NUMERIC_FIELDS:
        print(f"  max abs diff [{field}]: {max_abs_diff[field]:.3e}  (tolerance {TOLERANCE:.1e})")
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")

    result = {
        "passed": passed,
        "tolerance": TOLERANCE,
        "n_pilot_rows": len(pilot_keys),
        "n_compared": n_compared,
        "n_missing_in_full700": len(missing_in_full700),
        "missing_in_full700_sample": missing_in_full700[:20],
        "n_extra_in_full700": len(extra_in_full700),
        "n_int_field_mismatches": len(int_mismatches),
        "int_field_mismatches_sample": int_mismatches[:20],
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_detail": max_abs_diff_row,
        "pilot_long_csv": PILOT_LONG_CSV,
        "full700_long_csv": FULL700_LONG_CSV,
        "pilot_image_ids": sorted(pilot_image_ids),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n  Saved: {OUT_JSON}")

    if not passed:
        raise RuntimeError(
            "STOP: pilot reproduction check FAILED -- do not proceed to statistics "
            f"or figures. See {OUT_JSON} for details.")

    print("\n" + "=" * 65)
    print("  Pilot reproduction check: PASS")
    print("=" * 65)


if __name__ == "__main__":
    main()
