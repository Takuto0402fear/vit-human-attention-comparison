"""
Post-generation verification for the paper figures (see the "検証" section
of the task brief). Read-only: re-derives numbers straight from the source
CSV/JSON files and compares them against (a) the values actually plotted
and (b) the task's sanity-check targets. Prints a PASS/FAIL style report;
does not modify any file. Run after figure1/figure2/figure3/appendix have
been generated.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.paper_figures import common as C

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILURES.append(name)
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def close(a, b, tol=1e-3):
    return abs(a - b) <= tol


def main():
    # 1. Layer ordering 1..12 in every per-image source.
    for model_key, spec in C.MODEL_SOURCES.items():
        df = pd.read_csv(spec["per_image_csv"])
        layers = sorted(df["layer"].unique())
        check(f"layers 1..12 present: {model_key}", layers == list(range(1, 13)), str(layers))

    # 2. No missing values / duplicate (image, layer) rows.
    for model_key, spec in C.MODEL_SOURCES.items():
        df = pd.read_csv(spec["per_image_csv"])
        dup = df.duplicated(subset=["image", "layer"]).sum()
        nan = df[list(spec["col_map"].values())].isna().sum().sum()
        check(f"no dup (image,layer): {model_key}", dup == 0, f"dup={dup}")
        check(f"no NaNs: {model_key}", nan == 0, f"nan={nan}")

    # 3. Cross-check CLIP-B NSS curve used in Figure 1/2 against the
    #    pre-existing metrics_by_layer.csv (independent aggregate).
    canonical = pd.read_csv(REPO_ROOT / "outputs" / "expA_clip" / "healthy" / "metrics_by_layer.csv")
    _, mat = C.load_model_layer_matrix("CLIP-B", "NSS")
    recomputed_mean = mat.mean(axis=0)
    for layer in C.LAYERS:
        ref = canonical.loc[canonical["layer"] == layer, "NSS_mean"].iloc[0]
        got = recomputed_mean[layer - 1]
        check(f"CLIP-B NSS L{layer} matches metrics_by_layer.csv", close(ref, got, 1e-6),
              f"ref={ref:.6f} got={got:.6f}")

    # 4. Sanity-check targets from the task brief.
    targets = [
        ("CLIP-B NSS peak layer", 4, np.argmax(recomputed_mean) + 1),
        ("CLIP-B NSS peak value", 1.388, round(float(recomputed_mean.max()), 3)),
    ]
    _, dino_s_mat = C.load_model_layer_matrix("DINO-S", "NSS")
    dino_s_mean = dino_s_mat.mean(axis=0)
    targets.append(("DINO-S NSS peak layer", 10, int(np.argmax(dino_s_mean) + 1)))
    targets.append(("DINO-S NSS peak value", 1.918, round(float(dino_s_mean.max()), 3)))
    _, dino_b_mat = C.load_model_layer_matrix("DINO-B", "NSS")
    dino_b_mean = dino_b_mat.mean(axis=0)
    targets.append(("DINO-B NSS peak layer", 11, int(np.argmax(dino_b_mean) + 1)))
    targets.append(("DINO-B NSS peak value", 1.755, round(float(dino_b_mean.max()), 3)))

    for name, want, got in targets:
        ok = (want == got) if isinstance(want, int) else close(want, got, 0.01)
        check(f"sanity target: {name}", ok, f"want={want} got={got}")

    # 5. m_contrast CLIP-B ~ +0.883 (from existing statistics_summary.md /
    #    m_contrast_summary.csv -- NOT recomputed here, just re-read).
    mcs = pd.read_csv(REPO_ROOT / "outputs" / "model_comparison_statistics" / "m_contrast_summary.csv")
    clip_mc = mcs[(mcs["model"] == "CLIP-B") & (mcs["metric"] == "NSS") & (mcs["measure"] == "m_contrast")]
    check("sanity target: CLIP-B NSS m_contrast ~ +0.883",
          close(0.883, clip_mc["mean"].iloc[0], 0.01), f"got={clip_mc['mean'].iloc[0]:.4f}")

    # 6. Semantic-probe / low-level-probe group means match layer_profile_table.csv.
    lpt = pd.read_csv(REPO_ROOT / "outputs" / "osie_layer_profile_synthesis" / "layer_profile_table.csv")
    sem_df = C.load_semantic_probe_table()
    sem_pivot = sem_df.pivot_table(index="layer", columns="attribute", values="mean")
    sem_group_mean = sem_pivot[C.SEMANTIC_ATTRIBUTES].mean(axis=1)
    low_df = C.load_lowlevel_probe_table()
    low_pivot = low_df.pivot_table(index="layer", columns="target", values="mean")
    low_group_mean = low_pivot[C.LOWLEVEL_FEATURES].mean(axis=1)
    for _, row in lpt.iterrows():
        layer = int(row["layer"])
        if layer not in sem_group_mean.index:
            continue
        check(f"semantic mean AUPRC L{layer} matches layer_profile_table.csv",
              close(row["semantic_probe_mean_auprc"], sem_group_mean.loc[layer], 1e-3),
              f"ref={row['semantic_probe_mean_auprc']:.6f} got={sem_group_mean.loc[layer]:.6f}")
        check(f"lowlevel mean R2 L{layer} matches layer_profile_table.csv",
              close(row["lowlevel_probe_mean_r2"], low_group_mean.loc[layer], 1e-3),
              f"ref={row['lowlevel_probe_mean_r2']:.6f} got={low_group_mean.loc[layer]:.6f}")

    # 7. B2 spatial summary sanity targets (used in Figure 3).
    with open(C.B2_SUMMARY_JSON, encoding="utf-8") as fh:
        summary = json.load(fh)
    desc = summary["descriptive_per_layer"]
    sim = summary["spatial_similarity_descriptive"]
    b2_targets = [
        ("mass50_n_patches L4", 562, desc["mass50_n_patches"]["L4"]["mean"], 5),
        ("mass50_n_patches L8", 122, desc["mass50_n_patches"]["L8"]["mean"], 3),
        ("mass50_n_patches L12", 68, desc["mass50_n_patches"]["L12"]["mean"], 3),
        ("foreground_enrichment L4", 1.60, desc["foreground_enrichment"]["L4"]["mean"], 0.02),
        ("foreground_enrichment L8", 1.00, desc["foreground_enrichment"]["L8"]["mean"], 0.02),
        ("foreground_enrichment L12", 1.57, desc["foreground_enrichment"]["L12"]["mean"], 0.02),
        ("spatial corr L4-L8", -0.048, sim["L4-L8_pearson"]["same_image_mean"], 0.005),
        ("spatial corr L8-L12", 0.958, sim["L8-L12_pearson"]["same_image_mean"], 0.005),
        ("spatial corr L4-L12", 0.015, sim["L4-L12_pearson"]["same_image_mean"], 0.005),
    ]
    for name, want, got, tol in b2_targets:
        check(f"sanity target: {name}", close(want, got, tol), f"want={want} got={got:.3f}")
    check("B2 summary.json bootstrap convention matches repo standard",
          summary["seed"] == 42 and summary["n_boot"] == 10000 and summary["alpha"] == 0.05,
          f"seed={summary['seed']} n_boot={summary['n_boot']} alpha={summary['alpha']}")

    # 8. Figure 3 (revised) representative-image / cache / fixmap consistency.
    cache = np.load(str(C.ATTN_CACHE_NPZ), allow_pickle=False)
    stems_all = cache["stems"].tolist()
    layers_display = cache["layers_display"].tolist()
    check("Figure3 cache layer order is [4, 8, 12]", layers_display == [4, 8, 12],
          str(layers_display))
    attn_all = cache["attn"]
    fig3_stems = ["1156", "1159", "1213"]
    for stem in fig3_stems:
        in_cache = stem in stems_all
        check(f"Figure3 stem {stem} present in attention cache", in_cache)
        fixmap_path = C.FIXMAP_DIR / f"{stem}.npz"
        check(f"Figure3 stem {stem} has a healthy-group fixmap", fixmap_path.is_file(),
              str(fixmap_path))
        stim_path = C.STIMULI_DIR / f"{stem}.jpg"
        check(f"Figure3 stem {stem} has a stimulus image", stim_path.is_file(), str(stim_path))
        if not (in_cache and fixmap_path.is_file()):
            continue
        idx = stems_all.index(stem)
        layer_maps = attn_all[idx]  # (3, 38, 50)
        check(f"Figure3 stem {stem} attention is non-negative and finite",
              bool(np.isfinite(layer_maps).all()) and bool((layer_maps >= 0).all()))
        fixmap = np.load(str(fixmap_path))
        check(f"Figure3 stem {stem} fixmap 'heat_all' shape is (600, 800)",
              fixmap["heat_all"].shape == (600, 800), str(fixmap["heat_all"].shape))

    print(f"\n{len(FAILURES)} check(s) failed." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
