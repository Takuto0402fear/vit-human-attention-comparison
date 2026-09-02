"""One-off helper (this audit only) to build appendix_semantic_labels.csv
and appendix_low_level_features.csv from existing canonical outputs.
Read-only: does not modify any existing file. Not part of the regular
paper_figures pipeline (not imported by run_all.py)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

# ---------------------------------------------------------------------
# Semantic attribute labels
# ---------------------------------------------------------------------

meta = pd.read_csv(REPO_ROOT / "outputs" / "osie_probe_all12_full700" / "object_features_metadata.csv")
summ = pd.read_csv(REPO_ROOT / "outputs" / "osie_probe_all12_full700" / "probe_summary.csv")
summ = summ[(summ["metric"] == "auprc") & (summ["condition"] == "all_objects")]

ATTRS = ["Text", "Face", "Emotion", "Sound", "Smell", "Taste", "Touch",
         "Motion", "Operability", "Watchability", "Touched", "Gazed"]

JA_NAMES = {
    "Text": "文字", "Face": "顔", "Emotion": "感情表出", "Sound": "発音性(音を発する)",
    "Smell": "におい", "Taste": "味覚(食べ物・飲み物)", "Touch": "触感的特徴",
    "Motion": "動き", "Operability": "操作可能性", "Watchability": "鑑賞対象性",
    "Touched": "接触されている(画像内で)", "Gazed": "注視されている(画像内で)",
}
EN_NAMES = {
    "Text": "Text", "Face": "Face", "Emotion": "Emotion (clear facial emotion)",
    "Sound": "Sound-producing", "Smell": "Smell", "Taste": "Taste (food/drink)",
    "Touch": "Distinctive tactile quality", "Motion": "Motion",
    "Operability": "Operability", "Watchability": "Watchability",
    "Touched": "Touched (by a person/animal in the image)",
    "Gazed": "Gazed at (by a person/animal in the image)",
}
DEFS = {
    "Text": "written text, letters, or numbers", "Face": "a face",
    "Emotion": "a face showing a clear emotion", "Sound": "an object producing sound",
    "Smell": "an object with a noticeable smell", "Taste": "food or drink that can be tasted",
    "Touch": "an object with a distinctive tactile quality",
    "Motion": "an object or living being showing motion",
    "Operability": "an object designed to be operated by hand",
    "Watchability": "an object designed to be watched",
    "Touched": "an object being touched by a person or animal",
    "Gazed": "an object being looked at by a person or animal",
}

rows = []
n_total = len(meta)
for a in ATTRS:
    pos = meta["positive_attributes"].fillna("").apply(lambda s, a=a: a in s.split(";")).sum()
    neg = meta["negative_attributes"].fillna("").apply(lambda s, a=a: a in s.split(";")).sum()
    excluded = n_total - (pos + neg)
    s = summ[summ["attribute"] == a].set_index("layer")["mean"]
    peak_l = int(s.idxmax())
    peak_v = float(s.max())
    rows.append(dict(
        internal_label=a, paper_label_ja=JA_NAMES[a], paper_label_en=EN_NAMES[a],
        definition=DEFS[a], positive_count=int(pos), negative_count=int(neg),
        excluded_count=int(excluded), total_count=n_total,
        best_layer=peak_l, best_auprc=round(peak_v, 6),
        source_file="outputs/osie_probe_all12_full700/object_features_metadata.csv; "
                     "outputs/osie_probe_all12_full700/probe_summary.csv",
        evidence="positive_attributes/negative_attributes columns, exact split counted "
                 "in this audit (object_features_metadata.csv); peak layer/value = "
                 "argmax of probe_summary.csv metric==auprc, condition==all_objects, "
                 "mean column",
    ))
sem_df = pd.DataFrame(rows)
sem_out = REPO_ROOT / "appendix_semantic_labels.csv"
sem_df.to_csv(sem_out, index=False, encoding="utf-8-sig")
print(f"wrote {sem_out} ({len(sem_df)} rows)")

# ---------------------------------------------------------------------
# Low-level visual features
# ---------------------------------------------------------------------

low = pd.read_csv(REPO_ROOT / "outputs" / "osie_lowlevel_probe_all12_full700" / "lowlevel_summary.csv")
low_r2 = low[low["metric"] == "r2"]
targets_raw = pd.read_csv(REPO_ROOT / "outputs" / "osie_lowlevel_targets_full700" / "lowlevel_object_targets.csv")

FEATS = ["mean_luminance", "mean_red", "mean_green", "mean_blue", "mean_saturation",
         "edge_strength", "fine_texture", "luminance_contrast"]

LOW_JA = {
    "mean_luminance": "平均輝度", "mean_red": "平均赤(R)", "mean_green": "平均緑(G)",
    "mean_blue": "平均青(B)", "mean_saturation": "平均彩度",
    "edge_strength": "エッジ強度", "fine_texture": "微細テクスチャ", "luminance_contrast": "輝度コントラスト",
}
LOW_EN = {
    "mean_luminance": "Mean luminance", "mean_red": "Mean red", "mean_green": "Mean green",
    "mean_blue": "Mean blue", "mean_saturation": "Mean saturation",
    "edge_strength": "Edge strength", "fine_texture": "Fine texture",
    "luminance_contrast": "Luminance contrast",
}
LOW_DEF = {
    "mean_luminance": "0.2126R+0.7152G+0.0722B (ITU-R BT.709), mean inside mask, RGB in [0,1]",
    "mean_red": "mean R channel value inside mask, [0,1]",
    "mean_green": "mean G channel value inside mask, [0,1]",
    "mean_blue": "mean B channel value inside mask, [0,1]",
    "mean_saturation": "HSV saturation (matplotlib.colors.rgb_to_hsv), mean inside mask, [0,1]",
    "edge_strength": "Sobel gradient magnitude sqrt(gx^2+gy^2) (scipy.ndimage.sobel, computed on the "
                      "FULL 224x224 luminance image), mean inside mask",
    "fine_texture": "Laplacian (scipy.ndimage.laplace, FULL image), variance inside mask",
    "luminance_contrast": "standard deviation of luminance inside mask",
}

rows = []
for f in FEATS:
    s = low_r2[low_r2["target"] == f].set_index("layer")["mean"]
    peak_l = int(s.idxmax())
    peak_v = float(s.max())
    valid_count = int(low_r2[low_r2["target"] == f]["n_valid_objects"].iloc[0])
    excluded_count = len(targets_raw) - valid_count
    vmin = float(targets_raw[f].min())
    vmax = float(targets_raw[f].max())
    rows.append(dict(
        internal_name=f, paper_label_ja=LOW_JA[f], paper_label_en=LOW_EN[f],
        definition=LOW_DEF[f], value_range=f"[{vmin:.4f}, {vmax:.4f}] (empirical; theoretical [0,1] "
                                             f"for the 5 color/saturation features)",
        valid_count=valid_count, excluded_count=excluded_count,
        best_layer=peak_l, best_r2=round(peak_v, 6),
        source_file="outputs/osie_lowlevel_probe_all12_full700/lowlevel_summary.csv; "
                     "outputs/osie_lowlevel_targets_full700/lowlevel_object_targets.csv",
        evidence="peak layer/value = argmax of lowlevel_summary.csv metric==r2, mean column; "
                 "valid/excluded counts from lowlevel_object_targets.csv 'valid' column "
                 "(skip_reason=='empty_mask_after_transform' for all 288 excluded objects); "
                 "value range = empirical min/max in lowlevel_object_targets.csv",
    ))
low_df = pd.DataFrame(rows)
low_out = REPO_ROOT / "appendix_low_level_features.csv"
low_df.to_csv(low_out, index=False, encoding="utf-8-sig")
print(f"wrote {low_out} ({len(low_df)} rows)")
