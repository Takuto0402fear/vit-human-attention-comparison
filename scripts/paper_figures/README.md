# Paper figures (LAU-conference submission)

Re-plots existing analysis results into the paper's Figures 1-3 and
Appendix figures. **Does not run any model inference, probe training, or
attention extraction** -- every number comes from files already under
`outputs/`. See `paper_figures/FIGURE_AUDIT.md` for full provenance,
methodology notes, and caption drafts.

## Layout

- `common.py` -- shared constants (colors/linestyles/markers, font/size,
  layer highlighting), data loaders, and the image-level bootstrap CI
  helper (seed=42, n_boot=10000, alpha=0.05 -- the convention already used
  elsewhere in this repo, e.g. `scripts/stats_three_model_layerwise.py`
  and `outputs/clip_attention_b2_distribution/summary.json`).
- `figure1_model_comparison_nss.py` -- ORIGINAL Figure 1 (single-panel,
  4-model NSS comparison). Kept unmodified, not overwritten.
- `figure1_model_comparison_all_metrics.py` -- REVISED, main-text Figure 1:
  3 panels, (a) NSS / (b) AUC-Judd / (c) sAUC, same 4 models. Performs no
  new computation -- reuses `common.load_all_model_curves` unchanged (the
  same function behind the original Figure 1 and the Appendix's
  AUC-Judd/sAUC comparison figures). See paper_figures/FIGURE_AUDIT.md
  section 11 for the full rationale, per-metric CLIP-B profile, and the
  sAUC negative-pool caveat.
- `figure2_gaze_and_probes.py` -- Figure 2 (CLIP-B gaze alignment vs.
  semantic/low-level probes, 3 panels).
- `figure3_spatial_characterization.py` -- ORIGINAL Figure 3 (L4/L8/L12
  attention spatial characterization, one SHARED color scale across
  L4/L8/L12 per image). Kept unmodified; its output is now the "common
  scale" appendix supplementary figure (via
  `appendix_figure3_common_scale.py`), not the main-text figure.
- `figure3_spatial_characterization_revised.py` -- REVISED, main-text
  Figure 3: same quantitative bottom row, but the top-row L4/L8/L12
  attention maps each use their OWN (layer-specific) color scale and
  colorbar, so L4's much smaller raw attention values remain visible.
  Also produces the appendix "additional examples" figure (the other 2
  representative images, same layer-specific-scale convention). See
  paper_figures/FIGURE_AUDIT.md, "Figure 3 revision" section, for the full
  rationale and the exact color-scale/colorbar/contour conventions.
- `appendix_figure3_common_scale.py` -- thin wrapper that re-saves the
  ORIGINAL (unmodified) `figure3_spatial_characterization.make_figure()`
  output under the appendix path, so the shared-scale version stays
  available as supplementary material without duplicating its code.
- `appendix_figures.py` -- Appendix figures 1-5 (AUC-Judd/sAUC 4-model
  comparisons, CLIP-only 3-metric panel, per-attribute and per-feature
  small multiples).
- `verify_figures.py` -- re-derives key numbers straight from source files
  and checks them against the task's sanity-check targets. Run after
  regenerating figures.
- `run_all.py` -- runs all of the above in sequence, then verifies.

## Usage (PowerShell, from repo root)

```powershell
python -m scripts.paper_figures.run_all
```

or individually:

```powershell
python -m scripts.paper_figures.figure1_model_comparison_nss
python -m scripts.paper_figures.figure1_model_comparison_all_metrics
python -m scripts.paper_figures.figure2_gaze_and_probes
python -m scripts.paper_figures.figure3_spatial_characterization
python -m scripts.paper_figures.figure3_spatial_characterization_revised
python -m scripts.paper_figures.appendix_figure3_common_scale
python -m scripts.paper_figures.appendix_figures
python -m scripts.paper_figures.verify_figures
```

## Output

- `paper_figures/figure{1,2}_*.{pdf,png,svg}` -- primary outputs (2-column
  width). `figure1`/`figure2` also get explicit `_1col`/`_2col` width
  variants for layout-checking in the manuscript.
- `paper_figures/figure1_model_comparison_all_metrics_2col.{pdf,png,svg}`
  -- **the main-text Figure 1** (3 panels: NSS/AUC-Judd/sAUC).
  `_1col.{pdf,png}` stacks the 3 panels vertically for a 1-column check
  (3 side-by-side panels do not fit legibly in a single narrow column).
- `paper_figures/figure3_spatial_characterization.{pdf,png,svg}` -- the
  original, common-color-scale Figure 3 (no longer the main-text figure;
  kept for reference and reused verbatim as an appendix figure).
- `paper_figures/figure3_spatial_characterization_revised.{pdf,png,svg}`
  -- **the main-text Figure 3** (layer-specific color scales).
- `paper_figures/appendix/appendix_*.{pdf,png,svg}`,
  `paper_figures/appendix/figure_attention_common_scale.{pdf,png,svg}`,
  `paper_figures/appendix/figure_attention_layer_specific_additional_examples.{pdf,png}`.

PNGs are rasterized at >=300 dpi; PDFs embed fonts as real (searchable,
vector) text (`pdf.fonttype=42`), not outlined paths; SVGs keep text as
text (`svg.fonttype='none'`).

## Reproducibility

All aggregation (bootstrap CI, group means) is deterministic given a fixed
seed (`common.BOOT_SEED = 42`, matching the rest of the repo). Re-running
`run_all.py` regenerates byte-for-byte identical numeric results (plot
anti-aliasing/metadata may differ slightly between matplotlib versions).
No existing file under `outputs/`, `scripts/` (outside this directory), or
`lib/` is ever modified.
