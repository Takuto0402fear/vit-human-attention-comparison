"""Regenerate all paper figures (Figures 1-3 + Appendix) and verify them.

Usage (PowerShell, from repo root):
    python -m scripts.paper_figures.run_all

See scripts/paper_figures/README.md for details and paper_figures/
FIGURE_AUDIT.md for full data provenance.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper_figures import figure1_model_comparison_nss
from scripts.paper_figures import figure1_model_comparison_all_metrics
from scripts.paper_figures import figure2_gaze_and_probes
from scripts.paper_figures import figure3_spatial_characterization
from scripts.paper_figures import figure3_spatial_characterization_revised
from scripts.paper_figures import appendix_figure3_common_scale
from scripts.paper_figures import appendix_figures
from scripts.paper_figures import verify_figures


def main():
    figure1_model_comparison_nss.main()
    figure1_model_comparison_all_metrics.main()  # main-text Figure 1 (3-panel)
    figure2_gaze_and_probes.main()
    figure3_spatial_characterization.main()  # common-scale; kept for appendix reuse below
    figure3_spatial_characterization_revised.main()  # main-text Figure 3 (layer-specific scale)
    appendix_figure3_common_scale.main()
    appendix_figures.main()
    print("\n--- verification ---")
    return verify_figures.main()


if __name__ == "__main__":
    sys.exit(main())
