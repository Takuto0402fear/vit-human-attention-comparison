"""
Appendix supplementary figure: the ORIGINAL (common/shared color scale)
Figure 3, unmodified, re-saved under the appendix output path.

This exists so the common-scale version is not deleted after Figure 3 was
revised to use layer-specific color scales in the main text (see
figure3_spatial_characterization_revised.py and paper_figures/
FIGURE_AUDIT.md, "Figure 3 revision" section). The common-scale version
(one shared vmax = max raw attention over L4/L8/L12 per image) is the
correct visualization for comparing RAW ATTENTION INTENSITY across
layers -- it is only unsuitable for comparing each layer's own spatial
distribution in isolation, which is what the revised main-text figure is
for.

This script performs NO new computation and reads no new data: it simply
calls figure3_spatial_characterization.make_figure() (defined in, and
unmodified from, the original script) and saves the result under a new
filename. The original script's own outputs
(paper_figures/figure3_spatial_characterization.{pdf,png,svg}) are left
untouched.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from scripts.paper_figures import common as C
from scripts.paper_figures import figure3_spatial_characterization as F3


def main():
    fig = F3.make_figure()
    C.save_fig(fig, C.APPENDIX_DIR / "figure_attention_common_scale")
    plt.close(fig)
    print("[appendix] wrote appendix/figure_attention_common_scale.{pdf,png,svg} "
          "(verbatim copy of the original figure3_spatial_characterization figure)")


if __name__ == "__main__":
    main()
