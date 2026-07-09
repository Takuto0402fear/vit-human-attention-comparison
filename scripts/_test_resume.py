"""
Resume test harness: runs run_expA on a 20-image subset with CHUNK_SIZE=10.
Pass --abort-after-chunk=N to simulate crash after N chunks.
"""
import sys
import os

# Patch run_expA config BEFORE import
sys.path.insert(0, r"C:\Users\user\gaze")
sys.path.insert(0, r"C:\Users\user\gaze\scripts")

import run_expA

# Override to 20-image subset with small chunks
run_expA.CHUNK_SIZE = 10

# Monkey-patch get image list to return only 20 evenly spaced images
import glob
import numpy as np

_orig_stim_dir = run_expA.STIM_DIR

def _get_subset_image_list():
    all_jpgs = sorted(glob.glob(os.path.join(_orig_stim_dir, "*.jpg")))
    indices = np.linspace(0, len(all_jpgs) - 1, 20, dtype=int)
    return [(os.path.splitext(os.path.basename(all_jpgs[i]))[0], all_jpgs[i])
            for i in indices]

# Inject abort-after-chunk support
abort_after = None
for arg in sys.argv[1:]:
    if arg.startswith("--abort-after-chunk="):
        abort_after = int(arg.split("=")[1])

if abort_after is not None:
    _orig_append = run_expA._append_rows
    _chunk_counter = [0]

    def _counting_append(csv_path, rows):
        _orig_append(csv_path, rows)
        _chunk_counter[0] += 1
        if _chunk_counter[0] >= abort_after:
            print(f"\n*** SIMULATED CRASH after chunk {_chunk_counter[0]} ***")
            sys.exit(1)

    run_expA._append_rows = _counting_append

# Patch main to use subset
import types

_orig_main = run_expA.main

def _patched_main():
    """Wrap main to inject subset image list."""
    import run_expA as mod

    # Save originals
    _real_sorted = sorted
    _real_glob = glob.glob

    # We need to intercept the image_list construction inside main.
    # Simplest: temporarily replace glob.glob for STIM_DIR pattern
    _orig_glob_glob = glob.glob
    def _fake_glob(pattern, **kw):
        if "stimuli" in pattern and "*.jpg" in pattern:
            all_jpgs = _orig_glob_glob(pattern, **kw)
            indices = np.linspace(0, len(all_jpgs) - 1, 20, dtype=int)
            return [all_jpgs[i] for i in indices]
        return _orig_glob_glob(pattern, **kw)

    glob.glob = _fake_glob
    try:
        _orig_main()
    finally:
        glob.glob = _orig_glob_glob

_patched_main()
