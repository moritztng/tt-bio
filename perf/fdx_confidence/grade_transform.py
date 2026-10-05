#!/usr/bin/env python3
"""Grade tt-bio's contact rule on upstream OpenFold3's own logits from a real fold.

Upstream's `compute_global_predicted_distance_error` returns contact probabilities beside the
distogram logits it read them from (both saved by of3_upstream_ref.py). Feeding the same logits
to `confidence_export.contact_probs` must reproduce upstream's array: same inputs, so any
deviation is the transform's, not the model's. The diagonal is excluded (tt-bio defines it as 1).

    python perf/fdx_confidence/grade_transform.py perf/fdx_confidence/ref/of3_9bk6_seed0.npz
"""
import json
import sys

import numpy as np

from tt_bio import confidence_export as ce


def main(path):
    z = np.load(path)
    logits = z["distogram_logits"]
    while logits.ndim > 3:
        logits = logits[0]
    ref = z["contact_probs"]
    while ref.ndim > 2:
        ref = ref[0]
    ours, cutoff = ce.contact_probs(logits, ce.bin_upper_edges(*ce.DISTOGRAM_GRID["openfold3"]))
    off = ~np.eye(len(ref), dtype=bool)
    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64))[off]
    print(json.dumps({"n_tokens": len(ref), "cutoff_A": float(cutoff),
                      "max_abs_dev": float(d.max()), "mean_abs_dev": float(d.mean()),
                      "upstream_asymmetry_max": float(np.abs(ref - ref.T).max())}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
