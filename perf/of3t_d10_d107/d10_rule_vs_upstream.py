#!/usr/bin/env python3
"""Our sample-ranking rule against OpenFold3's own, on identical values, in float64.

The rule that picks which structure a user is handed is
`0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash` (`openfold3_fold.py:277`). Upstream
computes the same expression in `full_complex_sample_ranking_metric`
(`core/metrics/sample_ranking.py:105-112`) with the same four weights, so the question is not
the formula but the terms. This compares the pTM/ipTM term -- the only one that is non-zero on
a single chain -- against upstream's `compute_ptm`, imported from the 0.4.3 checkout by path
rather than reimplemented, on the same logits.

Three cases, because the answer is not the same in all three:

  monomer     one chain, every token a standard residue. What 1UBQ is.
  complex     two chains, every token a standard residue.
  ligand      two chains with eight tokens whose frame upstream rejects. `compute_ptm` zeroes
              a token's row before the outer max when `has_frame` is False (:154); ours takes
              the max over every token, because `_ptm_iptm` has no frame argument at all.

Reference dtype is float64 throughout. Ours casts to float32 inside `_ptm_iptm`
(`probs = torch.softmax(pae_logits.float(), -1)`), so upstream's own float32 reading is
carried beside it to separate "our arithmetic" from "our rule".
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("REF", "/home/ttuser/of3t_refprec/of3pkg043"))
sys.path.insert(0, os.environ.get("WT", "/home/ttuser/.coworker/wt/of3t-d10-d107"))

from openfold3.core.metrics.confidence import compute_ptm          # noqa: E402
from tt_bio.protenix import ConfidenceHead                         # noqa: E402

BINS = dict(bin_min=0, bin_max=32, no_bins=64)    # of3_all_atom model_config.py:540-543
N_TOK, N_SAMPLE, SEED = 76, 5, 10


def upstream(logits, asym_id, token_mask, has_frame, interface):
    return compute_ptm(logits=logits, has_frame=has_frame, mask_i=token_mask,
                       asym_id=asym_id, interface=interface, **BINS)


def case(name, n_chains, n_frameless):
    g = torch.Generator().manual_seed(SEED)
    logits = torch.randn(N_SAMPLE, N_TOK, N_TOK, BINS["no_bins"], generator=g,
                         dtype=torch.float64)
    logits = 0.5 * (logits + logits.transpose(1, 2))      # PAE logits are not symmetric in
    asym_id = torch.zeros(N_TOK, dtype=torch.long)        # general; symmetric is the harder
    if n_chains > 1:                                      # case for a max-over-i reduction
        asym_id[N_TOK // 2:] = 1
    token_mask = torch.ones(N_TOK, dtype=torch.bool)
    has_frame = torch.ones(N_SAMPLE, N_TOK, dtype=torch.bool)
    if n_frameless:
        # The frameless tokens are given the HIGHEST rows, which is what an atomized ligand
        # token looks like: few neighbours, low predicted error, and upstream does not let it
        # win the outer max.
        has_frame[:, :n_frameless] = False
        logits[:, :n_frameless, :, :] += 4.0
        logits[:, :, :n_frameless, :] += 4.0

    up_ptm = upstream(logits, asym_id, token_mask, has_frame, False)
    up_iptm = (upstream(logits, asym_id, token_mask, has_frame, True)
               if n_chains > 1 else torch.zeros(N_SAMPLE, dtype=torch.float64))
    up32_ptm = upstream(logits.float(), asym_id, token_mask, has_frame, False).double()
    # Upstream with every frame accepted: isolates the has_frame mask from everything else.
    up_noframe_ptm = upstream(logits, asym_id, token_mask,
                              torch.ones_like(has_frame), False)

    ours_ptm, ours_iptm, fix_ptm, fix_iptm = [], [], [], []
    for s in range(N_SAMPLE):
        p, i = ConfidenceHead._ptm_iptm(logits[s], asym_id)
        ours_ptm.append(p)
        ours_iptm.append(i)
        # The fix arm: the same function, handed the mask upstream computes.
        p, i = ConfidenceHead._ptm_iptm(logits[s], asym_id, has_frame=has_frame[s])
        fix_ptm.append(p)
        fix_iptm.append(i)
    ours_ptm = torch.tensor(ours_ptm, dtype=torch.float64)
    ours_iptm = torch.tensor(ours_iptm, dtype=torch.float64)
    fix_ptm = torch.tensor(fix_ptm, dtype=torch.float64)
    fix_iptm = torch.tensor(fix_iptm, dtype=torch.float64)

    def order(x):
        return np.argsort(-np.asarray(x, dtype=float), kind="stable").tolist()

    score_up = 0.8 * up_iptm + 0.2 * up_ptm
    score_ours = 0.8 * ours_iptm + 0.2 * ours_ptm
    score_fix = 0.8 * fix_iptm + 0.2 * fix_ptm
    return {
        "max_abs_ptm_fix_vs_upstream_f64": float((fix_ptm - up_ptm).abs().max()),
        "max_abs_iptm_fix_vs_upstream_f64": float((fix_iptm - up_iptm).abs().max()),
        "max_abs_ptm_fix_vs_ours_unmasked": float((fix_ptm - ours_ptm).abs().max()),
        "rank_order_fix": order(score_fix),
        "fix_matches_upstream_order": order(score_fix) == order(score_up),
        "case": name, "chains": n_chains, "frameless_tokens": n_frameless,
        "ptm_ours": [round(float(v), 9) for v in ours_ptm],
        "ptm_upstream_f64": [round(float(v), 9) for v in up_ptm],
        "max_abs_ptm_ours_vs_upstream_f64": float((ours_ptm - up_ptm).abs().max()),
        "max_abs_ptm_upstream_f32_vs_f64": float((up32_ptm - up_ptm).abs().max()),
        "max_abs_ptm_upstream_noframe_vs_frame": float(
            (up_noframe_ptm - up_ptm).abs().max()),
        "max_abs_iptm_ours_vs_upstream_f64": float((ours_iptm - up_iptm).abs().max()),
        "rank_order_ours": order(score_ours),
        "rank_order_upstream": order(score_up),
        "rank_orders_agree": order(score_ours) == order(score_up),
        "served_index_ours": order(score_ours)[0],
        "served_index_upstream": order(score_up)[0],
    }


if __name__ == "__main__":
    out = {"bins": BINS, "n_tokens": N_TOK, "n_samples": N_SAMPLE, "seed": SEED,
           "upstream": compute_ptm.__module__,
           "ours": "tt_bio.protenix.ConfidenceHead._ptm_iptm, via openfold3_fold._confidence",
           "cases": [case("monomer", 1, 0), case("complex", 2, 0),
                     case("ligand", 2, 8)]}
    print(json.dumps(out, indent=1))
