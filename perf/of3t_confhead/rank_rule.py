"""What the OpenFold3 selection rule reduces to on a single-chain target, and who else does it.

D10 is stated as a calibration failure. Before measuring calibration, this checks the rule that
consumes the head: `openfold3_fold.py::_confidence` selects with

    ranking_score = 0.8*iptm + 0.2*ptm + 0.5*disorder - 100*has_clash

AF3 SI 5.9.3's full-complex metric, which upstream openfold3 0.5.0 also implements
(`core/metrics/sample_ranking.py::full_complex_sample_ranking_metric`). On ONE chain both
implementations return ipTM = 0 by construction -- there is no cross-chain pair to average --
and `has_clash` is an INTER-chain indicator, so it is 0 too. The rule that actually ranks
1UBQ's five samples is therefore `0.2*ptm + 0.5*disorder`, in which the RASA disorder term
outweighs the only confidence term 2.5 to 1 and is REWARDED, not penalised.

Three checks, no device:
 1. ours and a transcription of upstream's `compute_ptm` agree that ipTM collapses to 0.0 on
    one chain, and both are non-zero on two -- the negative control, so "0.0" is the rule and
    not a broken call;
 2. the arithmetic of what that leaves: how much pTM spread a sample needs to overcome a given
    disorder difference;
 3. the other four models in the family, read from source: each one substitutes pTM when there
    is no interface. OpenFold3 is the only member that does not.

    python3 perf/of3t_confhead/rank_rule.py
"""
import json
import os
import re

import torch

from tt_bio.protenix import ConfidenceHead

OUT = "perf/of3t_confhead/rank_rule.json"
N, NB = 76, 64


def upstream_compute_ptm(logits, has_frame, bin_min, bin_max, no_bins, mask_i,
                         asym_id=None, interface=False, eps=1e-8):
    """Transcribed verbatim from openfold3 0.5.0 core/metrics/confidence.py::compute_ptm,
    so the single-chain behaviour is read off their arithmetic and not off their prose."""
    mask_i = mask_i.bool()
    if asym_id is not None:
        asym_id = asym_id[mask_i]
    num_tokens = mask_i.sum().clamp_min(1).double()
    clipped = torch.maximum(num_tokens, torch.tensor(19.0, dtype=torch.float64))
    d0 = 1.24 * (clipped - 15.0).clamp_min(0).pow(1.0 / 3.0) - 1.8
    width = (bin_max - bin_min) / no_bins
    centers = bin_min + width * (torch.arange(no_bins, dtype=torch.float64) + 0.5)
    bin_weight = 1.0 / (1.0 + (centers / d0) ** 2)
    logits = logits[:, mask_i, ...][..., mask_i, :]
    has_frame = has_frame[:, mask_i].bool()
    ptm_ij = (torch.softmax(logits.double(), -1) * bin_weight).sum(-1)
    if interface:
        pair = asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        tm_i = (ptm_ij * pair).sum(-1) / pair.sum(-1).clamp_min(eps)
    else:
        tm_i = ptm_ij.sum(-1) / num_tokens
    return float(tm_i.masked_fill(~has_frame, 0.0).max(-1).values)


torch.manual_seed(0)
logits = torch.randn(N, N, NB)
mono = torch.zeros(N)
duo = torch.cat([torch.zeros(N // 2), torch.ones(N - N // 2)])

rep = {"n_tokens": N, "no_bins": NB, "iptm": {}}
for name, asym in (("one_chain", mono), ("two_chains", duo)):
    ours = ConfidenceHead._ptm_iptm(logits, asym)
    theirs_ptm = upstream_compute_ptm(logits[None], torch.ones(1, N, dtype=torch.bool),
                                      0, 32, NB, torch.ones(N, dtype=torch.bool), asym, False)
    theirs_iptm = upstream_compute_ptm(logits[None], torch.ones(1, N, dtype=torch.bool),
                                       0, 32, NB, torch.ones(N, dtype=torch.bool), asym, True)
    rep["iptm"][name] = {"ours_ptm": ours[0], "ours_iptm": ours[1],
                         "upstream_ptm": theirs_ptm, "upstream_iptm": theirs_iptm}
    print(f"{name:11s} ours ptm {ours[0]:.6f} iptm {ours[1]:.6f} | "
          f"upstream ptm {theirs_ptm:.6f} iptm {theirs_iptm:.6f}")

assert rep["iptm"]["one_chain"]["ours_iptm"] == 0.0
assert rep["iptm"]["one_chain"]["upstream_iptm"] == 0.0
assert rep["iptm"]["two_chains"]["ours_iptm"] > 0.0, "control: two chains must score non-zero"
assert rep["iptm"]["two_chains"]["upstream_iptm"] > 0.0
print("-> ipTM is 0.0 on one chain in BOTH implementations, and non-zero on two. "
      "The single-chain collapse is the rule, not a broken call.")

# 2. what the rule reduces to
w = {"iptm": 0.8, "ptm": 0.2, "disorder": 0.5}
rep["reduced_rule"] = "0.2*ptm + 0.5*disorder"
rep["disorder_over_ptm_weight"] = w["disorder"] / w["ptm"]
rep["ptm_spread_to_overcome"] = {f"d_disorder={d}": round(w["disorder"] * d / w["ptm"], 4)
                                 for d in (0.01, 0.02, 0.05, 0.1)}
print(f"\nsingle chain -> ranking_score = {rep['reduced_rule']}; disorder carries "
      f"{rep['disorder_over_ptm_weight']:.1f}x the weight of pTM")
for k, v in rep["ptm_spread_to_overcome"].items():
    print(f"  to outrank a sample {k}, a rival needs pTM higher by {v}")

# 3. who handles the degeneracy, read from source
SITES = {
    "openfold3/openbind": ("tt_bio/openfold3_fold.py",
                           r"ranking_score\s*=\s*0\.8 \* iptm \+ 0\.2 \* ptm"),
    "boltz-2/boltzgen": ("tt_bio/boltz2.py", r"if not torch\.allclose\(\s*out\[.iptm.\]"),
    "protenix-v2/opendde": ("tt_bio/worker.py", r"return ptm if ptm > 0\.0 else c\[.plddt.\]"),
    "rf3": ("tt_bio/rf3/confidence.py", r"iptm_v = ptm_v if ptm_v is not None else 0\.0"),
}
rep["monomer_fallback"] = {}
for model, (path, pat) in SITES.items():
    src = open(path).read()
    hit = bool(re.search(pat, src))
    rep["monomer_fallback"][model] = {"file": path, "matched": hit}
print("\nmonomer ipTM fallback, by source site:")
for model, r in rep["monomer_fallback"].items():
    has = r["matched"] if model != "openfold3/openbind" else not r["matched"]
    verdict = ("substitutes pTM" if has else
               "NO fallback -- 0.8 of the weight budget is identically zero")
    print(f"  {model:20s} {verdict}   ({r['file']})")
rep["openfold3_is_the_outlier"] = (rep["monomer_fallback"]["openfold3/openbind"]["matched"]
                                   and all(rep["monomer_fallback"][m]["matched"]
                                           for m in SITES if m != "openfold3/openbind"))
assert rep["openfold3_is_the_outlier"], "source shape changed -- re-read the four sites"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=1)
print("\nwrote", OUT)
