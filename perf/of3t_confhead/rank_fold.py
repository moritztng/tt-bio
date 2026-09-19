"""One OpenFold3 fold, with every confidence signal recorded PER SAMPLE beside the true RMSD.

`of3t-pairbias` stored only the winner's aggregate confidence, which is why D10 -- "the
confidence head mis-ranks its own diffusion samples" -- could be seen and not diagnosed. This
harness keeps, for every diffusion sample of every run:

  rmsd_ca      Ca-RMSD against the experimental structure (Kabsch, gemmi), the ground truth
               the head is being judged against;
  plddt        mean predicted lDDT over atoms, and over the representative (Ca) atoms alone;
  ptm / iptm   the two pTM reductions the ranking score reads. `iptm` is 0.0 by construction
               on a single-chain target (tt_bio.protenix.ConfidenceHead._ptm_iptm, and
               upstream's compute_ptm the same way), which is the point of recording it;
  disorder     the AF3 RASA term, and `has_clash`, the other two ranking inputs;
  rank_score   0.8*iptm + 0.2*ptm + 0.5*disorder - 100*has_clash, what actually selects;
  pae / pde    mean expected error in Angstrom over token pairs, plus gPDE (AF3 SI 5.7 Eq 16),
               the contact-weighted PDE upstream reports and we currently do not;
  resolved     mean P(experimentally resolved), the fourth head output.

Two arms, one lever, exactly D1: the trunk Pairformer's `scale_pair_bias`. `--arm ship`
forces it back to False at the trunk site ONLY (the confidence head's own Pairformer is left
at the branch value, so the lever is the trunk's token pair bias and nothing else); `--arm
fix` is the branch as it stands. The patch is applied to `openfold3_trunk.Pairformer`, the
name that module resolved at import, so it cannot reach the confidence head by accident, and
the resolved flag is asserted and printed before the fold.

    python3 perf/of3t_confhead/rank_fold.py --arm fix --seed 1 --card 2 \
        --msa-dir ~/of3t_confhead_msa --out-root /tmp/of3t/of3t-confhead/fold
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=("ship", "fix"), required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--sampling-steps", type=int, default=200)
ap.add_argument("--target", default="examples/ubq.yaml")
ap.add_argument("--gt", default="examples/ground_truth_structures/ubiquitin.pdb")
ap.add_argument("--msa-dir", default=os.path.expanduser("~/of3t_confhead_msa"))
ap.add_argument("--out-root", default="/tmp/of3t/of3t-confhead/fold")
ap.add_argument("--card", type=int, default=None, help="tt-smi index to sample AICLK from")
ap.add_argument("--tt-smi", default=os.path.expanduser("~/.local/bin/tt-smi"))
args = ap.parse_args()

import gemmi                                                          # noqa: E402
import torch                                                          # noqa: E402

# ---------------------------------------------------------------- the D1 arm lever
import tt_bio.openfold3_trunk as of3_trunk                            # noqa: E402

_built = []
_Pairformer = of3_trunk.Pairformer


def _trunk_pairformer(*a, **kw):
    """Trunk-site Pairformer with the arm's `scale_pair_bias`. Records what it resolved."""
    if args.arm == "ship":
        kw["scale_pair_bias"] = False
        kw["tri_att_scale_pair_bias"] = False
    _built.append((kw.get("scale_pair_bias"), kw.get("tri_att_scale_pair_bias")))
    return _Pairformer(*a, **kw)


of3_trunk.Pairformer = _trunk_pairformer

# ---------------------------------------------------------------- per-sample capture
import tt_bio.openfold3_confidence as of3_conf                        # noqa: E402
import tt_bio.openfold3_fold as of3_fold                              # noqa: E402

_raw = []
_orig_head_forward = of3_conf.OF3ConfidenceHead.forward
_orig_confidence = of3_fold.OF3Fold._confidence
RECORDS = []


def _head_forward(self, *a, **kw):
    out = _orig_head_forward(self, *a, **kw)
    _raw.append(out)
    return out


def _expected(logits, bin_min, bin_max, no_bins):
    """AF3 probs_to_expected_error: E[error] under the binned head distribution."""
    width = (bin_max - bin_min) / no_bins
    centers = bin_min + width * (torch.arange(no_bins, dtype=torch.float32) + 0.5)
    return (torch.softmax(logits.float(), -1) * centers).sum(-1)


def _gpde(pde, distogram_logits):
    """AF3 SI 5.7 Eq 16: PDE averaged with the predicted contact probability as weight."""
    probs = torch.softmax(distogram_logits.float(), -1)
    ends = torch.linspace(2, 22, probs.shape[-1] + 1)[1:]
    contact = probs[..., ends <= 8.0].sum(-1)
    return float((contact * pde).sum() / (contact.sum() + 1e-8))


def _ca(path):
    st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    return [r.find_atom("CA", "*").pos for c in st[0] for r in c
            if r.find_atom("CA", "*") is not None]


GT = _ca(args.gt)


def _rmsd_to_gt(coords):
    pred = [gemmi.Position(*[float(v) for v in xyz]) for xyz in coords]
    n = min(len(pred), len(GT))
    return gemmi.superpose_positions(pred[:n], GT[:n]).rmsd, n


def _confidence(self, sample, si_input, si_trunk, zij_trunk, aux):
    out = _orig_confidence(self, sample, si_input, si_trunk, zij_trunk, aux)
    raw = _raw[-1]
    ca_idx = aux["representative_atom_indices"].long()
    rmsd, n_ca = _rmsd_to_gt(sample[ca_idx].detach().cpu().numpy())
    pae = _expected(raw["pae_logits"], 0, 32, raw["pae_logits"].shape[-1])
    pde = _expected(raw["pde_logits"], 0, 32, raw["pde_logits"].shape[-1])
    resolved = torch.softmax(raw["experimentally_resolved_logits"].float(), -1)[..., 1]
    plddt_atom = out["plddt_atom"].detach().float()
    RECORDS.append({
        "sample": len(RECORDS), "rmsd_ca": rmsd, "n_ca": n_ca,
        "plddt": float(plddt_atom.mean()), "plddt_ca": float(plddt_atom[ca_idx].mean()),
        "plddt_min": float(plddt_atom.min()),
        "ptm": out["ptm"], "iptm": out["iptm"], "disorder": out["disorder"],
        "has_clash": out["has_clash"], "rank_score": out["ranking_score"],
        "pae_mean": float(pae.mean()), "pae_offdiag_mean": float(
            (pae.sum() - pae.diagonal().sum()) / max(1, pae.numel() - pae.shape[0])),
        "pde_mean": float(pde.mean()),
        "gpde": _gpde(pde, raw["distogram_logits"]),
        "resolved_mean": float(resolved.mean()),
    })
    return out


of3_conf.OF3ConfidenceHead.forward = _head_forward
of3_fold.OF3Fold._confidence = _confidence

# ---------------------------------------------------------------- AICLK, sampled DURING
clk = []
stop = threading.Event()


def _sample_clock():
    while not stop.wait(10.0):
        try:
            raw = subprocess.run([args.tt_smi, "-s"], capture_output=True, text=True,
                                 timeout=30).stdout
            t = json.loads(raw)["device_info"][args.card]["telemetry"]
            clk.append(int(t["aiclk"]))
        except Exception:
            pass


if args.card is not None and os.path.exists(args.tt_smi):
    threading.Thread(target=_sample_clock, daemon=True).start()

# ---------------------------------------------------------------- the fold
out_dir = os.path.join(args.out_root, f"{args.arm}_s{args.seed}")
os.makedirs(out_dir, exist_ok=True)
argv = ["predict", args.target, "--model", "openfold3", "--out_dir", out_dir,
        "--seed", str(args.seed), "--diffusion_samples", str(args.samples),
        "--sampling_steps", str(args.sampling_steps),
        "--use_msa_server", "--msa_dir", args.msa_dir]

from tt_bio.main import cli                                           # noqa: E402

t0 = time.time()
try:
    cli.main(argv, standalone_mode=False)
finally:
    stop.set()
    wall = time.time() - t0

assert _built, "trunk Pairformer was never constructed -- the arm patch did not take"
resolved_flags = set(_built)
assert len(resolved_flags) == 1, f"trunk built with mixed flags: {resolved_flags}"
scale, tri = next(iter(resolved_flags))
expect = False if args.arm == "ship" else True
assert scale is expect, f"arm {args.arm} wanted scale_pair_bias={expect}, got {scale}"

report = {
    "arm": args.arm, "seed": args.seed, "samples": args.samples,
    "trunk_scale_pair_bias": scale, "trunk_tri_att_scale_pair_bias": tri,
    "n_trunk_pairformers": len(_built), "wall_s": round(wall, 2),
    "aiclk_during": {"n": len(clk), "min": min(clk) if clk else None,
                     "max": max(clk) if clk else None,
                     "median": sorted(clk)[len(clk) // 2] if clk else None},
    "ground_truth": args.gt, "n_ca_gt": len(GT),
    "per_sample": RECORDS,
}
path = os.path.join(out_dir, "samples.json")
json.dump(report, open(path, "w"), indent=1)
print(f"\n[{args.arm} s{args.seed}] trunk scale_pair_bias={scale} tri={tri}  "
      f"wall {wall:.1f}s  aiclk {report['aiclk_during']}")
for r in RECORDS:
    print(f"  sample {r['sample']}: rmsd {r['rmsd_ca']:6.3f} A  plddt {r['plddt']:.4f}  "
          f"ptm {r['ptm']:.4f}  iptm {r['iptm']:.4f}  disorder {r['disorder']:.4f}  "
          f"score {r['rank_score']:.4f}  pae {r['pae_mean']:5.2f}  gpde {r['gpde']:5.2f}")
print("wrote", path)
