"""Localize the ESMFold2 device-vs-reference coordinate gap (GB1 vs trp-cage control).

Follow-up to scripts/esmfold2_e2e_parity.py. That harness found GB1's (L=56)
device-vs-reference Kabsch-RMSD gap does NOT resolve into the sampler noise floor
(X/floor ~= 2.0 on RMSD, ~3.6 on 1-coord-dm-PCC) while trp-cage's (L=20) does
(~1.18). The sampler-independent heads (pLDDT/distogram/pTM) agree to >0.999, so
the divergence lives entirely in the diffusion structure head's sampled coords.

This script re-uses the SAME e2e pipeline (shared featurization + shared ttnn
ESMC-6B LM states; only the folding port differs between backends) and the SAME
R/D/X noise-floor framing, but adds Kabsch-aligned PER-RESIDUE deviation profiles
to answer: is the gap a localized systematic port divergence (a specific region /
module) or diffuse global sampler diversity the reference itself also shows?

Two phases, coords dumped to disk first so analysis is re-runnable after a timeout:
  phase "fold"     -> fold each protein at seeds on BOTH backends, dump per-atom
                      coords + atom_to_token + per-residue pLDDT to <out>/*.npz
  phase "analyze"  -> pure-numpy: per-residue Kabsch-aligned RMSD for the R/D/X
                      legs, plus the diagnostic X_r / max(R_r,D_r) profile.

Usage (fold, on card 1):
  PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=1 \
    /home/ttuser/tt-bio-dev/env/bin/python scripts/esmfold2_gb1_localize.py fold \
      --proteins gb1,trpcage --seeds 0,1,2 --out /tmp/ef2_gb1_loc
  ... (analyze re-uses the dumps, no device needed)
  python scripts/esmfold2_gb1_localize.py analyze --out /tmp/ef2_gb1_loc
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np


# GB1 (1PGB) is a beta1-beta2 hairpin, central alpha-helix, beta3-beta4 hairpin.
# Residue ranges (1-based, DSSP on 1PGB, mapped onto the 56-mer used here) used
# only to LABEL the per-residue profile, never to drive the verdict.
GB1_SSE = [
    ("beta1", 1, 8), ("loop12", 9, 9), ("beta2", 10, 17), ("loop2a", 18, 22),
    ("helix", 23, 36), ("loopa3", 37, 40), ("beta3", 41, 46), ("turn34", 47, 51),
    ("beta4", 52, 55), ("Cterm", 56, 56),
]


def _load_common():
    import torch
    torch.set_grad_enabled(False)
    return torch


# ---------------------------------------------------------------------------
# fold phase
# ---------------------------------------------------------------------------
def fold(args):
    torch = _load_common()
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from esmfold2_e2e_parity import (PROTEINS, build_features, run_forward,
                                     _FORWARD_KEYS)  # noqa: F401
    from tt_bio import tenstorrent
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2_common import compute_lm_hidden_states
    from tt_bio.esmfold2_runtime import _ESMCAdapter, patch_esmfold2

    names = [n.strip() for n in args.proteins.split(",") if n.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)

    esmc = _ESMCAdapter(args.esmc_repo, persistent=True)
    esmc.preload()
    print("loading torch reference model ...", flush=True)
    ref_model = ESMFold2Model.from_pretrained(args.esmfold2_repo, load_esmc=False).eval()
    print(f"loading ttnn model (fast={args.fast}) ...", flush=True)
    tenstorrent.set_fast_mode(args.fast)
    tt_model = ESMFold2Model.from_pretrained(args.esmfold2_repo, load_esmc=False).eval()
    patch_esmfold2(tt_model, esmc_repo=args.esmc_repo)
    tt_model._esmc = esmc

    for name in names:
        seq = PROTEINS[name]
        print(f"\n=== {name} (L={len(seq)}) seeds={seeds} ===", flush=True)
        feats = build_features(seq, args.feature_seed, ref_model.device)
        lm_hs = compute_lm_hidden_states(
            esmc, feats["input_ids"], feats["asym_id"], feats["residue_index"],
            feats["mol_type"], feats["token_attention_mask"])
        atom_mask = feats["atom_attention_mask"].float()
        if atom_mask.dim() == 1:
            atom_mask = atom_mask.unsqueeze(0)
        # atom_to_token: one-hot [.,n_atom,n_tok] (3D) or index [.,n_atom] (<=2D).
        a2t = feats["atom_to_token"]
        a2t_idx = (a2t[0].argmax(-1) if a2t.dim() == 3 else a2t.reshape(-1))
        a2t_idx = a2t_idx.long().cpu().numpy()

        for backend, model in (("ref", ref_model), ("dev", tt_model)):
            for s in seeds:
                print(f"  {backend} seed={s} ...", flush=True)
                out = run_forward(model, feats, lm_hs, loops=args.loops,
                                  steps=args.steps, samples=1, seed=s)
                coords = out["sample_atom_coords"][0].float().cpu().numpy()  # [n_atom,3]
                plddt = out["plddt"].float().cpu().numpy().reshape(-1)        # per-residue
                np.savez(os.path.join(args.out, f"{name}_{backend}_seed{s}.npz"),
                         coords=coords, atom_mask=atom_mask[0].cpu().numpy(),
                         a2t=a2t_idx, plddt=plddt, L=len(seq))
        print(f"  dumped {name}", flush=True)
    print(f"\nwrote coord dumps to {args.out}", flush=True)


# ---------------------------------------------------------------------------
# analyze phase (pure numpy, no torch/device)
# ---------------------------------------------------------------------------
def _kabsch_align(a, b, w):
    """Rigid-align a onto b with per-point weights w (all [n,3]/[n]). Returns aligned a."""
    w = w / w.sum()
    ca = (a * w[:, None]).sum(0)
    cb = (b * w[:, None]).sum(0)
    A = a - ca
    B = b - cb
    H = (A * w[:, None]).T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return (A @ R.T) + cb


def _per_residue_dev(a, b, mask, a2t, n_res):
    """Whole-structure weighted Kabsch align of a onto b, then per-residue RMSD."""
    m = mask > 0.5
    aligned = _kabsch_align(a[m], b[m], mask[m])
    d2 = ((aligned - b[m]) ** 2).sum(-1)          # per masked-atom squared dev
    tok = a2t[m]
    out = np.full(n_res, np.nan)
    for r in range(n_res):
        sel = tok == r
        if sel.any():
            out[r] = np.sqrt(d2[sel].mean())
    whole = float(np.sqrt(d2.mean()))
    return out, whole


def _leg_profiles(dumps, pairs, n_res):
    prof, whole = [], []
    for da, db in pairs:
        p, w = _per_residue_dev(da["coords"], db["coords"], da["atom_mask"], da["a2t"], n_res)
        prof.append(p); whole.append(w)
    return np.array(prof), np.array(whole)


def analyze(args):
    files = [f for f in os.listdir(args.out) if f.endswith(".npz")]
    proteins = sorted({f.split("_")[0] for f in files})
    report = {}
    lines = []
    for name in proteins:
        seeds = sorted({int(f.split("seed")[1].split(".")[0])
                        for f in files if f.startswith(name + "_")})
        load = lambda be, s: dict(np.load(os.path.join(args.out, f"{name}_{be}_seed{s}.npz")))
        ref = {s: load("ref", s) for s in seeds}
        dev = {s: load("dev", s) for s in seeds}
        n_res = int(ref[seeds[0]]["L"])

        X_pairs = [(dev[s1], ref[s2]) for s1 in seeds for s2 in seeds]
        R_pairs = [(ref[s1], ref[s2]) for s1, s2 in itertools.combinations(seeds, 2)]
        D_pairs = [(dev[s1], dev[s2]) for s1, s2 in itertools.combinations(seeds, 2)]
        Xp, Xw = _leg_profiles(dev, X_pairs, n_res)
        Rp, Rw = _leg_profiles(ref, R_pairs, n_res)
        Dp, Dw = _leg_profiles(dev, D_pairs, n_res)

        Xr = np.nanmean(Xp, 0); Rr = np.nanmean(Rp, 0); Dr = np.nanmean(Dp, 0)
        floor_r = np.maximum(Rr, Dr)
        ratio_r = Xr / np.maximum(floor_r, 1e-6)
        plddt = np.mean([ref[s]["plddt"] for s in seeds] + [dev[s]["plddt"] for s in seeds], 0)
        if plddt.shape[0] != n_res:  # guard: some heads emit [L] already
            plddt = plddt[:n_res]

        report[name] = {
            "L": n_res, "seeds": seeds,
            "whole": {"X": float(Xw.mean()), "R": float(Rw.mean()), "D": float(Dw.mean()),
                      "X_over_floor": float(Xw.mean() / max(Rw.mean(), Dw.mean()))},
            "systematic_offset_est_A": float(np.sqrt(max(Xw.mean()**2 - max(Rw.mean(), Dw.mean())**2, 0.0))),
            "per_residue": {"X": Xr.tolist(), "R": Rr.tolist(), "D": Dr.tolist(),
                            "ratio": ratio_r.tolist(), "plddt": plddt.tolist()},
        }

        lines.append(f"\n## {name} (L={n_res}) seeds={seeds}")
        lines.append(f"whole-structure RMSD  X={Xw.mean():.3f}  R={Rw.mean():.3f}  "
                     f"D={Dw.mean():.3f}  X/floor={Xw.mean()/max(Rw.mean(),Dw.mean()):.2f}")
        lines.append(f"est. systematic dev-vs-ref mean offset = "
                     f"{np.sqrt(max(Xw.mean()**2 - max(Rw.mean(),Dw.mean())**2,0)):.2f} A")
        # correlation of X profile with the reference floor profile: high => same
        # regions are hot in both (intrinsic floppiness); low => device-specific.
        finite = np.isfinite(Xr) & np.isfinite(Rr)
        if finite.sum() > 2:
            cc_XR = float(np.corrcoef(Xr[finite], Rr[finite])[0, 1])
            cc_Xpl = float(np.corrcoef(Xr[finite], plddt[finite])[0, 1])
            lines.append(f"corr(X_res, R_res) = {cc_XR:+.3f}   "
                         f"corr(X_res, pLDDT) = {cc_Xpl:+.3f}")
            report[name]["corr_X_R"] = cc_XR
            report[name]["corr_X_plddt"] = cc_Xpl
        # top divergent residues
        order = np.argsort(-np.nan_to_num(Xr))
        lines.append("top-8 divergent residues (1-based res: X R D ratio pLDDT):")
        for r in order[:8]:
            lines.append(f"  res {r+1:3d}:  X={Xr[r]:5.2f}  R={Rr[r]:5.2f}  "
                         f"D={Dr[r]:5.2f}  X/floor={ratio_r[r]:4.1f}  pLDDT={plddt[r]:.2f}")
        if name == "gb1":
            lines.append("by secondary-structure element (mean X / mean R / mean ratio):")
            for label, lo, hi in GB1_SSE:
                sl = slice(lo - 1, hi)
                lines.append(f"  {label:7s} res {lo:2d}-{hi:2d}:  "
                             f"X={np.nanmean(Xr[sl]):5.2f}  R={np.nanmean(Rr[sl]):5.2f}  "
                             f"ratio={np.nanmean(ratio_r[sl]):4.1f}  "
                             f"pLDDT={np.nanmean(plddt[sl]):.2f}")

    text = "\n".join(lines)
    print(text)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.report}")
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    f = sub.add_parser("fold")
    f.add_argument("--proteins", default="gb1,trpcage")
    f.add_argument("--seeds", default="0,1,2")
    f.add_argument("--steps", type=int, default=20)
    f.add_argument("--loops", type=int, default=3)
    f.add_argument("--feature_seed", type=int, default=7)
    f.add_argument("--fast", action="store_true")
    f.add_argument("--esmfold2_repo", default="biohub/ESMFold2")
    f.add_argument("--esmc_repo", default="biohub/ESMC-6B")
    f.add_argument("--out", default="/tmp/ef2_gb1_loc")
    f.set_defaults(func=fold)
    a = sub.add_parser("analyze")
    a.add_argument("--out", default="/tmp/ef2_gb1_loc")
    a.add_argument("--report", default="")
    a.set_defaults(func=analyze)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
