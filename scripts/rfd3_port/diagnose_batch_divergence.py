"""Localize RFD3's batch-size-dependent numeric divergence.

Two questions, answered with replicated-input controls (identical inputs in every
batch lane, so any lane-to-lane or B=1-vs-B=D difference is pure device numerics):

  A. Does the per-step divergence MAGNITUDE grow with D, or is it flat?
     Flat magnitude means the trajectory min-PCC trend across D that p4 measured
     is an order-statistic of sampling more seeds, not a batch-dependent bug.
  B. WHICH submodule first breaks batch invariance? Every submodule with a
     host-boundary __call__ is wrapped: it runs at B=D, then re-runs on the
     batch-0 slice at B=1, and we report maxabs(out_D[0] - out_1[0]).

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/diagnose_batch_divergence.py \
      [--batches 1 2 4 8] [--localize-batch 8] [--contig "A1-10,20,A31-40"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--localize-batch", type=int, default=8)
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--contig", default="A1-10,20,A31-40")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d > 0 else float("nan")


def maxabs(a, b):
    return float((a.float() - b.float()).abs().max())


def build(args):
    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification

    spec = InputSpecification.from_dict({"input": str(args.pdb), "contig": args.contig})
    spec.validate()
    f = featurize(str(args.pdb), spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                    map_location="cpu", weights_only=True)
    dm_w = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                      map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti)
    dm = build_diffusion_module(dm_w)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    return f, dm, init


def main():
    args = parse_args()
    f, dm, init = build(args)
    L = f["ref_pos"].shape[0]
    print(f"fixture: contig={args.contig!r} L={L}")

    torch.manual_seed(args.seed)
    X1 = torch.randn(1, L, 3) * 16.0
    T_HAT = 8.0

    # --- A. does the divergence magnitude grow with D? ---
    with torch.no_grad():
        out1 = dm(X_noisy_L=X1, t=torch.tensor([T_HAT]), f=f, **init)["X_L"]
    print("\n=== A. single-forward replicated-input invariance vs D ===")
    for D in args.batches:
        XD = X1.expand(D, -1, -1).contiguous()
        with torch.no_grad():
            outD = dm(X_noisy_L=XD, t=torch.full((D,), T_HAT), f=f, **init)["X_L"]
        vs1 = [maxabs(outD[i], out1[0]) for i in range(D)]
        iso = [maxabs(outD[i], outD[0]) for i in range(D)]
        print(f"D={D:2d}  maxabs(elem_i - B1): max={max(vs1):.6e} min={min(vs1):.6e}"
              f"   pcc(elem0,B1)={pcc(outD[0], out1[0]):.8f}"
              f"   lane-isolation maxabs={max(iso):.6e}")

    # --- B. which submodule breaks batch invariance ---
    D = args.localize_batch
    print(f"\n=== B. submodule localization at D={D} (replicated identical input) ===")
    records = []

    def slice_b0(x):
        if torch.is_tensor(x) and x.ndim >= 1 and x.shape[0] == D:
            return x[:1].contiguous()
        return x

    def wrap(owner, name, label):
        orig = getattr(owner, name)

        def probe(*a, **kw):
            outD = orig(*a, **kw)
            a1 = tuple(slice_b0(x) for x in a)
            kw1 = {k: slice_b0(v) for k, v in kw.items()}
            out1 = orig(*a1, **kw1)
            tD = outD if torch.is_tensor(outD) else outD[0]
            t1 = out1 if torch.is_tensor(out1) else out1[0]
            n = sum(1 for r in records if r[0] == label)
            records.append((label, n, maxabs(tD[0], t1[0]), pcc(tD[0], t1[0]),
                            tuple(tD.shape)))
            return outD

        setattr(owner, name, probe)

    wrap(dm, "_downcast_c", "downcast_c")
    wrap(dm, "_downcast_q", "downcast_q")
    wrap(dm, "encoder", "encoder(atom,3blk)")
    wrap(dm, "diffusion_token_encoder", "diffusion_token_encoder")
    wrap(dm, "diffusion_transformer", "token_DiT(18blk)")
    wrap(dm, "decoder", "decoder")
    wrap(dm, "sequence_head", "sequence_head")

    XD = X1.expand(D, -1, -1).contiguous()
    with torch.no_grad():
        dm(X_noisy_L=XD, t=torch.full((D,), T_HAT), f=f, **init)
    for label, n, ma, p, shape in records:
        print(f"  {label:26s} call#{n}  shape={shape}  maxabs={ma:.6e}  pcc={p:.8f}")

    print("\nfirst nonzero submodule = earliest batch-invariance break")


if __name__ == "__main__":
    main()
