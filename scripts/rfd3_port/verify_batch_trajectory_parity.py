"""Batched trajectory parity at arbitrary D and fixture.

Generalizes the original B=2 batch-invariance spike: verifies the
RFD3 device forward is batch-invariant and that a full stochastic trajectory
with per-element generators reproduces each element's standalone seeded
trajectory, at batch size D and any contig/spec fixture. Required before raising
the production batch default above 8, or the atom-count clamp that decides which
batch a large design actually gets (`_BATCH_ATOM_PAIR_BUDGET`, rfd3_design.py) --
batch exactness is a property of the whole (M, K, N, D) tuple, so clearing it at
419 atoms says nothing about 3359.

Checks:
  1. Single-forward batch invariance: B=1 vs B=D elem0 PCC (cross-batch numerics).
  2. Identical-input isolation: B=D elem0 vs elem1 maxabs (no cross-contamination).
  3. Full trajectory: B=D batched (per-element generators) vs D standalone seeded
     runs -> per-element PCC; min must clear the bf16 noise floor (>0.99).

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_batch_trajectory_parity.py \
      [--batch 16] [--contig "A1-10,20,A31-40" | --spec fixture.json] [--timesteps 8]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--contig", default="A1-10,20,A31-40")
    ap.add_argument("--spec", type=Path,
                    help="JSON InputSpecification; overrides --pdb/--contig")
    ap.add_argument("--timesteps", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    return ap.parse_args()


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main():
    args = parse_args()
    D = args.batch
    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

    if args.spec:
        data = json.loads(args.spec.read_text())
        path = Path(data["input"])
        data["input"] = str(path if path.is_absolute() else (args.spec.parent / path).resolve())
        fixture = f"spec={args.spec}"
    else:
        data = {"input": str(args.pdb), "contig": args.contig}
        fixture = f"pdb={args.pdb} contig={args.contig!r}"
    spec = InputSpecification.from_dict(data)
    spec.validate()
    f = featurize(data["input"], spec)
    f = {
        k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in f.items()
    }
    L = f["ref_pos"].shape[0]
    ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti)
    dev_dm = build_diffusion_module(dm)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

    seeds = args.seeds if args.seeds else list(range(42, 42 + D))
    assert len(seeds) == D, f"--seeds must provide exactly {D} seeds"

    print(f"fixture: {fixture} L={L} D={D} timesteps={args.timesteps}")

    # 1 + 2. Single-forward batch invariance + identical-input isolation
    with torch.no_grad():
        X1 = torch.randn(1, L, 3) * 16.0
        XD = X1.expand(D, -1, -1).contiguous()
        t1 = torch.tensor([8.0])
        tD = torch.full((D,), 8.0)
        out1 = dev_dm(X_noisy_L=X1, t=t1, f=f, **init)
        outD = dev_dm(X_noisy_L=XD, t=tD, f=f, **init)
    p_invar = pcc(out1["X_L"][0], outD["X_L"][0])
    ma_iso = (outD["X_L"][0] - outD["X_L"][1]).abs().max().item()
    fin = torch.isfinite(outD["X_L"]).all().item()
    print(f"single-fwd: B=1 vs B={D} elem0 PCC={p_invar:.6f}; elem0 vs elem1 maxabs={ma_iso:.3e} finite={fin}")

    # 3. Full stochastic trajectory: batched (per-element generators) vs standalone
    sampler = RFD3Sampler(num_timesteps=args.timesteps)
    coord = f["motif_pos"].float().unsqueeze(0)
    fixed = f["is_motif_atom_with_fixed_coord"]
    with torch.no_grad():
        batched, _ = sampler.sample(
            dev_dm, D, L, coord, f, init, fixed,
            generator=[torch.Generator().manual_seed(s) for s in seeds],
        )
        standalone = []
        for s in seeds:
            v, _ = sampler.sample(
                dev_dm, 1, L, coord, f, init, fixed,
                generator=torch.Generator().manual_seed(s),
            )
            standalone.append(v[0])
    traj_pcc = [pcc(batched[i], standalone[i]) for i in range(D)]
    traj_ma = [(batched[i] - standalone[i]).abs().max().item() for i in range(D)]
    worst = min(traj_pcc)
    print("trajectory PCC (batched vs standalone): " + ", ".join(
        f"seed={s} pcc={c:.6f} maxabs={m:.3e}" for s, c, m in zip(seeds, traj_pcc, traj_ma)
    ))
    print(f"min trajectory PCC = {worst:.6f}")
    ok = (p_invar > 0.999 and ma_iso < 1e-4 and worst > 0.99 and fin)
    print("PARITY", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
