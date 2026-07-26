"""Gate: the RFD3 design forward must be bit-identical across batch size.

Batching D designs into one forward is only parity-preserving if lane i of a
batched forward returns exactly what a standalone forward returns. This drives
identical data through every lane and requires maxabs == 0 against the B=1
output, at every batch size and every fixture size given.

It catches the failure mode that `core_grid=` reintroduces: ttnn derives a
linear's program config from M = batch * rows, so its K-blocking (and therefore
its bf16 rounding) changes with the batch. See BATCH_INVARIANT_GRID in
tt_bio/rfd3.py, and probe_callsites.py to re-derive which linears need it.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_batch_invariance.py \
      [--batches 2 4 8 16] [--contigs "A1-10,20,A31-40" "A1-10,130,A31-40"]
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
    ap.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--contigs", nargs="+",
                    default=["A1-10,20,A31-40", "A1-10,130,A31-40"])
    ap.add_argument("--pdb", type=Path, default=PDB)
    return ap.parse_args()


def main():
    args = parse_args()
    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification

    ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                    map_location="cpu", weights_only=True)
    dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                     map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti)
    dm = build_diffusion_module(dmw)

    failures = 0
    for contig in args.contigs:
        spec = InputSpecification.from_dict({"input": str(args.pdb), "contig": contig})
        spec.validate()
        f = featurize(str(args.pdb), spec)
        f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
             for k, v in f.items()}
        L = f["ref_pos"].shape[0]
        with torch.no_grad():
            init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v)
                           for k, v in f.items()})
        torch.manual_seed(0)
        X1 = torch.randn(1, L, 3) * 16.0
        with torch.no_grad():
            out1 = dm(X_noisy_L=X1, t=torch.tensor([8.0]), f=f, **init)["X_L"]
        for D in args.batches:
            XD = X1.expand(D, -1, -1).contiguous()
            with torch.no_grad():
                outD = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
            vs1 = max(float((outD[i] - out1[0]).abs().max()) for i in range(D))
            iso = max(float((outD[i] - outD[0]).abs().max()) for i in range(D))
            ok = vs1 == 0.0 and iso == 0.0 and bool(torch.isfinite(outD).all())
            failures += not ok
            print(f"contig={contig!r} L={L} D={D:2d}  maxabs vs B=1 = {vs1:.6e}  "
                  f"lane isolation = {iso:.6e}  {'OK' if ok else 'FAIL'}", flush=True)

    print("BATCH INVARIANCE", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
