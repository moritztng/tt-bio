"""Dump one deterministic RFD3 device forward so two trees can be compared bit-for-bit.

`verify_batch_invariance.py` proves a tree is self-consistent across batch size. It cannot
see a change that moves BOTH the D=1 and the D=8 answer by the same amount -- which is
exactly what a matmul program-config change could do. Run this in the change tree and in a
`git archive` of its parent, then diff the two dumps: a perf change that claims
bit-exactness must produce maxabs 0.0.

  PYTHONPATH=<tree> python3 <tree>/scripts/rfd3_port/dump_forward_for_crosstree_parity.py \
      --out /tmp/fwd_<tree>.pt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contigs", nargs="+",
                    default=["A1-10,20,A31-40", "A1-10,130,A31-40"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--pdb", type=Path, default=PDB)
    args = ap.parse_args()

    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification

    ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                    map_location="cpu", weights_only=True)
    dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                     map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti)
    dm = build_diffusion_module(dmw)

    dump = {}
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
        for D in args.batches:
            XD = X1.expand(D, -1, -1).contiguous()
            with torch.no_grad():
                out = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
            dump[f"{contig}|D{D}"] = out.clone()
            print(f"{contig} L={L} D={D}: sum={out.double().sum().item():.12e} "
                  f"finite={bool(torch.isfinite(out).all())}", flush=True)

    torch.save(dump, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
