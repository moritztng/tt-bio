#!/usr/bin/env python3
"""of3t-confpfe: auxgrad's capture boundary rewritten as a conf_DIAG-format dump, with swaps.

    boundary_dump.py --boundary boundary_aux_heads.pt --step conf_DIAG.pt --out-dir D

Writes `D/conf_boundary_<swap>.pt` with `inputs` and `repr_x` only (head_probe.py adds the device
outputs). `none` is upstream's captured boundary as auxgrad fed it; each other swap replaces one
input by the training step's own value from conf_DIAG.pt, so the probe finds which input
separates auxgrad's 5e-3 from the step's 0.23.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_fullstep64"))
import ref_step  # noqa: E402,F401  (puts upstream on sys.path the same way)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", type=Path, required=True)
    ap.add_argument("--step", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    from openfold3.core.utils.atomize_utils import get_token_representative_atoms

    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    kw, pos = B["inputs"]["kwargs"], B["inputs"]["args"]
    batch = kw["batch"] if "batch" in kw else pos[0]
    si_input = kw["si_input"] if "si_input" in kw else pos[1]
    outd = kw["output"] if "output" in kw else pos[2]
    si_trunk, zij_trunk = outd["si_trunk"], outd["zij_trunk"]
    xpred = outd["atom_positions_predicted"].to(dtype=si_trunk.dtype)
    repr_x, _ = get_token_representative_atoms(batch=batch, x=xpred, atom_mask=batch["atom_mask"])
    n = int(si_trunk.shape[-2])
    bnd = {"s_input": si_input.reshape(n, -1).float(), "s_trunk": si_trunk.reshape(n, -1).float(),
           "z_trunk": zij_trunk.reshape(n, n, -1).float()}
    rb = repr_x.reshape(-1, n, 3)[0].double()
    st = torch.load(a.step, weights_only=False)
    sti = {k: v.reshape(bnd[k].shape).float() for k, v in st["inputs"].items()}
    rs = st["repr_x"][:n].double()
    swaps = {"none": (bnd, rb), "s_trunk": ({**bnd, "s_trunk": sti["s_trunk"]}, rb),
             "z_trunk": ({**bnd, "z_trunk": sti["z_trunk"]}, rb),
             "s_input": ({**bnd, "s_input": sti["s_input"]}, rb),
             "repr_x": (bnd, rs)}
    for name, (inp, r) in swaps.items():
        torch.save({"inputs": inp, "repr_x": r,
                    "token_mask": batch["token_mask"].reshape(-1)[:n].float()},
                   a.out_dir / f"conf_boundary_{name}.pt")
        print(f"wrote conf_boundary_{name}.pt", flush=True)
    for k in bnd:
        d = (bnd[k][:56] - sti[k][:56]).norm() / bnd[k][:56].norm()
        print(f"boundary vs step {k}: rel {float(d):.4e}", flush=True)
    print("repr_x rms", float(((rb[:56] - rs[:56]) ** 2).sum(-1).mean().sqrt()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
