"""Which of RFD3's core_grid= linears actually lose batch invariance, and at what D?

Records every (input shape, weight shape) ttnn.linear sees during one real D=1
forward, then replays each shape standalone at B=1/2/4/8 with the SAME data in
every lane. A shape whose lane-0 output differs from its B=1 output is one whose
program config ttnn re-derived because M = B*rows changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
CONTIG = "A1-10,20,A31-40"
BATCHES = [2, 4, 8, 16]

seen: dict[tuple, bool] = {}
_linear = ttnn.linear


def recording_linear(a, b, *args, **kw):
    if kw.get("core_grid") is not None:
        seen[(tuple(a.shape), tuple(b.shape), kw.get("bias") is not None)] = True
    return _linear(a, b, *args, **kw)


def main():
    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification

    spec = InputSpecification.from_dict({"input": str(PDB), "contig": CONTIG})
    spec.validate()
    f = featurize(str(PDB), spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    L = f["ref_pos"].shape[0]
    ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                    map_location="cpu", weights_only=True)
    dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                     map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti)
    dm = build_diffusion_module(dmw)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

    ttnn.linear = recording_linear
    torch.manual_seed(0)
    X1 = torch.randn(1, L, 3) * 16.0
    with torch.no_grad():
        dm(X_noisy_L=X1, t=torch.tensor([8.0]), f=f, **init)
    ttnn.linear = _linear
    print(f"\ncore_grid linear shapes in one D=1 forward: {len(seen)}\n")

    dev = get_device()
    K = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    print(f"{'in_shape':26s} {'w_shape':16s} " + " ".join(f"D={d}" for d in BATCHES))
    n_break = 0
    for (ash, bsh, has_bias) in sorted(seen, key=lambda t: (len(t[0]), t[0])):
        host = torch.randn(*ash)
        w = ttnn.from_torch(torch.randn(*bsh) * 0.05, layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)

        def run(x):
            return ttnn.to_torch(_linear(
                ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16),
                w, compute_kernel_config=K, dtype=ttnn.bfloat16,
                core_grid=CORE_GRID_MAIN)).float()

        out1 = run(host)
        cells, broke = [], False
        for D in BATCHES:
            rep = host.expand(D, *([-1] * (host.ndim - 1))).contiguous()
            ma = float((run(rep)[0] - out1[0]).abs().max())
            broke |= ma > 0
            cells.append("  .  " if ma == 0 else f"{ma:.1e}")
        n_break += broke
        print(f"{str(tuple(ash)):26s} {str(tuple(bsh)):16s} " + " ".join(f"{c:8s}" for c in cells)
              + ("  <-- BREAKS" if broke else ""))
    print(f"\n{n_break}/{len(seen)} core_grid linear shapes break batch invariance")


if __name__ == "__main__":
    main()
