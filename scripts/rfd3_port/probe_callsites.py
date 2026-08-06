"""Map every non-batch-invariant core_grid linear to its tt_bio/rfd3/model.py call site.

Records (caller line, input shape, weight shape) for each core_grid linear in one
real forward, then replays that shape at B=1 vs B=2/4/8/16 with identical data in
every lane. A call site is listed as BREAKS if any shape it issues is not
bit-exact across batch size.
"""

from __future__ import annotations

import sys
import traceback
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
CONTIGS = ["A1-10,20,A31-40", "A1-10,130,A31-40"]
BATCHES = [2, 4, 8, 16]

_linear = ttnn.linear
sites: dict[tuple, set] = defaultdict(set)


def recording_linear(a, b, *args, **kw):
    if kw.get("core_grid") is not None:
        for fr in reversed(traceback.extract_stack()[:-1]):
            if fr.filename.endswith("tt_bio/rfd3/model.py"):
                sites[(fr.lineno, fr.name, fr.line.strip()[:52])].add(
                    (tuple(a.shape), tuple(b.shape)))
                break
    return _linear(a, b, *args, **kw)


def forward(contig):
    from tt_bio.rfd3.model import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3.featurize import featurize
    from tt_bio.rfd3.input import InputSpecification

    spec = InputSpecification.from_dict({"input": str(PDB), "contig": contig})
    spec.validate()
    f = featurize(str(PDB), spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                    map_location="cpu", weights_only=True)
    dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                     map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti)
    dm = build_diffusion_module(dmw)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = f["ref_pos"].shape[0]
    torch.manual_seed(0)
    ttnn.linear = recording_linear
    with torch.no_grad():
        dm(X_noisy_L=torch.randn(1, L, 3) * 16.0, t=torch.tensor([8.0]), f=f, **init)
    ttnn.linear = _linear
    print(f"recorded contig={contig!r} L={L}: {len(sites)} call sites so far")


def main():
    for contig in CONTIGS:
        forward(contig)

    dev = get_device()
    K = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    print(f"\n{'line':>5s}  {'function':22s} {'source':54s} verdict")
    for key in sorted(sites):
        lineno, fname, src = key
        worst = 0.0
        worst_shape = None
        for (ash, bsh) in sorted(sites[key]):
            host = torch.randn(*ash)
            w = ttnn.from_torch(torch.randn(*bsh) * 0.05, layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16)

            def run(x):
                return ttnn.to_torch(_linear(
                    ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev,
                                    dtype=ttnn.bfloat16),
                    w, compute_kernel_config=K, dtype=ttnn.bfloat16,
                    core_grid=CORE_GRID_MAIN)).float()

            out1 = run(host)
            for D in BATCHES:
                rep = host.expand(D, *([-1] * (host.ndim - 1))).contiguous()
                ma = float((run(rep)[0] - out1[0]).abs().max())
                if ma > worst:
                    worst, worst_shape = ma, (ash, bsh, D)
        verdict = "ok" if worst == 0 else f"BREAKS {worst:.1e} at {worst_shape}"
        print(f"{lineno:5d}  {fname:22s} {src:54s} {verdict}")

    broken = sorted(k[0] for k in sites if any(True for _ in [0]))
    print(f"\n{len(sites)} core_grid call sites across {len(CONTIGS)} fixtures")


if __name__ == "__main__":
    main()
