"""The trunk's own perturbation floor at the calls the rows view reaches.

fold_ab's seed floor varies only the diffusion seed. The trunk is deterministic, so every seed
of one arm shares one trunk and the seed floor cannot see how far a rounding-sized change in
the trunk carries. This measures that directly: the served (rank) arm, with each call the view
would take multiplied by (1 + eps * N(0, 1)), eps sized to the measured rank-vs-view gap
(1.2e-3 at the 512 pair projections, `callcheck_pxv2_512.json`), under two noise draws. If
noise of the view's size moves the structure as far as the view does, the view's deviation is
the trunk's floor and not a property of the view.

    python3 perf/bcx_oplin/perturb.py --model protenix-v2 --out perf/bcx_oplin/folds/protenix-v2_512_perturb
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.ops as ops  # noqa: E402
from common import Clock, arm  # noqa: E402
from fold_ab import boltz2_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", type=Path, default=ROOT / "perf/size512/fixtures/cdk2x2_512.yaml")
    ap.add_argument("--a3m", type=Path, default=ROOT / "perf/size512/fixtures/cdk2x2_512.a3m")
    ap.add_argument("--eps", type=float, default=1.2e-3)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import tt_baseline as B
    one_fold, meta, _ = B.build_fold(a.model, Path(tempfile.mkdtemp(prefix="oplin-pt-")),
                                     a.target, a.a3m, extra_cfg=boltz2_cfg(a.model, B))
    real_via2d = ops._via2d
    gen = {}

    def noisy(x, fn, kw=None):
        s = [int(d) for d in x.shape]
        kw = kw or {}
        y = fn(x)
        if (len(s) <= 2 or math.prod(s[:-2]) <= 1 or s[-2] % 32 or x.layout != ttnn.TILE_LAYOUT
                or x.is_sharded() or kw.get("program_config") is not None
                or kw.get("memory_config") is not None or "g" not in gen):
            return y
        yh = ttnn.to_torch(y).float()
        yh = yh * (1 + a.eps * torch.randn(yh.shape, generator=gen["g"]))
        out = ttnn.from_torch(yh.to(torch.bfloat16), dtype=y.dtype, layout=ttnn.TILE_LAYOUT,
                              device=y.device(), memory_config=y.memory_config())
        ttnn.deallocate(y)
        return out

    folds = []

    def fold(tag, seed, draw):
        meta["job_cfg"]["seed"] = seed
        gen.clear()
        if draw is not None:
            gen["g"] = torch.Generator().manual_seed(1000 + draw)
        with arm(False):
            ops._via2d = noisy
            try:
                with Clock() as clk:
                    t, m = one_fold()
            finally:
                ops._via2d = real_via2d
        cif = sorted(Path(meta["struct_dir"]).glob("*.cif"))[0]
        dst = a.out / f"{tag}.cif"
        shutil.copy(cif, dst)
        row = dict(tag=tag, seed=seed, draw=draw, wall_s=round(t, 2), plddt=m.get("plddt"),
                   aiclk=clk.stats(), sha16=hashlib.sha256(dst.read_bytes()).hexdigest()[:16])
        folds.append(row)
        print(json.dumps(row), flush=True)

    for seed in range(a.seeds):
        fold(f"s{seed}_off", seed, None)
        for d in range(a.draws):
            fold(f"s{seed}_noise{d}", seed, d)
    (a.out / "perturb.json").write_text(json.dumps(dict(model=a.model, eps=a.eps, folds=folds),
                                                   indent=1) + "\n")


if __name__ == "__main__":
    main()
