#!/usr/bin/env python3
"""Is the MSA softmax knife-edge at bf16 resolution, or did 0.70.1 break it?

The op trace puts the first 0.69.0/0.70.1 disagreement at the MSA module's pair-weighted-averaging
softmax, [8, 320, 320] bf16 (`tt_bio/tenstorrent.py:9591`), and the disagreement grows smoothly
from 3.3e-4 there to 1.0 by call 557 with no step anywhere: chaotic amplification, not a second
broken op. But `softmax_probe.py` scores BOTH stacks within 2e-3 of a float64 reference at that
exact call's shape, dtype and compute kernel config, so neither softmax is wrong.

That leaves one question, and it decides the attribution. Perturb the pin's OWN output at that call
by the smallest amount bf16 can express -- plus or minus one ULP, chosen per element from a seeded
generator -- and fold. Any two bf16 implementations of the same reduction may differ by at least
this much, so:

  fold survives  ->  0.70.1 does something worse than one ULP at this call and the probe missed the
                     regime; keep hunting on their side
  fold collapses ->  the model is knife-edge here. The stack change only tripped it, and the
                     sensitivity is ours to fix or to accept

    ulp_perturb.py --out folds.json --cifdir cif [--shape 8,320,320] [--eps-ulp 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--shape", default="8,320,320", help="only perturb softmax outputs of this shape")
    ap.add_argument("--ulp", type=int, default=1, help="how many bf16 ULPs, 0 = control")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--coherent", action="store_true",
                    help="step every element the SAME way. Two implementations of one\n                         reduction differ systematically, not randomly, and coherent\n                         per-op error is what accumulates through this model\n                         (af2ig-chained-error-accumulates-coherently-per-op-instrument-blind).")
    a = ap.parse_args()
    want = [int(x) for x in a.shape.split(",")]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a_, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO)
    import ab_flag_levers as AB
    import importlib.metadata as md

    dev = get_device()
    gen = torch.Generator().manual_seed(12345)
    stats = {"hit": 0, "seen": 0}
    orig = ttnn.softmax

    def softmax(*args, **kw):
        out = orig(*args, **kw)
        stats["seen"] += 1
        if a.ulp == 0 or list(getattr(out, "shape", [])) != want:
            return out
        t = ttnn.to_torch(out)
        if t.dtype is not torch.bfloat16:
            return out
        # +-1 ULP in bf16. Softmax outputs are non-negative, so a +1 on the int16 view is the next
        # representable value up and -1 the next down; no sign or subnormal case to handle.
        bits = t.contiguous().view(torch.int16)
        step = (torch.full(bits.shape, 1, dtype=torch.int16) if a.coherent else
                torch.randint(0, 2, bits.shape, generator=gen, dtype=torch.int16) * 2 - 1) * a.ulp
        nz = bits != 0                              # leave exact zeros alone, -1 would wrap them
        bits = torch.where(nz, bits + step, bits)
        pert = bits.view(torch.bfloat16)
        stats["hit"] += 1
        new = ttnn.from_torch(pert, dtype=out.dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                              memory_config=out.memory_config())
        ttnn.deallocate(out)
        return new

    ttnn.softmax = softmax

    work = Path(tempfile.mkdtemp(prefix="ttx-ulp-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    AB._seed_msa(AB.FIX / "cdk2x2_298.yaml", (AB.FIX / "cdk2x2_298.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("ttx-ulp", cfg)

    TAG = f"ulp{a.ulp}{'c' if a.coherent else ''}"
    rec = {"doc": __doc__, "coherent": a.coherent, "ttnn": md.version("ttnn"), "arch": str(dev.arch()),
           "ulp": a.ulp, "shape": want, "runs": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in a.seeds.split(",")]:
        os.environ["TT_BIO_SHARED_DRAW_SEED"] = str(seed)
        cfg["seed"] = seed
        stats["hit"] = stats["seen"] = 0
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(AB.FIX / "cdk2x2_298.yaml", cfg)
        wall = time.perf_counter() - t0
        cif = sorted(struct_dir.glob("*.cif"))[0]
        keep = a.cifdir / f"298_{TAG}-s{seed}"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cif, keep / cif.name)
        r = {"arm": TAG, "tag": f"{TAG}-s{seed}", "seed": seed,
             "target": "cdk2x2_298", "mode": "shared", "fold_s": round(wall, 3),
             "sha256": hashlib.sha256(cif.read_bytes()).hexdigest()[:16],
             "plddt": round(float(metrics.get("plddt", 0.0)), 6),
             "softmax_calls": stats["seen"], "perturbed_calls": stats["hit"]}
        rec["runs"].append(r)
        a.out.write_text(json.dumps(rec, indent=1))
        print(f"  {TAG} s{seed} {r['fold_s']:8.3f}s plddt={r['plddt']} "
              f"sha={r['sha256']} perturbed {r['perturbed_calls']}/{r['softmax_calls']} softmax calls",
              flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
