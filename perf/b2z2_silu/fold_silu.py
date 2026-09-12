#!/usr/bin/env python3
"""The folds a per-pseudo-domain reading of the unfused-silu lever needs.

`b2z2-fusion-rebuild` killed `TT_BIO_UNFUSED_SILU` on 0.805 A whole-molecule all-atom RMSD at
512 aa. `cdk2x2_512` is CDK2 fused to its own first 214 residues with no inter-domain interface,
so the hinge between the two pseudo-domains saturates whole-molecule RMSD for any reassociation:
the shipped default itself moves 8.60 A on it. That run also had no stochastic floor -- its
0.000000 A A/A pair is two identical arms, which measures the device's determinism, not the
sampler's spread. So the 0.805 A was read on the wrong axis against the wrong floor.

This run produces what a per-domain reading needs: the same arm at four seeds, so the sampler's own
spread is measured, and every arm at each of those seeds, so each lever's move is priced against it.
Scored by `perf/b2z2_fusebias/score.py`, the shared instrument `b2z2-fusebias-512-parity` built --
per pseudo-domain, CA-lDDT arm-against-arm and against the experimental 1HCL, hinge angle.

Arms, in ONE process so they share a device, a program cache and a model load:

    base      the shipped default: ttnn.linear(activation="silu")
    usilu     TT_BIO_UNFUSED_SILU, silu as its own op on the bf16 projection
    usilu32   the same with TT_BIO_UNFUSED_SILU_FP32: the projection is packed fp32, silu runs on
              it and the result is rounded once, which is the fused arm's arithmetic

Both gates are module-level in `tt_bio.tenstorrent` (they pick a device program, so they are read
at build time, not per call), so they are set on the module here rather than in the environment,
and asserted absent from the environment so a pinned value cannot silently serve every arm.
Different activations are different programs, so the program cache cannot serve one arm the
other's kernel. The last fold repeats the first: that repeat is the A/A floor and must come back
byte-identical or nothing else here is interpretable.

    fold_silu.py --out <json> --cifdir <dir> [--plan base:0,1,2,3;usilu:0,1,2,3;usilu32:0,1]
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402  -- the fixtures, cfg and MSA seeding, unmodified

# arm -> (_UNFUSED_SILU, _UNFUSED_SILU_FP32)
ARMS = {"base": (False, False), "usilu": (True, False), "usilu32": (True, True)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--plan", default="base:0,1,2,3;usilu:0,1,2,3;usilu32:0,1")
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()

    plan = {}
    for part in args.plan.split(";"):
        arm, _, seeds = part.partition(":")
        assert arm in ARMS, f"unknown arm {arm}"
        plan[arm] = [int(s) for s in seeds.split(",")]
    sizes = args.sizes.split(",")

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    for v in ("TT_BIO_UNFUSED_SILU", "TT_BIO_UNFUSED_SILU_FP32"):
        assert v not in os.environ, f"{v} may not be pinned; the arm is set per fold"

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
        "torch": torch.__version__,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "plan": args.plan, "diffusion_trace": False},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-silu-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-silu", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(arm, seed, target, keep):
        T._UNFUSED_SILU, T._UNFUSED_SILU_FP32 = ARMS[arm]
        cfg["seed"] = seed
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": arm, "seed": seed, "target": target.stem, "fold_s": round(wall, 3),
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    first = next(iter(plan))
    for size in sizes:
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        order = []
        for seed in sorted({s for v in plan.values() for s in v}):
            order += [(arm, seed) for arm in plan if seed in plan[arm]]
        order.append((first, plan[first][0]))             # the A/A repeat, last
        seen = set()
        for arm, seed in order:
            tag = f"{arm}-s{seed}" + ("_r1" if (arm, seed) in seen else "")
            seen.add((arm, seed))
            r = fold(arm, seed, target, args.cifdir / f"{size}_{tag}")
            r["tag"] = tag
            out["runs"].append(r)
            print(f"  {size} {tag:12s} {r['fold_s']:7.3f}s sha={r['sha256']} "
                  f"plddt={r['plddt']}", flush=True)
            dump()

    aa = [r for r in out["runs"] if r["tag"].startswith(f"{first}-s{plan[first][0]}")]
    out["aa_floor_identical"] = {
        size: len({r["sha256"] for r in aa if r["target"].endswith(size)}) == 1 for size in sizes}
    dump()
    print(json.dumps(out["aa_floor_identical"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
