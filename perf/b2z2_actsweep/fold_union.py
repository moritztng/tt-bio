#!/usr/bin/env python3
"""The fold A/B for the union of every site the fused-activation threshold gate fires at.

`sweep.py` prices each site on its own. Sites that share a kernel or a program cache entry are not
independent, so the fold number has to come from switching them all on together and timing the whole
fold, never from adding up the per-site savings. That is what this does.

At 512 aa the gate fires at exactly the four `Transition` fc1 shapes (4096, 2048, 768 and 768
output tiles, all silu, all L1) and declines the DiT sigmoid gate (384 tiles to DRAM) and the
atom->token relu. So the device behaviour of the `split` arm here is identical to what
`TT_BIO_UNFUSED_SILU` produced before the gate replaced it, and `b2z2-unfused-silu-recover`'s
four-seed per-pseudo-domain parity read applies to it unchanged. The point of this run is the
ratio on this build, plus the A/A floor that makes it interpretable.

Arms, in ONE process so they share a device, a program cache and a model load:

    base     the shipped default: ttnn.linear(activation="silu")
    split    TT_BIO_SPLIT_FUSED_ACT, silu as its own op wherever the threshold is met

The gate is module-level in `tt_bio.tenstorrent` (it picks a device program, so it is read at build
time, not per call), so it is set on the module here rather than in the environment, and asserted
absent from the environment so a pinned value cannot silently serve every arm. The last fold
repeats the first: that repeat is the A/A floor and must come back byte-identical or nothing else
here is interpretable.

    fold_union.py --out <json> --cifdir <dir> [--plan base:0,1,2,3;split:0,1,2,3]
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

# arm -> _SPLIT_FUSED_ACT. The threshold gate decides per call site from the shape, dtype and
# memory config, so the arm is one switch and the union of sites it fires at is what gets measured.
ARMS = {"base": False, "split": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--plan", default="base:0,1,2,3;split:0,1,2,3")
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
    assert "TT_BIO_SPLIT_FUSED_ACT" not in os.environ, \
        "TT_BIO_SPLIT_FUSED_ACT may not be pinned; the arm is set per fold"

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

    work = Path(tempfile.mkdtemp(prefix="b2z2-actsweep-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-actsweep", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    # The gate has to be observed firing, not assumed: a threshold that silently declines every
    # site would produce two identical arms and a ratio of 1.000 that reads as "no win" rather
    # than as "the lever never ran".
    _silu, _sig = ttnn.silu, ttnn.sigmoid
    fired = [0]

    def _count(orig):
        def w(*aa, **kk):
            fired[0] += 1
            return orig(*aa, **kk)
        return w
    ttnn.silu, ttnn.sigmoid = _count(_silu), _count(_sig)

    def fold(arm, seed, target, keep):
        T._SPLIT_FUSED_ACT = ARMS[arm]
        fired[0] = 0
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
                "split_ops": fired[0],
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
            print(f"  {size} {tag:12s} {r['fold_s']:7.3f}s split_ops={r['split_ops']:<6d} "
                  f"sha={r['sha256']} plddt={r['plddt']}", flush=True)
            dump()

    aa = [r for r in out["runs"] if r["tag"].startswith(f"{first}-s{plan[first][0]}")]
    out["aa_floor_identical"] = {
        size: len({r["sha256"] for r in aa if r["target"].endswith(size)}) == 1 for size in sizes}
    dump()
    print(json.dumps(out["aa_floor_identical"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
