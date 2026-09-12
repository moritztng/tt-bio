#!/usr/bin/env python3
"""Is the shared `layer_norm(s)` safe at 512 aa? The folds that question needs.

`BOLTZ2_ADALN_SHARED_SNORM` normalises the token stack's `s` once instead of once per AdaLN and
carries each site's `s_norm.weight` in the projection that consumes it. Exact in real arithmetic,
NOT bit-exact in bf16. At one AdaLN, against an fp32 reference of the same weights and the same
real `s`, the folded form is the more accurate of the two (0.00246 vs 0.00255 relative rms on
`s_scale`, 0.00294 vs 0.00306 on `s_bias`, and the worst layer improves 0.00356 -> 0.00277), so
what this run has to settle is whether a 0.25 %-sized change 24 layers deep reaches the structure.

This is `perf/b2z2_fusebias/fold_seeds.py` with one line changed: the arm is a module global read
per call (`tt_bio.tenstorrent._B2_ADALN_SHARED_SNORM`), not an environment variable, so both arms
share a device, a program cache and a model load exactly as they do there. Everything else --
the fixtures, the MSA seeding, the per-size plan, the A/A repeat as the last fold -- is that row's
and is unmodified, and the CIFs it writes are scored by that row's `score.py`, not by a new one.

    fold_seeds_snorm.py --out <json> --cifdir <dir> [--seeds 0,1] [--sizes 512,298]
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

ARMS = ("off", "on")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--sizes", default="512,298")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sizes = args.sizes.split(",")

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    assert "BOLTZ2_ADALN_SHARED_SNORM" not in os.environ, \
        "the arm under test may not be pinned in the environment; it is set per fold"
    import tt_bio.tenstorrent as T

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
                     "seeds": seeds, "diffusion_trace": False},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-snorm-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-step-layernorm-fusion", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(arm, seed, target, keep):
        T._B2_ADALN_SHARED_SNORM = (arm == "on")
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

    # 512 aa first: it is the question. The 298 aa control follows, on the same device open, so
    # the two sizes are comparable on this box without a second model load.
    for size in sizes:
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        plan = [(arm, s) for s in seeds for arm in ARMS]
        plan.append(("off", seeds[0]))                     # the A/A repeat, last
        seen = set()
        for arm, seed in plan:
            tag = f"{arm}-s{seed}" + ("_r1" if (arm, seed) in seen else "")
            seen.add((arm, seed))
            r = fold(arm, seed, target, args.cifdir / f"{size}_{tag}")
            r["tag"] = tag
            out["runs"].append(r)
            print(f"  {size} {tag:10s} {r['fold_s']:7.3f}s sha={r['sha256']} "
                  f"plddt={r['plddt']}", flush=True)
            dump()

    aa = [r for r in out["runs"] if r["tag"].startswith(f"off-s{seeds[0]}")]
    out["aa_floor_identical"] = {
        size: len({r["sha256"] for r in aa if r["target"].endswith(size)}) == 1 for size in sizes}
    dump()
    print(json.dumps(out["aa_floor_identical"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
