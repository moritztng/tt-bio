#!/usr/bin/env python3
"""Boltz-2 A/B for the three levers `b2z-levers-default-on` turned on: coordinates and wall time.

The levers are the fused bias stacks in the diffusion conditioning (TT_BIO_FUSE_BIAS_STACKS, not
bit-exact), the batched DST acquire in the fused SDPA's mask add (TT_BIO_SDPA_ADD_GRANULARITY,
bit-exact) and the same treatment in the gated channel permute (TT_BIO_GATE_GRANULARITY,
bit-exact). Two questions, one harness, because they want the same folds:

  correctness     do the shipped defaults move the structure past the 0.35 A bar, and does 512 aa
                  still fold at all? Run it on Wormhole as well as Blackhole --
                  `b2x-integrate-wh-neutral` is the precedent, and the reason is that the last two
                  flipped-on Boltz-2 levers shipped fleet-wide having only ever run on qb2.
  speed           what did flipping the defaults actually bank on the fold? --reps --warmup.

Arms in ONE process so the A/A floor and the A/B share a program cache:

    base    every lever forced off -- the pre-merge path
    ship    every lever at its shipped default

The default order is base/base/ship: the repeated base arm IS the A/A floor, and a non-zero floor
invalidates the third arm. Same fixtures and protocol as the Blackhole control, so hosts compare
directly. The 298 aa fixture is the one the 0.35 A bar is written against; 512 aa answers the
crash-and-OOM half and is the size the published cell is measured at.

The arms are applied in-process, which is legal for each lever but for a different reason each
time: TT_BIO_FUSE_BIAS_STACKS is read per call, TT_BIO_SDPA_ADD_GRANULARITY is read inside
`sdpa_generic.build` AND is in the program-cache key, and GATE_GRANULARITY is a module constant
so it is assigned rather than exported -- it is in that kernel's cache key too. Without those two
keys both arms would get the first arm's compiled program back and read a parity that means
nothing.

    fold_ab.py --out <json> --cifdir <dir> [--reps N] [--sizes 512,298] [--warmup]

Defaults reproduce the Wormhole neutrality run exactly: one warmup-free pass of base/base/ship at
512 then 298 aa. --reps interleaves base and ship N times each for a paired timing run, and
--warmup discards one fold per arm first so the program cache is not in the first rep.
"""
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

import ab_flag_levers as AB  # noqa: E402  -- the fixtures, cfg and MSA seeding, unmodified

BASE_ORDER = ["base", "base", "ship"]


def apply_arm(arm, RB):
    """Set every lever for this arm. See the module docstring for why each spelling differs."""
    if arm == "ship":
        for k in ("TT_BIO_FUSE_BIAS_STACKS", "TT_BIO_SDPA_ADD_GRANULARITY"):
            os.environ.pop(k, None)
        RB.GATE_GRANULARITY = 2
    else:
        os.environ["TT_BIO_FUSE_BIAS_STACKS"] = "0"
        os.environ["TT_BIO_SDPA_ADD_GRANULARITY"] = "1"
        RB.GATE_GRANULARITY = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--reps", type=int, default=0,
                    help="interleave base/ship this many times each instead of the default "
                         "base/base/ship. The A/A floor then comes from the base arm's own reps.")
    ap.add_argument("--sizes", default="512,298")
    ap.add_argument("--warmup", action="store_true",
                    help="discard one fold per arm first. Required for a timing run; the "
                         "neutrality run does not need it because it compares coordinates.")
    args = ap.parse_args()
    order = BASE_ORDER if args.reps < 1 else ["base", "ship"] * args.reps

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from tt_bio import reblock_permute as RB
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    assert not (set(os.environ) & {"TT_BIO_FUSE_BIAS_STACKS", "TT_BIO_GATE_GRANULARITY",
                                   "TT_BIO_SDPA_ADD_GRANULARITY"}), \
        "no lever may be pinned in the environment; the arms are set in-process"

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    import socket
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y],
        "arch": str(dev.arch()), "torch": torch.__version__,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "seed": AB.SEED, "diffusion_trace": False},
        "shipped_defaults": {"gate_granularity": RB.GATE_GRANULARITY},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z-wh-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z-levers-wh", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(arm, target, keep):
        apply_arm(arm, RB)
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
        return {"arm": arm, "target": target.stem, "fold_s": round(wall, 3),
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    # 512 aa first: it is the crash-and-OOM question, and a failure there should not wait behind
    # the accuracy fixture.
    for size in args.sizes.split(","):
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        if args.warmup:
            for arm in ("base", "ship"):
                r = fold(arm, target, args.cifdir / f"{size}_warm_{arm}")
                r["warmup"] = True
                out["runs"].append(r)
                print(f"  {size} warm {arm:5s} {r['fold_s']:7.3f}s", flush=True)
            dump()
        n = {}
        for arm in order:
            i = n[arm] = n.get(arm, -1) + 1
            keep = args.cifdir / f"{size}_{arm}_{i}"
            r = fold(arm, target, keep)
            r["rep"] = i
            out["runs"].append(r)
            print(f"  {size} {arm:5s} rep{i} {r['fold_s']:7.3f}s sha={r['sha256']} "
                  f"plddt={r['plddt']}", flush=True)
            dump()

    warm = [r for r in out["runs"] if not r.get("warmup")]
    for arm in ("base", "ship"):
        v = sorted(r["fold_s"] for r in warm if r["arm"] == arm and "512" in r["target"])
        if v:
            out.setdefault("median_512_s", {})[arm] = v[len(v) // 2]
    m = out.get("median_512_s", {})
    if "base" in m and "ship" in m and m["ship"]:
        out["fold_speedup_512"] = round(m["base"] / m["ship"], 5)
    aa = [r for r in warm if r["arm"] == "base"]
    out["aa_floor_identical"] = {
        s: len({r["sha256"] for r in aa if r["target"].endswith(s)}) == 1
        for s in ("512", "298")}
    dump()
    print(json.dumps({k: out[k] for k in ("aa_floor_identical", "median_512_s",
                                          "fold_speedup_512") if k in out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
