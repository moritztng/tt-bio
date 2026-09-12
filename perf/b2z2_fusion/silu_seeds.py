#!/usr/bin/env python3
"""The 512 aa folds the silu midpoints need, scored the way this fixture can be read.

`b2z2-fused-twopass-loop` killed two activation arms on a whole-molecule RMSD bar of 0.60 A. That
bar sits UNDER the fixture's own sampler noise: `b2z2-fusebias-512-parity` measured the same arm at
different seeds at 0.967-1.906 A per pseudo-domain and 6.80-17.36 A whole-molecule. So this run
produces what `perf/b2z2_fusebias/score.py` reads -- the arm under test and the incumbent fused
kernel at the same seed, the incumbent at a second seed so the sampler's own spread is measured
rather than assumed, and one repeat fold as the A/A floor.

`score.py` is taken from `b2z2-fusebias-512-parity` UNMODIFIED, so its verdict cannot drift from
the one that row published. It names its two arms `off` (the reference) and `on` (the arm under
test), so this writes each candidate its own view of the same folds under exactly those names:

    off = `swiglu`, the fused kernel as b2z2-fusion-rebuild left it (TRIMUL_TAIL_SILU 1)
    on  = the candidate, one of the silu arms below

The folds themselves are shared: the incumbent is folded once per seed and every candidate view
links to the same CIF, so N candidates cost N + 3 folds, not 3N.

    silu_seeds.py --out <json> --cifdir <dir> --cands mid_b:7,mid_c:8 [--seeds 0,1] [--size 512]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--cands", default="mid_b:7,mid_c:8",
                    help="name:TRIMUL_TAIL_SILU pairs for the arms under test")
    ap.add_argument("--seeds", default="0,1", help="first is the A/B seed, the rest are the floor")
    ap.add_argument("--size", default="512")
    ap.add_argument("--grid", default="main")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    cands = [(t.split(":")[0], int(t.split(":")[1])) for t in args.cands.split(",")]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    import tt_bio.transition_swiglu as TS
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    TS.GRID = None if args.grid == "main" else tuple(int(v) for v in args.grid.split("x"))
    served = [0, 0]
    orig_fs = TS.fused_swiglu

    def counted(*a_, **k_):
        o = orig_fs(*a_, **k_)
        served[0 if o is not None else 1] += 1
        return o

    TS.fused_swiglu = counted

    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
        "torch": torch.__version__,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
        "candidates": {n: m for n, m in cands},
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "seeds": seeds, "fused_grid": args.grid},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()
    work = Path(tempfile.mkdtemp(prefix="b2z2-silu-mid-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    name = f"cdk2x2_{args.size}"
    AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-silu-midpoint", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    target = AB.FIX / f"{name}.yaml"

    def fold(silu_mode, seed, keep):
        TS.set_enabled(True)
        TS.SILU = silu_mode
        TS.REJECTS.clear()
        served[0] = served[1] = 0
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
        assert served[0] > 0, "the fused kernel was never served; this arm folded the incumbent"
        return {"silu": silu_mode, "seed": seed, "target": name, "fold_s": round(wall, 3),
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "served": served[0], "declined": served[1],
                "plddt": round(float(metrics.get("plddt",
                                                 metrics.get("confidence_score", 0))), 6)}

    #: (tag, silu mode, seed). The incumbent at the A/B seed first, then its repeat (the A/A
    #: floor), then every candidate at that seed, then the incumbent at the floor seeds.
    plan = [("ref-s%d" % seeds[0], 1, seeds[0]), ("ref-s%d_r1" % seeds[0], 1, seeds[0])]
    plan += [("%s-s%d" % (n, seeds[0]), m, seeds[0]) for n, m in cands]
    plan += [("ref-s%d" % s, 1, s) for s in seeds[1:]]
    for tag, mode, seed in plan:
        r = fold(mode, seed, args.cifdir / f"{args.size}_{tag}")
        r["tag"] = tag
        out["runs"].append(r)
        print(f"  {args.size} {tag:12s} silu={mode} seed={seed} {r['fold_s']:7.3f}s "
              f"sha={r['sha256']} plddt={r['plddt']} served={r['served']}", flush=True)
        dump()

    ref0 = [r for r in out["runs"] if r["tag"].startswith("ref-s%d" % seeds[0])]
    out["aa_floor_identical"] = len({r["sha256"] for r in ref0}) == 1
    dump()

    # One view per candidate, in score.py's own vocabulary. `off` is the incumbent fused kernel,
    # `on` is the candidate; the seed floor comes out of the incumbent folds, which is the arm the
    # candidate has to beat the noise of.
    for cname, _m in cands:
        vdir = args.cifdir.parent / (args.cifdir.name + "_" + cname)
        view = {"env": dict(out["env"], view=cname), "runs": []}
        for r in out["runs"]:
            tag = r["tag"]
            if tag.startswith("ref-"):
                new = tag.replace("ref-", "off-")
            elif tag.startswith(cname + "-"):
                new = tag.replace(cname + "-", "on-")
            else:
                continue
            src = args.cifdir / f"{args.size}_{tag}"
            dst = vdir / f"{args.size}_{new}"
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
            view["runs"].append(dict(r, tag=new))
        (vdir / "runs.json").write_text(json.dumps(view, indent=1))
        print("view for %s: %s" % (cname, vdir), flush=True)
    print(json.dumps({"aa_floor_identical": out["aa_floor_identical"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
