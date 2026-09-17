#!/usr/bin/env python3
"""Boltz-2 fold A/B for TT_BIO_LINEAR_KBLOCK: the shape-routed matmul block config.

The lever names a measured `program_config` at eight `linear` sites instead of letting ttnn derive
one. Op-level it is worth 1.1565-1.2434x on fc2, 1.1595-1.3878x on fc3 and 1.0360-1.0627x on the DiT
projection, measured through `_linear_blocked` itself (`perf/c12_kblock/lever_ab_*.json`). Predicted
at 512 aa from those rates against the census call counts: +0.2052 s / 277.0 Mc. An op-level ratio is
NOT a fold-level second -- two C10 rows pre-registered 3.04 s and 4.908 s and delivered -0.0214 s
and 0.156 s -- so this harness exists to score that prediction rather than to confirm it.

Arms in ONE process so both share a program cache and a model load:

    off     `_LINEAR_KBLOCK = False` -- today's path, `ttnn.linear(core_grid=...)`
    on      `_LINEAR_KBLOCK = True`  -- the table's config where it names the shape

Switching in-process is legal here because the lever changes `program_config`, which is part of the
matmul op's program-cache key: the two arms cannot be served each other's compiled program. That is
the trap `perf/b2z_levers/fold_ab.py` documents for its own levers, and it is the reason this one
needs no environment export.

WITNESS, and it is not optional. A fold A/B against a lever that never fired measures noise and
reads like a null result, so `_linear_block_cfg` is wrapped here to count hits and misses per
(mt_total, kt, nt). The run FAILS if the `on` arm recorded no hits, and the per-key hit counts are
written to the json so the measured seconds can be divided by the calls that actually took the
config rather than by the census's expectation of them.

The default order is off/off/on: the repeated `off` arm IS the A/A floor, and a ratio inside that
floor is not a result. --reps interleaves off/on instead, which is what the headline number wants.

    fold_ab_kblock.py --out <json> --cifdir <dir> [--reps N] [--sizes 512,298,768] [--warmup]

Must run under `~/.coworker/scripts/benchlock.sh` (exit 75 = retry later, never measure anyway) and
with the AICLK forced and sampled: a fold second is meaningless without the clock it was taken at.
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
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
sys.path.insert(0, str(REPO / "perf" / "c12_kblock"))

import ab_flag_levers as AB  # noqa: E402  -- the fixtures, cfg and MSA seeding, unmodified

OFF_ORDER = ["off", "off", "on"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--reps", type=int, default=0,
                    help="interleave off/on this many times each instead of off/off/on. The A/A "
                         "floor then comes from the off arm's own spread across reps.")
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--warmup", action="store_true",
                    help="discard one fold per arm first, so neither arm pays compile in a timed "
                         "rep. Required for any timing run.")
    ap.add_argument("--mhz", type=int, default=1350)
    args = ap.parse_args()
    order = OFF_ORDER if args.reps < 1 else ["off", "on"] * args.reps

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import clk
    import tt_bio.tenstorrent as TB
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    assert "TT_BIO_LINEAR_KBLOCK" not in os.environ, (
        "the lever may not be pinned in the environment; the arms are set in-process")
    assert TB._LINEAR_KBLOCK is False, "the lever must default off"

    # --- the witness -------------------------------------------------------------------------
    hits, misses = Counter(), Counter()
    _real = TB._linear_block_cfg

    def counting_cfg(a_shape, w_shape, activation, bias, core_grid):
        cfg = _real(a_shape, w_shape, activation, bias, core_grid)
        if TB._LINEAR_KBLOCK:
            m, k, n = int(a_shape[-2]), int(a_shape[-1]), int(w_shape[-1])
            b = 1
            for d in a_shape[:-2]:
                b *= int(d)
            key = "%d,%d,%d" % (b * m // 32, k // 32, n // 32)
            (hits if cfg is not None else misses)[key] += 1
        return cfg

    TB._linear_block_cfg = counting_cfg

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = TB.get_device()
    nodes = clk.nodes_open_by_this_process()
    held = clk.force(args.mhz, nodes) if args.mhz else []
    g = dev.compute_with_storage_grid_size()
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
        "torch": torch.__version__, "tt_bio_file": _TB.__file__,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen("git -C %s rev-parse HEAD" % REPO).read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "clock_forced_MHz": args.mhz, "clock_forced_nodes": held,
        "table": {"%d,%d,%d" % k: list(v) for k, v in TB._LINEAR_BLOCK.items()},
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "seed": AB.SEED, "order": order},
        "predicted_s_at_512": 0.2052, "predicted_Mc_at_512": 277.0,
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()
    work = Path(tempfile.mkdtemp(prefix="c12-kblock-", dir=str(REPO / "perf" / "c12_kblock")))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    fixtures = {s: AB.FIX / ("cdk2x2_%s.yaml" % s) for s in args.sizes.split(",")}
    for s, y in fixtures.items():
        assert y.is_file(), "no fixture for %s aa at %s" % (s, y)
        AB._seed_msa(y, (AB.FIX / ("cdk2x2_%s.a3m" % s)).read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("c12-kblock-unlock", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(arm, target, keep):
        TB._LINEAR_KBLOCK = (arm == "on")
        hits.clear()
        misses.clear()
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        s = clk.Sampler(nodes[0])
        s.start()
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        clock = s.stop()
        TB._LINEAR_KBLOCK = False
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": arm, "target": target.stem, "fold_s": round(wall, 3), "clock": clock,
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "cif": str(keep / cifs[0].name),
                "hits": dict(hits), "misses": dict(misses),
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    for size, target in fixtures.items():
        if args.warmup:
            for arm in ("off", "on"):
                r = fold(arm, target, args.cifdir / ("%s_warm_%s" % (size, arm)))
                r["warmup"] = True
                out["runs"].append(r)
                print("  %s warm %-3s %7.3fs" % (size, arm, r["fold_s"]), flush=True)
            dump()
        n = {}
        for arm in order:
            i = n[arm] = n.get(arm, -1) + 1
            r = fold(arm, target, args.cifdir / ("%s_%s_%d" % (size, arm, i)))
            out["runs"].append(r)
            print("  %s %-3s #%d %7.3fs  clock %s  hits %s"
                  % (size, arm, i, r["fold_s"], r["clock"].get("min"), r["hits"]), flush=True)
            dump()

        timed = [x for x in out["runs"] if x["target"].endswith(size) and not x.get("warmup")]
        on = [x["fold_s"] for x in timed if x["arm"] == "on"]
        off = [x["fold_s"] for x in timed if x["arm"] == "off"]
        fired = sum(sum(x["hits"].values()) for x in timed if x["arm"] == "on")
        assert fired > 0, (
            "the `on` arm recorded ZERO table hits at %s aa: the lever never fired, so any ratio "
            "here is noise. Check the fixture's token axis against the table keys." % size)
        if on and off:
            med = lambda v: sorted(v)[len(v) // 2]
            aa = (max(off) / min(off)) if len(off) > 1 else None
            res = {"size": size, "off_s": med(off), "on_s": med(on),
                   "delta_s": round(med(off) - med(on), 4),
                   "x": round(med(off) / med(on), 4), "aa_spread": aa,
                   "table_hits_on_arm": fired,
                   "digest_moved": len({x["sha256"] for x in timed if x["arm"] == "on"}
                                       | {x["sha256"] for x in timed if x["arm"] == "off"}) > 1}
            out.setdefault("result", {})[size] = res
            print("  == %s aa: %.3f -> %.3f s, %+.4f s, %.4fx  [A/A spread %s]  %d table hits"
                  % (size, res["off_s"], res["on_s"], res["delta_s"], res["x"],
                     "n/a" if aa is None else "%.4fx" % aa, fired), flush=True)
            if aa is not None and abs(res["x"] - 1) <= abs(aa - 1):
                print("  ^ NOT A RESULT: inside this session's own A/A spread", flush=True)
            dump()

    clk.release()
    print("wrote %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
