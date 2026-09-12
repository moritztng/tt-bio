#!/usr/bin/env python3
"""The three credible Blackhole levers, singly and together, in one interleaved process.

    HOST   TT_BIO_DEVICE_CONDITIONING   the diffusion conditioning pair track on the device
    SILU   TT_BIO_UNFUSED_SILU          silu as its own op instead of inside ttnn.linear
    AKW    TT_BIO_ATOM_KEY_WINDOW       the atom key window as tile-aligned slices, not a matmul

Each was measured alone and none of them has been measured beside the other two. HOST carries the
campaign's largest single factor (1.06704x) and it was taken in a session whose own A/A floor was
1.02155x on a box a co-tenant joined mid-run, so the number it carries is not one this wave can
publish. This driver retakes all three on one tree, one device open, arms interleaved inside every
rep, and records the loadavg and the device occupancy of every single fold, so a co-tenant arriving
mid-run is visible in the artifact instead of inside a ratio.

The flags live in two different places and both are handled here rather than in the environment of
the process, because an A/B has to flip arms inside one process:

  * `_UNFUSED_SILU` and `_ATOM_KEY_WINDOW` pick a device program, so they are read at build time
    and are module attributes of `tt_bio.tenstorrent`;
  * `TT_BIO_DEVICE_CONDITIONING` is read per call by `tt_bio.boltz2._device_conditioning`, so it
    is an environment variable this driver sets per fold.

EVERY arm applies the FULL flag set, not just the flags it turns on. An arm that only sets its own
flags inherits whatever the previous arm left behind, which is how an interleaved run silently
measures the union five times. `base` is therefore all three explicitly off, and it is the only
thing any ratio here is a ratio of.

    ab_union.py --out <json> --cifdir <dir> --mode parity --sizes 512,298
    ab_union.py --out <json> --mode timing --reps 5 --order base,HOST,base,SILU,base,AKW,base,UNION
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402  -- the fixtures, cfg and MSA seeding, unmodified

# flag -> ("attr", name on tt_bio.tenstorrent) or ("env", variable name)
FLAGS = {
    "SILU": ("attr", "_UNFUSED_SILU"),
    "AKW": ("attr", "_ATOM_KEY_WINDOW"),
    "HOST": ("env", "TT_BIO_DEVICE_CONDITIONING"),
}
ARMS = {
    "base": (),
    "HOST": ("HOST",),
    "SILU": ("SILU",),
    "AKW": ("AKW",),
    "UNION": ("HOST", "SILU", "AKW"),
}


def apply_arm(arm, T):
    """Set every flag this driver knows about: on for this arm, off for every other."""
    on = set(ARMS[arm])
    for flag, (kind, name) in FLAGS.items():
        want = flag in on
        if kind == "attr":
            assert hasattr(T, name), f"{name} missing from this checkout; arm {arm} would be a lie"
            setattr(T, name, want)
        else:
            os.environ[name] = "1" if want else "0"


def read_back(T):
    """What the tree actually believes, read from where the model reads it, after apply_arm."""
    import tt_bio.boltz2 as B
    return {"SILU": bool(T._UNFUSED_SILU), "AKW": bool(T._ATOM_KEY_WINDOW),
            "HOST": bool(B._device_conditioning())}


def occupancy():
    """Which Tenstorrent chips are open, and by whom. Recorded per fold, not once per run."""
    nodes = [f"/dev/tenstorrent/{i}" for i in range(8)]
    nodes = [n for n in nodes if os.path.exists(n)]
    try:
        out = subprocess.run(["lsof", "-F", "pcn"] + nodes,
                             capture_output=True, text=True, timeout=20).stdout
    except Exception as e:
        return {"error": str(e)}
    pids, names = set(), set()
    for line in out.splitlines():
        if line.startswith("p"):
            pids.add(line[1:])
        elif line.startswith("n") and "tenstorrent" in line:
            names.add(line[1:])
    return {"pids": sorted(pids), "nodes": sorted(names)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path)
    ap.add_argument("--mode", choices=("parity", "timing"), required=True)
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--arms", default="base,HOST,SILU,AKW,UNION", help="parity mode: arms per size")
    ap.add_argument("--seeds", default="0", help="parity mode: seeds per arm")
    ap.add_argument("--reps", type=int, default=5, help="timing mode")
    ap.add_argument("--order", default="base,HOST,base,SILU,base,AKW,base,UNION,base,UNION",
                    help="timing mode: arms per rep, in order")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()

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

    # A pinned value in the inherited environment would serve every arm the same flag and the run
    # would read as a null. The driver owns these for the whole process.
    for _kind, _name in FLAGS.values():
        if _kind == "env":
            assert _name not in os.environ, f"{_name} may not be pinned; it is set per fold"

    assert args.steps == 200 and args.recycles == 3, (
        f"the cell is 200 sampling steps and 3 recycles; got {args.steps}/{args.recycles}")
    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
        "torch": torch.__version__, "ttnn": getattr(ttnn, "__version__", "?"),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__,
        "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_at_start": os.getloadavg(), "occupancy_at_start": occupancy(),
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "mode": args.mode, "order": args.order, "reps": args.reps,
                     "diffusion_trace": False},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-union-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"
    msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-union", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    # The block two of the three levers are supposed to be in. A null on the fold with a moved
    # block is a different finding from a lever that never fired.
    wall = {"n": 0, "s": 0.0}
    layer = getattr(T, "PairformerLayer", None)
    if layer is not None:
        inner = layer.__call__

        def timed(self, *a, **kw):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            r = inner(self, *a, **kw)
            ttnn.synchronize_device(dev)
            wall["n"] += 1
            wall["s"] += time.perf_counter() - t
            return r

        layer.__call__ = timed

    def fold(arm, seed, target, keep):
        apply_arm(arm, T)
        flags = read_back(T)
        want = {f: (f in ARMS[arm]) for f in FLAGS}
        assert flags == want, f"arm {arm} asked for {want} and the tree reports {flags}"
        cfg["seed"] = seed
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        wall["n"] = 0
        wall["s"] = 0.0
        ttnn.synchronize_device(dev)
        load_before = os.getloadavg()
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        elapsed = time.perf_counter() - t
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": arm, "seed": seed, "target": target.stem, "fold_s": round(elapsed, 3),
                "sha256": hashlib.sha256(body).hexdigest()[:16], "flags": flags,
                "block_s": round(wall["s"], 4), "block_n": wall["n"],
                "loadavg_before": [round(x, 2) for x in load_before],
                "loadavg_after": [round(x, 2) for x in os.getloadavg()],
                "occupancy": occupancy(),
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    cifdir = args.cifdir or Path(tempfile.mkdtemp(prefix="b2z2-union-cif-"))

    if args.mode == "parity":
        arms = args.arms.split(",")
        seeds = [int(s) for s in args.seeds.split(",")]
        for arm in arms:
            assert arm in ARMS, f"unknown arm {arm}"
        for size in args.sizes.split(","):
            target = AB.FIX / f"cdk2x2_{size}.yaml"
            order = [(a, s) for s in seeds for a in arms]
            order.append((arms[0], seeds[0]))                  # the A/A repeat, last
            seen = set()
            for arm, seed in order:
                tag = f"{arm}-s{seed}" + ("_r1" if (arm, seed) in seen else "")
                seen.add((arm, seed))
                r = fold(arm, seed, target, cifdir / f"{size}_{tag}")
                r["tag"] = tag
                out["runs"].append(r)
                print("  {} {:12s} {:7.3f}s sha={} load={:.2f} plddt={}".format(
                    size, tag, r["fold_s"], r["sha256"], r["loadavg_before"][0], r["plddt"]),
                    flush=True)
                dump()
        summarise_parity(out, dump)
        return 0

    order = args.order.split(",")
    for arm in order:
        assert arm in ARMS, f"unknown arm {arm}"
    target = AB.FIX / "cdk2x2_{}.yaml".format(args.sizes.split(",")[0])
    for rep in range(-1, args.reps):                    # rep -1 is the cold fold, discarded
        for pos, arm in enumerate(order if rep >= 0 else order[:1]):
            r = fold(arm, 0, target, cifdir / "t{}_{}_{}".format(rep, pos, arm))
            r.update(rep=rep, pos=pos, cold=rep < 0, tag="t{}_{}_{}".format(rep, pos, arm))
            out["runs"].append(r)
            print("  rep{:<2d} {:6s} pos{:<2d} fold {:7.3f}s  block {:7.3f}s/{:<4d} "
                  "load {:5.2f} sha={}".format(rep, arm, pos, r["fold_s"], r["block_s"],
                                               r["block_n"], r["loadavg_before"][0], r["sha256"]),
                  flush=True)
            dump()
    summarise_timing(out, dump, order)
    return 0


def summarise_parity(out, dump):
    by_size = {}
    for r in out["runs"]:
        by_size.setdefault(r["target"], {}).setdefault(r["arm"], []).append(r["sha256"])
    out["bit_exact_vs_base"] = {
        size: {arm: (set(sh) == set(arms["base"])) for arm, sh in arms.items()}
        for size, arms in by_size.items() if "base" in arms}
    out["aa_floor_identical"] = {size: len(set(arms.get("base", []))) == 1
                                 for size, arms in by_size.items()}
    dump()
    print(json.dumps({"bit_exact_vs_base": out["bit_exact_vs_base"],
                      "aa_floor_identical": out["aa_floor_identical"]}, indent=1))


def summarise_timing(out, dump, order):
    warm = [r for r in out["runs"] if not r["cold"]]
    by = {a: [r["fold_s"] for r in warm if r["arm"] == a] for a in set(order)}
    blk = {a: [r["block_s"] for r in warm if r["arm"] == a] for a in set(order)}
    base = st.median(by["base"])

    # The A/A floor of THIS run: base at its first position against base at each later position,
    # taken the expensive way round so the floor can never read below 1.000x.
    pos = sorted({r["pos"] for r in warm if r["arm"] == "base"})
    p0 = [r["fold_s"] for r in warm if r["arm"] == "base" and r["pos"] == pos[0]]
    floors = {}
    for p in pos[1:]:
        v = [r["fold_s"] for r in warm if r["arm"] == "base" and r["pos"] == p]
        if v:
            floors[p] = round(max(st.median(p0) / st.median(v), st.median(v) / st.median(p0)), 5)
    aa = max(floors.values()) if floors else None

    summary = {"aa_floor": aa, "aa_floor_per_position": floors,
               "base_spread_pct": round(100 * (max(by["base"]) - min(by["base"])) / base, 2),
               "loadavg_1m_min_max": [min(r["loadavg_before"][0] for r in warm),
                                      max(r["loadavg_after"][0] for r in warm)],
               "foreign_pids_seen": sorted({p for r in warm for p in r["occupancy"].get("pids", [])
                                            if p != str(os.getpid())}),
               "arms": {}}
    for arm in sorted(by):
        v = by[arm]
        ratio = base / st.median(v)
        row = {"n": len(v), "median_fold_s": round(st.median(v), 3),
               "min_fold_s": round(min(v), 3), "max_fold_s": round(max(v), 3),
               "ratio_vs_base": round(ratio, 5),
               "spread_pct": round(100 * (max(v) - min(v)) / st.median(v), 2),
               "above_floor": bool(aa is None or ratio > aa),
               "ratio_resolved": round(ratio, 5) if (aa is None or ratio > aa) else 1.0}
        if blk[arm] and blk["base"]:
            row["median_block_s"] = round(st.median(blk[arm]), 3)
            row["block_ratio_vs_base"] = round(st.median(blk["base"]) / st.median(blk[arm]), 5)
        summary["arms"][arm] = row

    # The union discount: the product of the singles against the measured union.
    singles = [summary["arms"][a]["ratio_vs_base"] for a in ("HOST", "SILU", "AKW")
               if a in summary["arms"]]
    if len(singles) == 3 and "UNION" in summary["arms"]:
        prod = singles[0] * singles[1] * singles[2]
        meas = summary["arms"]["UNION"]["ratio_vs_base"]
        summary["union_discount"] = {
            "product_of_singles": round(prod, 5), "measured_union": round(meas, 5),
            "discount_ratio": round(prod / meas, 5),
            "discount_pct": round(100 * (prod / meas - 1), 2),
            "product_seconds_saved": round(base - base / prod, 3),
            "measured_seconds_saved": round(base - base / meas, 3),
            "inside_aa_floor": bool(aa is not None and abs(prod / meas - 1) < (aa - 1))}
    summary["identical_across_arms"] = len({r["sha256"] for r in warm}) == 1
    summary["sha_by_arm"] = {a: sorted({r["sha256"] for r in warm if r["arm"] == a})
                             for a in sorted(by)}
    out["timing_summary"] = summary
    dump()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
