#!/usr/bin/env python3
"""The three credible Blackhole levers, singly and together, in one interleaved process.

    HOST   TT_BIO_DEVICE_CONDITIONING   the diffusion conditioning pair track on the device
    SILU   TT_BIO_UNFUSED_SILU          silu as its own op instead of inside ttnn.linear
    AKW    TT_BIO_ATOM_KEY_WINDOW       the atom key window as tile-aligned slices, not a matmul
    LN     BOLTZ2_ADALN_SHARED_SNORM    one shared LayerNorm for the 48 AdaLN sites of a step
    LAY    TT_BIO_HEAD_PAD_TAIL         the head-split tail carried padded, four layout programs gone

LN and LAY are the two step levers this pass adds. Both were measured on Wormhole only, 1.03932x
and 1.02574x ON THE STEP, and neither has ever held a Blackhole chip. The third step lever the
brief names, `wk/b2z2-step-program-fusion`, is ALREADY IN THE UNION: it is AKW, and its Blackhole
number is the union row's 1.02744x. Anything that stacks it a second time is double counting.

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
    ab_union.py --out <json> --mode timing --reps 5 --order base,LN,base,LAY,base,UNION,base,STACK

The step wall is recorded per fold beside the PairformerLayer wall, because LN and LAY are step
levers: a null on the fold with a moved step is a different finding from a lever that never fired.
`DiffusionModule.forward` returns a torch tensor, so it already reads back from the device and no
extra synchronize is added around it; the Pairformer wrap keeps the explicit syncs it was built
with so its numbers stay comparable with `b2z2-bh-union-clean`'s.
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
    "LN": ("attr", "_B2_ADALN_SHARED_SNORM"),
    "LAY": ("attr", "_HEAD_PAD_TAIL"),
    # Added by `b2z2-bh-stack-atom`. SG is the same transform as AKW built on the matrix instead
    # of on the geometry, so the two are alternatives and never appear in one arm; L1 and KVP sit
    # on top of whichever gather is live.
    "SG": ("attr", "_ATOM_SHIFT_GATHER"),
    "L1": ("attr", "_ATOM_L1"),
    "KVP": ("attr", "_ATOM_KV_PREPROJ"),
}
UNION = ("HOST", "SILU", "AKW")
CUNION = ("HOST", "SILU", "SG")
ARMS = {
    "base": (),
    "HOST": ("HOST",),
    "SILU": ("SILU",),
    "AKW": ("AKW",),
    "LN": ("LN",),
    "LAY": ("LAY",),
    "UNION": UNION,
    "U_LN": UNION + ("LN",),
    "U_LAY": UNION + ("LAY",),
    "STACK": UNION + ("LN", "LAY"),
    # The corrected ladder: the same five levers with the gather that reproduces the one-hot.
    "SG": ("SG",),
    "L1": ("L1",),
    "SG_L1": ("SG", "L1"),
    "SG_KVP": ("SG", "KVP"),
    "CUNION": CUNION,
    "CSTACK": CUNION + ("LN", "LAY"),
    "CSTACK_L1": CUNION + ("LN", "LAY", "L1"),
    "CSTACK_L1_KVP": CUNION + ("LN", "LAY", "L1", "KVP"),
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
    got = {}
    for flag, (kind, name) in FLAGS.items():
        got[flag] = bool(getattr(T, name)) if kind == "attr" else None
    got["HOST"] = bool(B._device_conditioning())
    return got


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
    ap.add_argument("--mode", choices=("parity", "timing", "ladder"), required=True)
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

    # The 200-step sampler. `DiffusionModule.forward` hands a torch tensor back to the torch
    # sampler loop, so it has already read back from the device and a bare wall clock around it
    # is the step's real cost. No synchronize is added: forcing one per step would delete the
    # exposed-dispatch overlap that is part of what a step costs.
    step = {"n": 0, "s": 0.0}
    dmod = getattr(T, "DiffusionModule", None)
    if dmod is not None:
        _fwd = dmod.forward

        def timed_step(self, *a, **kw):
            t = time.perf_counter()
            r = _fwd(self, *a, **kw)
            step["n"] += 1
            step["s"] += time.perf_counter() - t
            return r

        dmod.forward = timed_step

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
        step["n"] = 0
        step["s"] = 0.0
        # A lever that can decline needs an instrument that says it declined.
        T.ATOM_L1_STATS["l1"] = T.ATOM_L1_STATS["dram"] = 0
        T.ATOM_SHIFT_GATHER_STATS[0] = T.ATOM_SHIFT_GATHER_STATS[1] = 0
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
                "step_s": round(step["s"], 4), "step_n": step["n"],
                "step_ms": round(1000 * step["s"] / step["n"], 4) if step["n"] else None,
                "atom_l1": dict(T.ATOM_L1_STATS),
                "atom_gather": {"slice_concat": T.ATOM_SHIFT_GATHER_STATS[0],
                                "one_hot_matmul": T.ATOM_SHIFT_GATHER_STATS[1]},
                "loadavg_before": [round(x, 2) for x in load_before],
                "loadavg_after": [round(x, 2) for x in os.getloadavg()],
                "occupancy": occupancy(),
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    cifdir = args.cifdir or Path(tempfile.mkdtemp(prefix="b2z2-union-cif-"))

    if args.mode == "ladder":
        # One fold per (size, arm). An OOM is the answer the ladder exists to get, so it is caught,
        # recorded with its message and the size it happened at, and the ladder keeps climbing.
        arms = args.arms.split(",")
        for arm in arms:
            assert arm in ARMS, f"unknown arm {arm}"
        for size in args.sizes.split(","):
            target = AB.FIX / f"cdk2x2_{size}.yaml"
            assert target.exists(), f"no fixture {target}"
            AB._seed_msa(target, (AB.FIX / f"cdk2x2_{size}.a3m").read_text(), msa_dir)
            for arm in arms:
                tag = f"{size}_{arm}"
                try:
                    r = fold(arm, 0, target, cifdir / tag)
                except Exception as e:                       # noqa: BLE001 -- the finding
                    r = {"arm": arm, "target": target.stem, "failed": True,
                         "error_type": type(e).__name__, "error": str(e)[:2000],
                         "loadavg_before": [round(x, 2) for x in os.getloadavg()]}
                    print("  {:14s} FAILED {}: {}".format(
                        tag, type(e).__name__, str(e)[:200]), flush=True)
                else:
                    print("  {:14s} fold {:8.3f}s  atom_l1={} gather={} sha={} plddt={}".format(
                        tag, r["fold_s"], r["atom_l1"], r["atom_gather"], r["sha256"],
                        r["plddt"]), flush=True)
                r["tag"] = tag
                r["size"] = size
                out["runs"].append(r)
                dump()
        summarise_ladder(out, dump)
        return 0

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
                  "step {:6.3f}s/{:<4d} load {:5.2f} sha={}".format(
                      rep, arm, pos, r["fold_s"], r["block_s"], r["block_n"],
                      r["step_s"], r["step_n"], r["loadavg_before"][0], r["sha256"]),
                  flush=True)
            dump()
    summarise_timing(out, dump, order)
    return 0


def summarise_ladder(out, dump):
    """Per size: did the gate take L1, did the fold survive, and is the arm bit-exact against base.

    A DECLINE is a pass for the gate and is reported as one. A failure is a NO-GO for a default
    flip and is reported with the size it happened at.
    """
    lad = {}
    for r in out["runs"]:
        row = lad.setdefault(r.get("size", r["target"]), {})
        if r.get("failed"):
            row[r["arm"]] = {"failed": True, "error_type": r["error_type"],
                             "error": r["error"][:300]}
        else:
            row[r["arm"]] = {"fold_s": r["fold_s"], "atom_l1": r["atom_l1"],
                             "atom_gather": r["atom_gather"], "sha256": r["sha256"],
                             "plddt": r["plddt"]}
    for size, row in lad.items():
        base = row.get("base", {})
        for arm, v in row.items():
            if arm != "base" and not v.get("failed") and base.get("sha256"):
                v["bit_exact_vs_base"] = (v["sha256"] == base["sha256"])
    out["ladder"] = lad
    dump()
    print(json.dumps(lad, indent=1), flush=True)


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
    stp = {a: [r["step_s"] for r in warm if r["arm"] == a and r["step_n"]] for a in set(order)}
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
        if stp[arm] and stp["base"]:
            row["median_step_s"] = round(st.median(stp[arm]), 3)
            row["step_ratio_vs_base"] = round(st.median(stp["base"]) / st.median(stp[arm]), 5)
        summary["arms"][arm] = row

    # Composition: the product of the parts against the measured whole, for every whole this
    # run has both halves of. Each entry names its own parts so nothing is charged twice.
    def compose(name, parts, whole):
        if whole not in summary["arms"] or any(p not in summary["arms"] for p in parts):
            return
        prod = 1.0
        for p in parts:
            prod *= summary["arms"][p]["ratio_vs_base"]
        meas = summary["arms"][whole]["ratio_vs_base"]
        summary.setdefault("composition", {})[name] = {
            "parts": list(parts), "whole": whole,
            "product_of_parts": round(prod, 5), "measured": round(meas, 5),
            "discount_ratio": round(prod / meas, 5),
            "discount_pct": round(100 * (prod / meas - 1), 2),
            "product_seconds_saved": round(base - base / prod, 3),
            "measured_seconds_saved": round(base - base / meas, 3),
            "inside_aa_floor": bool(aa is not None and abs(prod / meas - 1) < (aa - 1))}

    compose("union_from_singles", ("HOST", "SILU", "AKW"), "UNION")
    compose("u_ln_from_union_and_ln", ("UNION", "LN"), "U_LN")
    compose("u_lay_from_union_and_lay", ("UNION", "LAY"), "U_LAY")
    compose("stack_from_union_ln_lay", ("UNION", "LN", "LAY"), "STACK")
    compose("stack_from_u_ln_and_lay", ("U_LN", "LAY"), "STACK")
    if "union_from_singles" in summary.get("composition", {}):
        summary["union_discount"] = summary["composition"]["union_from_singles"]
    summary["identical_across_arms"] = len({r["sha256"] for r in warm}) == 1
    summary["sha_by_arm"] = {a: sorted({r["sha256"] for r in warm if r["arm"] == a})
                             for a in sorted(by)}
    out["timing_summary"] = summary
    dump()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
