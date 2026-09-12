#!/usr/bin/env python3
"""`TT_BIO_ATOM_L1` against base across the size range a user actually folds, one process per size.

`b2z2-bh-stack-atom` measured this lever at 512 aa, where it sits under the A/A floor, and then ran
a single unpaired fold per arm at 768 aa that read 85.504 -> 45.243 s. One fold per arm is not a
ratio. This driver takes the same question properly: for each size, `base, L1, base` interleaved
inside ONE process and ONE device open, n reps, so the A/A floor comes out of the same session as
the ratio and a co-tenant arriving mid-run lands in the artifact instead of inside the number.

Three things are recorded per fold that the seconds alone cannot tell you apart:

  * `ATOM_L1_STATS` -- `{l1: n, dram: n}`. A lever that can decline needs an instrument that says
    it declined, and the whole question here is where it stops engaging. A size where the gate
    silently falls back to DRAM and the fold still looks fast is the trap this exists to catch.
  * the CIF sha256 -- the lever sets a memory config and nothing else, so base and L1 must write
    the same bytes at every size. This is checked against BASE ON THE SAME BOX AND SIZE, never
    against `tt_bio.reference`, which zero-initialises 23 PairformerLayer weights and would pass
    for an arm that computed nothing.
  * `step_n` and `block_n` -- read back off the model on every fold, so the protocol (200 sampling
    steps, 3 recycles) is asserted by the driver rather than by its author.

    size_curve.py --out <json> --cifdir <dir> --size 768 --reps 5

The deliverable is the pair of exponents and where the curves separate, so `fit_curve.py` reads
the per-size JSONs this writes and does the fitting; this file only measures.
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

# flag -> ("attr", name on tt_bio.tenstorrent) or ("env", variable name).
# This branch is origin/main plus `TT_BIO_ATOM_L1` and nothing else, so there is exactly one flag
# here. `apply_arm` asserts the attribute exists, so a checkout without the lever fails loudly
# instead of measuring base twice and calling it a ratio.
FLAGS = {
    "L1": ("attr", "_ATOM_L1"),
}
ARMS = {
    "base": (),
    "L1": ("L1",),
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
    nodes = [f"/dev/tenstorrent/{i}" for i in range(32)]
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
    ap.add_argument("--size", required=True, help="one size per process: 512|640|768|896|1024")
    ap.add_argument("--reps", type=int, default=5)
    # base at two positions per rep, so the A/A floor is the SAME estimator as the ratio: both
    # are medians over the same session. Quoting a per-position floor beside a median-of-n ratio
    # is the mistake CONTEXT.md's A/A FLOORS block names.
    ap.add_argument("--order", default="base,L1,base",
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
        "size": int(args.size),
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "order": args.order, "reps": args.reps,
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
    fixture = AB.FIX / f"cdk2x2_{args.size}.yaml"
    a3m = AB.FIX / f"cdk2x2_{args.size}.a3m"
    assert fixture.exists() and a3m.exists(), f"no fixture pair for size {args.size}"
    AB._seed_msa(fixture, a3m.read_text(), msa_dir)

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
        ttnn.synchronize_device(dev)
        load_before = os.getloadavg()
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        elapsed = time.perf_counter() - t
        # CHEAT-CHECK, asserted per fold: the sampler ran its full 200 steps. A ratio taken
        # against a short-sampled arm is the one thing this campaign is not allowed to ship.
        assert step["n"] == 200, f"sampler ran {step['n']} steps, not 200"
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
                "loadavg_before": [round(x, 2) for x in load_before],
                "loadavg_after": [round(x, 2) for x in os.getloadavg()],
                "occupancy": occupancy(),
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    cifdir = args.cifdir or Path(tempfile.mkdtemp(prefix="b2z2-union-cif-"))

    order = args.order.split(",")
    for arm in order:
        assert arm in ARMS, f"unknown arm {arm}"
    target = fixture
    for rep in range(-1, args.reps):                    # rep -1 is the cold fold, discarded
        for pos, arm in enumerate(order if rep >= 0 else order[:1]):
            r = fold(arm, 0, target, cifdir / "t{}_{}_{}".format(rep, pos, arm))
            r.update(rep=rep, pos=pos, cold=rep < 0, size=int(args.size),
                     tag="t{}_{}_{}".format(rep, pos, arm))
            out["runs"].append(r)
            print("  {}aa rep{:<2d} {:4s} pos{:<2d} fold {:8.3f}s  step {:7.3f}s/{:<4d} "
                  "block {:7.3f}s/{:<4d} atom_l1={} load {:5.2f} sha={}".format(
                      args.size, rep, arm, pos, r["fold_s"], r["step_s"], r["step_n"],
                      r["block_s"], r["block_n"], r["atom_l1"], r["loadavg_before"][0],
                      r["sha256"]),
                  flush=True)
            dump()
    summarise_timing(out, dump, order)
    return 0


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

    summary["identical_across_arms"] = len({r["sha256"] for r in warm}) == 1
    summary["sha_by_arm"] = {a: sorted({r["sha256"] for r in warm if r["arm"] == a})
                             for a in sorted(by)}

    # The gate, per arm, over every warm fold. `dram > 0` anywhere is the finding this row exists
    # to be able to see: a size where the lever silently declines and the fold still looks fast.
    summary["atom_l1_gate"] = {
        a: {"l1": sorted({r["atom_l1"]["l1"] for r in warm if r["arm"] == a}),
            "dram": sorted({r["atom_l1"]["dram"] for r in warm if r["arm"] == a})}
        for a in sorted(by)}
    summary["gate_declined_anywhere"] = any(
        r["atom_l1"]["dram"] for r in warm if r["arm"] == "L1")
    summary["gate_took_l1_on_every_call"] = all(
        r["atom_l1"]["l1"] > 0 and r["atom_l1"]["dram"] == 0
        for r in warm if r["arm"] == "L1")

    # The protocol, read back off the model on every fold rather than asserted by the author.
    summary["protocol_readback"] = {
        "step_n": sorted({r["step_n"] for r in warm}),
        "block_n": sorted({r["block_n"] for r in warm})}
    out["timing_summary"] = summary
    dump()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
