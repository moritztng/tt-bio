#!/usr/bin/env python3
"""What the Transition row chunk is worth in FOLD SECONDS, and what its own session's floor is.

The op ratio is not the deliverable any more. `roof-true-floor` put the whole remaining prize at
2.309 s against a 17.340 s cell, so a lever is worth what it moves on the fold and nothing else.

One process, one device open, model loaded once. Arms alternate INSIDE the process with the
shipped arm run twice per rep, so the session's own A/A floor falls out of the same data instead
of being a separate run that a load change can sit between. Arm order is fixed and the whole set
runs once per rep, so a clock ramp or a JIT warm-up cannot bias one arm against another.

There are TWO contention detectors, because one of them is blind on its own.

The A/A floor is the variance detector: two shipped slots per rep, either side of the test arm, so
box drift lands in the floor and not in the ratio. `roof-pair-transition` got 0.157 % on a quiet
box; above ~1 % the session is BLOCKED, not scaled.

The floor cannot see SATURATION. On 2026-09-15 this harness ran at 512 aa with a sibling worker's
200-step fold on the box at 253 % CPU. The fold went host-bound, both arms clamped to the same
host-limited wall at 23.5 s against a 15.2 s quiet-box median, the 0.34 s of device time the lever
saves vanished into the stall -- and the A/A floor stayed tight, because steady saturation slows
both shipped slots equally. The session would have published ratio ~1.000 as INTERPRETABLE and
refuted a real win. A tight floor means "no jitter", not "not saturated".

So pass `--quiet-ship-median SIZE:SECONDS` with the reference arm's known quiet-box median. Any leg
whose shipped median exceeds it by more than `--saturation-tol` is BLOCKED-ON-SATURATION and its
ratio is suppressed. Digests are never suppressed: contention costs you the clock, never the bytes.

Every arm is bit-exact (proved fold-level at 298/512/768 aa by the ladder in this directory), so
the digest is carried here only as the cheap regression signal it is.

    foldab.py --out <json> --legs 512:h48,768:h32 --reps 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import statistics as st
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402  -- fixtures, cfg and MSA seeding, unmodified


def leg_verdict(floor_pct, ship_median_s, quiet_median_s, tol):
    """Both detectors have to clear, because neither sees the other's failure.

    The A/A floor catches JITTER: a box whose speed moves between the two shipped slots. It is
    blind to SATURATION, where the box is uniformly slow, both shipped slots agree, the floor
    reads tight and the device-side saving is hidden inside a host stall. The saturation check
    catches that, and is itself blind to jitter around a correct mean. Contention is only allowed
    to cost the clock, so this never touches the digests.
    """
    if floor_pct > 1.0:
        return "BLOCKED-ON-CONTENTION"
    if quiet_median_s and ship_median_s > quiet_median_s * tol:
        return "BLOCKED-ON-SATURATION"
    return "INTERPRETABLE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--legs", default="512:h48",
                    help="comma list of <size>:<arm>; the shipped arm is added around each")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--quiet-ship-median", default="",
                    help="SIZE:SECONDS[,...] -- the reference arm's median on a QUIET box. A leg "
                         "whose shipped median runs above it by more than --saturation-tol is "
                         "blocked, because a host-bound session hides the saving without moving "
                         "the A/A floor.")
    ap.add_argument("--saturation-tol", type=float, default=1.10,
                    help="how far above the quiet median a session may sit and still be read")
    ap.add_argument("--ref", default="ship",
                    help="the arm run on both sides of the test arm; `ship` is the tree's own "
                         "default, `off` pins the Blackhole row-height raise off")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()
    assert "TT_BIO_TRANSITION_H_CHUNK" not in os.environ, "the arm is set in-process, not pinned"
    assert not os.environ.get("TT_BIO_UNFUSED_SILU"), "the silu arm is held; every arm runs with it off"

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    legs = [(int(s.split(":")[0]), s.split(":")[1]) for s in args.legs.split(",") if s.strip()]

    out = {
        "doc": __doc__,
        "env": {
            "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
            "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
            "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loadavg_start": os.getloadavg(),
            "unfused_silu": bool(TT._UNFUSED_SILU),
            "protocol": {"sampling_steps": args.steps, "recycling_steps": args.recycles,
                         "seed": AB.SEED, "reps": args.reps},
        },
        "folds": [], "legs": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        out["env"]["loadavg_now"] = os.getloadavg()
        args.out.write_text(json.dumps(out, indent=1))
    dump()

    work = Path(tempfile.mkdtemp(prefix="chunk-foldab-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for size, _ in legs:
        AB._seed_msa(AB.FIX / f"cdk2x2_{size}.yaml",
                     (AB.FIX / f"cdk2x2_{size}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("roof-transition-chunk-foldab", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(size, arm, slot, rep):
        # `ship` leaves the tree alone. `off`/`on` pin the Blackhole per-shape row-height raise,
        # which is default-on, so the shipped lever is still A/B-able on one build without the
        # flat-height hook -- the hook forces ONE height for every shape and this lever gives a
        # different height to the pair track and the MSA track, so it cannot stand in for it.
        # `hNN` is the old flat-height screen arm, unchanged.
        if arm in ("ship", "off", "on"):
            os.environ.pop("TT_BIO_TRANSITION_H_CHUNK", None)
            if arm != "ship":
                TT._TRANSITION_L1_ROWS = arm == "on"
        else:
            os.environ["TT_BIO_TRANSITION_H_CHUNK"] = str(int(arm.lstrip("h")))
        TT.TRANSITION_H_CHUNK_STATS[:] = [0, 0]
        TT.TRANSITION_H_CHUNK_REJECTS.clear()
        TT.TRANSITION_H_CHUNK_SHAPES.clear()
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        rec = {"size": size, "arm": arm, "slot": slot, "rep": rep,
               "loadavg": round(os.getloadavg()[0], 2)}
        t = time.perf_counter()
        try:
            state.predict_one(AB.FIX / f"cdk2x2_{size}.yaml", cfg)
            ttnn.synchronize_device(dev)
            rec["wall_s"] = round(time.perf_counter() - t, 3)
            cifs = sorted(struct_dir.glob("*.cif"))
            rec["digest"] = (hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16]
                             if cifs else None)
        except Exception as e:
            rec["wall_s"] = round(time.perf_counter() - t, 3)
            rec["error"] = str(e)[:600]
            rec["traceback"] = traceback.format_exc()[-800:]
        rec["served"] = TT.TRANSITION_H_CHUNK_STATS[0]
        rec["declined"] = TT.TRANSITION_H_CHUNK_STATS[1]
        rec["heights"] = {f"{s}x{h}@h{n}": c
                          for (s, h, n), c in TT.TRANSITION_H_CHUNK_SHAPES.items()}
        print("  %5d rep%-2d %-6s %-5s %8.3fs load=%5.1f served=%d/%d sha=%s" % (
            size, rep, slot, arm, rec["wall_s"], rec["loadavg"],
            rec["served"], rec["declined"], rec.get("digest")), flush=True)
        return rec

    quiet = {int(k): float(v) for k, v in
             (t.split(":") for t in args.quiet_ship_median.split(",") if t.strip())}

    for size, arm in legs:
        # ship / arm / ship per rep. The two shipped slots are the A/A floor and they sit on
        # either side of the test arm, so a drift in the box shows up as the floor and not as
        # the ratio.
        for rep in range(args.reps):
            for slot, a in (("shipA", args.ref), ("test", arm), ("shipB", args.ref)):
                out["folds"].append(fold(size, a, slot, rep))
                dump()
        sel = lambda s: [f["wall_s"] for f in out["folds"]
                         if f["size"] == size and f["slot"] == s and "error" not in f]
        a1, a2, tt_ = sel("shipA"), sel("shipB"), sel("test")
        if not (a1 and a2 and tt_):
            out["legs"][f"{size}:{arm}"] = {"verdict": "INCOMPLETE"}
            dump()
            continue
        m1, m2, mt = st.median(a1), st.median(a2), st.median(tt_)
        ship = st.median(a1 + a2)
        floor_pct = 100.0 * abs(m1 - m2) / ship
        digests = {f["digest"] for f in out["folds"]
                   if f["size"] == size and "error" not in f}
        out["legs"][f"{size}:{arm}"] = {
            "ship_median_s": round(ship, 3), "arm_median_s": round(mt, 3),
            "shipA_median_s": round(m1, 3), "shipB_median_s": round(m2, 3),
            "fold_seconds_saved": round(ship - mt, 3),
            "ratio": round(ship / mt, 4) if mt else None,
            "aa_floor_pct": round(floor_pct, 3),
            "aa_floor_s": round(abs(m1 - m2), 3),
            "n": {"shipA": len(a1), "shipB": len(a2), "test": len(tt_)},
            "bit_exact_across_arms": len(digests) == 1,
            "digest": sorted(d for d in digests if d),
            "loadavg_span": [min(f["loadavg"] for f in out["folds"] if f["size"] == size),
                             max(f["loadavg"] for f in out["folds"] if f["size"] == size)],
            "quiet_ship_median_s": quiet.get(size),
            "saturation_x": round(ship / quiet[size], 3) if quiet.get(size) else None,
            "verdict": leg_verdict(floor_pct, ship, quiet.get(size), args.saturation_tol),
        }
        dump()
        print("[leg %d:%s] %s" % (size, arm, json.dumps(out["legs"][f"{size}:{arm}"])), flush=True)

    print(json.dumps(out["legs"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
