#!/usr/bin/env python3
"""What TT_BIO_UNFUSED_SILU is worth in FOLD SECONDS (c12-unfused-silu-bh).

The op-level A/B already decided the lever at the op level: the epilogue costs 0.0882 ms/call and
the standalone in-place silu costs 0.0400, so unfusing is worth 0.4339 s/fold at 8,960 calls. This
harness asks the only question that outranks that: does the fold move by it.

Arms alternate INSIDE one process on one device open with the shipped arm run twice per rep, either
side of the test arm, so the session's own A/A floor falls out of the same data and box drift lands
in the floor rather than in the ratio. Both contention detectors from
perf/roof_transition_chunk_bh/foldab.py are reused unchanged, including the saturation check -- a
tight A/A floor means "no jitter", not "not saturated", and a host-bound session hides a real
device-side saving without moving the floor.

The clock is FORCED for the whole session and sampled at 500 Hz in a subprocess, per-fold. On
Blackhole the AICLK sets the fold time, so an unforced arm pair measures the governor.

Each arm's first CIF is archived, so the paired accuracy read comes out of the same process, same
seed and same MSA as the timing. An unpaired RMSD carries the full seed floor; a paired one does not.
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
sys.path.insert(0, str(REPO / "perf" / "roof_transition_chunk_bh"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_flag_levers as AB          # noqa: E402  fixtures, cfg and MSA seeding, unmodified
import clk                           # noqa: E402  ARC clock force + subprocess sampler
from foldab import leg_verdict       # noqa: E402  both contention detectors, unmodified


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifs", type=Path, required=True)
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warm", type=int, default=1, help="untimed folds per arm before rep 0")
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--quiet-ship-median", default="",
                    help="SIZE:SECONDS[,...] the shipped arm's median on a QUIET box")
    ap.add_argument("--saturation-tol", type=float, default=1.10)
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()
    assert "TT_BIO_UNFUSED_SILU" not in os.environ, "the arm is set in-process, not pinned by env"

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
    assert TT._UNFUSED_SILU is False, "the shipped default must be off, else shipA is not shipped"

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    nodes = clk.nodes_open_by_this_process()
    assert len(nodes) == 1, f"expected exactly one open chip, got {nodes}"
    node = nodes[0]
    clk.force(args.mhz, [node])
    g = dev.compute_with_storage_grid_size()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    out = {
        "doc": __doc__,
        "env": {
            "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
            "card_node": node, "clock_forced_mhz": args.mhz,
            "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
            "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loadavg_start": os.getloadavg(),
            "cotenancy": os.environ.get("C12_COTENANCY", ""),
            "protocol": {"sampling_steps": args.steps, "recycling_steps": args.recycles,
                         "seed": AB.SEED, "reps": args.reps, "warm": args.warm},
        },
        "folds": [], "legs": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cifs.mkdir(parents=True, exist_ok=True)

    def dump():
        out["env"]["loadavg_now"] = os.getloadavg()
        args.out.write_text(json.dumps(out, indent=1))
    dump()

    work = Path(tempfile.mkdtemp(prefix="c12-silu-foldab-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for size in sizes:
        AB._seed_msa(AB.FIX / f"cdk2x2_{size}.yaml",
                     (AB.FIX / f"cdk2x2_{size}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("c12-unfused-silu-foldab", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(size, arm, slot, rep, timed=True):
        TT._UNFUSED_SILU = arm == "silu"
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        rec = {"size": size, "arm": arm, "slot": slot, "rep": rep, "timed": timed,
               "loadavg": round(os.getloadavg()[0], 2),
               "unfused_silu": bool(TT._UNFUSED_SILU)}
        sampler = clk.Sampler(node)
        t = time.perf_counter()
        try:
            state.predict_one(AB.FIX / f"cdk2x2_{size}.yaml", cfg)
            ttnn.synchronize_device(dev)
            rec["wall_s"] = round(time.perf_counter() - t, 3)
            cifs = sorted(struct_dir.rglob("*.cif"))
            if cifs:
                rec["digest"] = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16]
                dst = args.cifs / f"{size}_{arm}_{slot}_r{rep}.cif"
                if not dst.exists():
                    shutil.copy(cifs[0], dst)
                rec["cif"] = dst.name
        except Exception as e:
            rec["wall_s"] = round(time.perf_counter() - t, 3)
            rec["error"] = str(e)[:600]
            rec["traceback"] = traceback.format_exc()[-800:]
        rec["aiclk"] = sampler.stop()
        print("  %5d rep%-2d %-6s %-5s %8.3fs load=%5.1f clk=%d-%d sha=%s%s" % (
            size, rep, slot, arm, rec["wall_s"], rec["loadavg"],
            rec["aiclk"]["min"], rec["aiclk"]["max"], rec.get("digest"),
            "" if timed else "  (warm, untimed)"), flush=True)
        return rec

    quiet = {int(k): float(v) for k, v in
             (t.split(":") for t in args.quiet_ship_median.split(",") if t.strip())}

    for size in sizes:
        for w in range(args.warm):
            for a in ("ship", "silu"):
                out["folds"].append(fold(size, a, "warm", w, timed=False))
                dump()
        for rep in range(args.reps):
            for slot, a in (("shipA", "ship"), ("test", "silu"), ("shipB", "ship")):
                out["folds"].append(fold(size, a, slot, rep))
                dump()
        sel = lambda s: [f["wall_s"] for f in out["folds"] if f["size"] == size
                         and f["timed"] and f["slot"] == s and "error" not in f]
        a1, a2, tt_ = sel("shipA"), sel("shipB"), sel("test")
        if not (a1 and a2 and tt_):
            out["legs"][str(size)] = {"verdict": "INCOMPLETE"}
            dump()
            continue
        m1, m2, mt = st.median(a1), st.median(a2), st.median(tt_)
        ship = st.median(a1 + a2)
        floor_pct = 100.0 * abs(m1 - m2) / ship
        timed = [f for f in out["folds"] if f["size"] == size and f["timed"] and "error" not in f]
        clocks = [f["aiclk"] for f in timed]
        out["legs"][str(size)] = {
            "ship_median_s": round(ship, 3), "arm_median_s": round(mt, 3),
            "shipA_median_s": round(m1, 3), "shipB_median_s": round(m2, 3),
            "fold_seconds_saved": round(ship - mt, 3),
            "ratio": round(ship / mt, 4) if mt else None,
            "aa_floor_pct": round(floor_pct, 3), "aa_floor_s": round(abs(m1 - m2), 3),
            "n": {"shipA": len(a1), "shipB": len(a2), "test": len(tt_)},
            "aiclk_min": min(c["min"] for c in clocks),
            "aiclk_max": max(c["max"] for c in clocks),
            "digest_ship": sorted({f["digest"] for f in timed if f["arm"] == "ship"}),
            "digest_silu": sorted({f["digest"] for f in timed if f["arm"] == "silu"}),
            "bit_exact_within_arm": all(
                len({f["digest"] for f in timed if f["arm"] == a}) == 1 for a in ("ship", "silu")),
            "loadavg_span": [min(f["loadavg"] for f in timed), max(f["loadavg"] for f in timed)],
            "quiet_ship_median_s": quiet.get(size),
            "saturation_x": round(ship / quiet[size], 3) if quiet.get(size) else None,
            "verdict": leg_verdict(floor_pct, ship, quiet.get(size), args.saturation_tol),
        }
        dump()
        print("[leg %d] %s" % (size, json.dumps(out["legs"][str(size)])), flush=True)

    clk.release()
    print("LEGS " + json.dumps(out["legs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
