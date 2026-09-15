#!/usr/bin/env python3
"""ONE fold at one forced Transition row height, in its own process, with a digest.

foldab.py is the A/B harness and it runs the reference arm on both sides of every test arm, so a
three-fold leg costs minutes and the reference slot itself wedges whenever the shipped default is
the arm under test. To find the height at which the 1024 aa fold stops finishing, what is needed is
the cheapest possible "does this height finish, and with which bytes" -- one fold, one process, one
line of output. A process that wedges is killed by the caller and costs nothing else.

    hfold.py --size 1024 --arm h20 --steps 2 --out out/h20.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--arm", default="ship", help="ship | off | on | hNN")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

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
    if args.arm in ("off", "on"):
        os.environ.pop("TT_BIO_TRANSITION_H_CHUNK", None)
        TT._TRANSITION_L1_ROWS = args.arm == "on"
    elif args.arm != "ship":
        os.environ["TT_BIO_TRANSITION_H_CHUNK"] = str(int(args.arm.lstrip("h")))

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    work = Path(tempfile.mkdtemp(prefix="hfold-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    AB._seed_msa(AB.FIX / f"cdk2x2_{args.size}.yaml",
                 (AB.FIX / f"cdk2x2_{args.size}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    rec = {"doc": __doc__, "size": args.size, "arm": args.arm,
           "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
           "steps": args.steps, "recycles": args.recycles,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg_start": os.getloadavg()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1))

    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("hfold", cfg)
    rec["model_load_s"] = round(time.perf_counter() - t0, 3)
    TT.TRANSITION_H_CHUNK_STATS[:] = [0, 0]
    TT.TRANSITION_H_CHUNK_SHAPES.clear()
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    try:
        state.predict_one(AB.FIX / f"cdk2x2_{args.size}.yaml", cfg)
        ttnn.synchronize_device(dev)
        rec["wall_s"] = round(time.perf_counter() - t, 3)
        cifs = sorted(struct_dir.glob("*.cif"))
        rec["digest"] = (hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16]
                         if cifs else None)
        rec["verdict"] = "DONE" if rec["digest"] else "NO-CIF"
    except BaseException as e:
        rec["wall_s"] = round(time.perf_counter() - t, 3)
        rec["error"] = str(e)[:800]
        rec["traceback"] = traceback.format_exc()[-1200:]
        rec["verdict"] = "ERROR"
    rec["served"] = TT.TRANSITION_H_CHUNK_STATS[0]
    rec["declined"] = TT.TRANSITION_H_CHUNK_STATS[1]
    rec["heights"] = {f"{s}x{h}@h{n}": c for (s, h, n), c in TT.TRANSITION_H_CHUNK_SHAPES.items()}
    rec["loadavg_end"] = os.getloadavg()
    args.out.write_text(json.dumps(rec, indent=1))
    print("HFOLD %d %s %s %.3fs sha=%s served=%d/%d" % (
        args.size, args.arm, rec["verdict"], rec["wall_s"], rec.get("digest"),
        rec["served"], rec["declined"]), flush=True)
    return 0 if rec["verdict"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
