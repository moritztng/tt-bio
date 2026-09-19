#!/usr/bin/env python3
"""One fold arm of the writer-split A/B, 512 aa Boltz-2, on the patched tt-metal v0.67.4 build.

TTNN_MM2D_WRITER_ON_IN0 is read when the matmul program is built, so an arm is a whole process:
this script runs one arm and appends its record to a shared JSONL, and the driver alternates
arms A B A B A by launching it repeatedly.  The first fold of every process is cold (JIT plus
program cache) and is discarded; the timed folds follow in the same process.

Each record carries the AICLK sampled DURING the fold and the md5 of the CIF the fold wrote.
The op is bit-exact, so every arm's digest must match; a mismatch is grounds for NO-GO whatever
the timing says.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics as st
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "writeside"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

import clk  # noqa: E402


class ClockSampler(threading.Thread):
    def __init__(self, nodes):
        super().__init__(daemon=True)
        self.nodes, self.stop, self.vals = nodes, threading.Event(), []

    def run(self):
        while not self.stop.is_set():
            for n in self.nodes:
                v = clk.aiclk(n)
                if v:
                    self.vals.append(v)
            time.sleep(0.05)

    def take(self):
        self.stop.set()
        self.join(timeout=2)
        if not self.vals:
            return None
        return {"n": len(self.vals), "min": min(self.vals), "max": max(self.vals),
                "mean": round(st.mean(self.vals), 1)}


def cif_md5(struct_dir: Path):
    cifs = sorted(struct_dir.rglob("*.cif"))
    if not cifs:
        return None, None
    h = hashlib.md5()
    h.update(cifs[0].read_bytes())
    return cifs[0].name, h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", required=True, choices=("0", "1"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=2, help="timed folds after the cold one")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--out", default=str(HERE / "wsfold.jsonl"))
    a = ap.parse_args()

    os.environ["TTNN_MM2D_WRITER_ON_IN0"] = a.flag

    import tt_bio
    import tt_baseline as B
    # build_fold's cfg carries no Boltz-2 hyperparameters, so load_model raises
    # KeyError('conf_kwargs').  perf/other512 already owns that injection.
    # tt_baseline resolves the protocol per model now, so the module-level constants
    # fold_ab_multi reads have to be set first (progtape.py does the same).
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    from fold_ab_multi import patch_boltz2_cfg
    patch_boltz2_cfg()
    assert Path(tt_bio.__file__).resolve().is_relative_to(ROOT), (
        "imported tt_bio from %s, not this worktree" % tt_bio.__file__)

    fixdir = ROOT / "perf" / "size512" / "fixtures"
    tgt = fixdir / ("cdk2x2_%d.yaml" % a.size)
    a3m = fixdir / ("cdk2x2_%d.a3m" % a.size)
    one_fold, meta, *_rest = B.build_fold(a.model, ROOT / (".msa_wsfold_%s_%d" % (a.model, a.size)), tgt, a3m)
    struct_dir = Path(meta["struct_dir"])

    held = clk.force(a.clock, clk.nodes_open_by_this_process())
    t0 = time.time()
    while time.time() - t0 < 10.0 and not all(clk.aiclk(n) >= a.clock - 5 for n in held):
        time.sleep(0.01)

    rec = {"tag": a.tag, "flag": a.flag, "host": socket.gethostname(), "model": a.model,
           "size": a.size, "pid": os.getpid(), "t_start": time.time(),
           "git_head": os.popen("git -C %s rev-parse HEAD" % ROOT).read().strip(),
           "ttnn": __import__("ttnn").__file__, "folds": []}

    for i in range(a.folds + 1):
        cs = ClockSampler(held)
        cs.start()
        fold_s, m = one_fold()
        clock = cs.take()
        name, digest = cif_md5(struct_dir)
        rec["folds"].append({"i": i, "cold": i == 0, "fold_s": round(fold_s, 3),
                             "clock": clock, "cif": name, "md5": digest,
                             "plddt": m.get("plddt"), "n_tokens": m.get("n_tokens")})
        print("  %-8s fold %d %s %.3f s  clk %r  md5 %s"
              % (a.tag, i, "COLD" if i == 0 else "warm", fold_s, clock, digest), flush=True)

    warm = [f["fold_s"] for f in rec["folds"] if not f["cold"]]
    rec["warm_median_s"] = round(st.median(warm), 3) if warm else None
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("%s warm median %s s" % (a.tag, rec["warm_median_s"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
