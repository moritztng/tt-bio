#!/usr/bin/env python3
"""One fold arm of the three-arm A/B: ship, transcription, transcription+split.

Arms are whole processes, alternated by the driver, because between-process spread is 2 % against
0.02-0.16 % within an arm, so more processes per arm beats more folds per process.  Each record
carries the routing counts, the md5 of the CIF the fold wrote, and -- when --clock is asked for --
the AICLK sampled DURING the fold.

The op is bit-exact, so all three arms must write the same digest.  A digest mismatch is grounds
for NO-GO whatever the timing says.

Without --clock this is a correctness run: it neither forces nor samples the clock, and it says so
in the record, so a digest check can be taken on a busy host while a timing arm cannot.
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
sys.path.insert(0, str(ROOT / "perf" / "other512"))
sys.path.insert(0, str(HERE))

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
    return cifs[0].name, hashlib.md5(cifs[0].read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("ship", "tr", "split"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=2, help="timed folds after the cold one")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--clock", type=int, default=0, help="0 = correctness run, do not force")
    ap.add_argument("--no-selfcheck", action="store_true",
                    help="do not compare the first call of each signature against the native op")
    ap.add_argument("--verify", type=int, default=0,
                    help="compare routed against native for the first N calls of each signature")
    ap.add_argument("--wide", action="store_true",
                    help="also route L1-interleaved operands; moves the digest, see mm1droute")
    ap.add_argument("--out", default=str(HERE / "mm1dfold.jsonl"))
    a = ap.parse_args()

    import tt_bio
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    from fold_ab_multi import patch_boltz2_cfg
    patch_boltz2_cfg()
    assert Path(tt_bio.__file__).resolve().is_relative_to(ROOT), (
        "imported tt_bio from %s, not this worktree" % tt_bio.__file__)

    import mm1droute
    from tt_bio import tenstorrent as T

    fixdir = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, *_ = B.build_fold(
        a.model, ROOT / (".msa_mm1dfold_%s_%d" % (a.model, a.size)),
        fixdir / ("cdk2x2_%d.yaml" % a.size), fixdir / ("cdk2x2_%d.a3m" % a.size))
    struct_dir = Path(meta["struct_dir"])

    held = []
    if a.clock:
        held = clk.force(a.clock, clk.nodes_open_by_this_process())
        t0 = time.time()
        while time.time() - t0 < 10.0 and not all(clk.aiclk(n) >= a.clock - 5 for n in held):
            time.sleep(0.01)
        reached = {n: clk.aiclk(n) for n in held}
        if any(v < a.clock - 5 for v in reached.values()):
            print("REFUSING to measure: %r" % (reached,))
            return 2
        print("nodes %r forced to %d MHz" % (held, a.clock), flush=True)

    rec = {"tag": a.tag, "arm": a.arm, "host": socket.gethostname(), "model": a.model,
           "size": a.size, "pid": os.getpid(), "t_start": time.time(),
           "clock_forced": a.clock or None, "wide": bool(a.wide),
           "git_head": os.popen("git -C %s rev-parse HEAD" % ROOT).read().strip(),
           "folds": []}

    if a.arm != "ship":
        mm1droute.install(T.get_device(), split=(a.arm == "split"), wide=a.wide,
                          verify=a.verify, selfcheck=not a.no_selfcheck)

    for i in range(a.folds + 1):
        cs = ClockSampler(held) if held else None
        if cs:
            cs.start()
        fold_s, m = one_fold()
        clock = cs.take() if cs else None
        name, digest = cif_md5(struct_dir)
        rec["folds"].append({"i": i, "cold": i == 0, "fold_s": round(fold_s, 3),
                             "clock": clock, "cif": name, "md5": digest,
                             "plddt": m.get("plddt"), "n_tokens": m.get("n_tokens")})
        print("  %-8s fold %d %s %.3f s  clk %r  md5 %s"
              % (a.tag, i, "COLD" if i == 0 else "warm", fold_s, clock, digest), flush=True)

    counts, shapes = mm1droute.stats()
    vd = mm1droute.verdicts()
    if vd:
        rec["verify"] = [{"a": list(k[0]), "b": list(k[1]), **v} for k, v in vd.items()]
        bad = [x for x in rec["verify"] if x["differ"]]
        print("verify: %d signatures checked, %d differ" % (len(vd), len(bad)), flush=True)
        for x in bad:
            print("   DIFFERS a=%s b=%s  mem a/b/out %s/%s/%s  max|diff| %.3e  shapes %s vs %s"
                  % (x["a"], x["b"], x["mem_a"], x["mem_b"], x["mem_out"], x["max_abs"],
                     x["shape_native"], x["shape_routed"]), flush=True)
    rec["route"] = counts
    rec["route_shapes"] = [{"a": list(k[0]), "b": list(k[1]), "in0_block_w": k[2],
                            "per_core_M": k[3], "per_core_N": k[4], "n": v}
                           for k, v in sorted(shapes.items(), key=lambda kv: -kv[1])]
    mm1droute.remove()

    warm = [f["fold_s"] for f in rec["folds"] if not f["cold"]]
    rec["warm_median_s"] = round(st.median(warm), 3) if warm else None
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("%s warm median %s s  routed %s" % (a.tag, rec["warm_median_s"],
                                              counts.get("routed", 0)), flush=True)
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
        print("   %-34s %d" % (k, v), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
