#!/usr/bin/env python3
"""Interleaved old-stack/new-stack A/B driver. Runs on the target host, not in either venv.

Two ttnn versions cannot share an interpreter, so the arms are separate processes and the pairing
is the ROUND: one `old` process then one `new` process, repeated, so any drift in host contention
lands on both arms roughly equally instead of on whichever ran first. Each process discards its own
cold fold; only `warm` records are compared.

`--aa` adds a second `old` process per round, which gives the A/A floor from the same data rather
than from a separate session (the floor is then old-vs-old across two process launches, which is
the honest floor for a cross-process comparison -- a within-process floor would flatter the A/B).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path


# Every default-on tt-bio fused kernel, by its env gate.
FUSED_KERNEL_FLAGS = (
    "TT_BIO_TRIATT_PERSISTENT_MASK",
    "TT_BIO_TRIATT_HEAD_MAJOR_QKV",
    "TT_BIO_TRIATT_HEAD_MAJOR_TAIL",
    "TT_BIO_REBLOCK_PERMUTE_BACK",
    "TT_BIO_REBLOCK_PERMUTE_GATED",
    "TT_BIO_TRIMUL_DUAL_NOC",
)


def run_arm(py: Path, script: Path, out: Path, arm: str, rnd: int, folds: int,
            fixture: str, card: str, cifdir: Path | None, repo: Path, log: Path,
            cache: Path) -> int:
    env = dict(os.environ)
    env.update({
        "TT_VISIBLE_DEVICES": card,
        "TT_BIO_LEASE_CARDS": card,
        "TT_BIO_LEASE_HOLDER": "worker:b2z-ttnn-upgrade",
        "PYTHONPATH": str(repo),
        # one JIT cache per stack. The two stacks disagree about kernel flags and about which SFPI
        # compiled them, and the default cache is shared with every other task on this box.
        "TT_METAL_CACHE": str(cache),
    })
    # tt-bio's own fused kernels are off in BOTH arms. They are written against 0.68's LLK headers
    # and do not compile on 0.78 (VectorMode, RoundMode, InputClamping, PackMode and DataCopyType
    # all became typed enums, `mm_block_init_short` and `_sfpu_reciprocal_` are gone), so the arms
    # can only be compared on the stock op path. Turning them off on the OLD arm too is what keeps
    # this an A/B of the stack rather than an A/B of the stack plus our kernels.
    env.update(dict.fromkeys(FUSED_KERNEL_FLAGS, "0"))
    cmd = [str(py), str(script), "--out", str(out), "--arm", arm, "--round", str(rnd),
           "--folds", str(folds), "--fixture", fixture]
    if cifdir:
        cmd += ["--cifdir", str(cifdir)]
    with log.open("a") as fh:
        fh.write(f"\n===== {time.strftime('%H:%M:%S')} arm={arm} round={rnd} =====\n")
        fh.flush()
        return subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(repo))


def summarise(out: Path) -> dict:
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    warm = [r for r in recs if r["kind"] == "warm"]
    by = {}
    for r in warm:
        by.setdefault(r["arm"], []).append(r["fold_s"])
    res = {a: {"n": len(v), "median": round(st.median(v), 4),
               "min": round(min(v), 4), "max": round(max(v), 4),
               "ttnn": next(r["ttnn"] for r in warm if r["arm"] == a)}
           for a, v in sorted(by.items())}
    if "old" in res and "new" in res:
        res["FOLD_RATIO_new_over_old"] = round(res["old"]["median"] / res["new"]["median"], 4)
    if "oldA" in res and "old" in res:
        res["AA_FLOOR"] = round(res["old"]["median"] / res["oldA"]["median"], 4)
    digests = {}
    for r in warm:
        digests.setdefault(r["arm"], set()).add(r["cif_sha256"][:16])
    res["cif_digests"] = {a: sorted(v) for a, v in digests.items()}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("/home/mthuening/work/b2z-ttnn"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--card", default="2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=2, help="timed folds per process")
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--aa", action="store_true", help="add a second old-stack process per round")
    ap.add_argument("--cifdir", type=Path, default=None)
    args = ap.parse_args()

    repo = args.base / "src"
    script = repo / "perf" / "b2z_ttnn" / "stack_fold.py"
    log = args.out.with_suffix(".log")
    summary = args.out.with_suffix(".summary.json")
    order = ["old", "new"] + (["oldA"] if args.aa else [])
    pys = {"old": args.base / "venv-old" / "bin" / "python",
           "new": args.base / "venv-new" / "bin" / "python",
           "oldA": args.base / "venv-old" / "bin" / "python"}

    for rnd in range(1, args.rounds + 1):
        for arm in order:
            t0 = time.time()
            rc = run_arm(pys[arm], script, args.out, arm, rnd, args.folds,
                         args.fixture, args.card, args.cifdir, repo, log,
                         args.base / f"cache-{'new' if arm == 'new' else 'old'}")
            print(f"round {rnd} arm {arm}: rc={rc} {time.time()-t0:.0f}s", flush=True)
            if rc != 0:
                print(f"  !! arm {arm} failed, see {log}", flush=True)
            if args.out.exists():
                summary.write_text(json.dumps(summarise(args.out), indent=1))
    if args.out.exists():
        s = summarise(args.out)
        summary.write_text(json.dumps(s, indent=1))
        print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
