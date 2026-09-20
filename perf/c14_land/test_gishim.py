#!/usr/bin/env python3
"""Device-free proof that the firing shim records what it claims, and reports nothing when blind.

The shim is the thing that turns a green gate from unfalsifiable into evidence, so it needs its own
evidence. Every case runs a real child interpreter with the shim on PYTHONPATH, because that is how
it is used: `tt_bio.main predict` folds in a spawn child and a wrap installed in the driver sees
nothing (`in-process-patch-never-reaches-a-spawn-child`).

The last two cases are the ones that earn their keep. They BREAK what the check reads
(`negative-control-must-break-what-check-reads`): a module that is never imported must record
`counter: null` rather than a zero that would read as "the lever declined everything", and a run
with neither variable set must record nothing at all rather than a silent pass.

    python3 perf/c14_land/test_gishim.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM = HERE / "gishim"


def run(code: str, env_extra: dict, tmp: Path) -> list:
    out = tmp / f"rec{len(list(tmp.glob('rec*')))}"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SHIM}:{tmp}"
    env["C14_GI_OUT"] = str(out)
    env.pop("C14_GI_WRAP", None)
    env.pop("C14_GI_COUNTER", None)
    env.update(env_extra)
    rc = subprocess.call([sys.executable, "-c", code], env=env, cwd=str(tmp))
    assert rc == 0, f"child exited {rc}"
    return [json.load(open(f)) for f in sorted(glob.glob(str(out) + ".*"))]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gishim-test-"))
    (tmp / "faketarget.py").write_text(
        "STATS = [0, 0]\n"
        "def decide(on):\n"
        "    STATS[0 if on else 1] += 1\n"
        "    return on\n")
    fails = []

    # 1. counter mode records the live list
    recs = run("import faketarget as f\n[f.decide(True) for _ in range(5)]\n"
               "[f.decide(False) for _ in range(2)]\n",
               {"C14_GI_COUNTER": "faketarget:STATS"}, tmp)
    if not (len(recs) == 1 and recs[0]["counter"] == [5, 2]):
        fails.append(f"1 counter mode: expected [5,2], got {[r.get('counter') for r in recs]}")

    # 2. wrap mode counts calls
    recs = run("import faketarget as f\n[f.decide(True) for _ in range(9)]\n",
               {"C14_GI_WRAP": "faketarget:decide"}, tmp)
    if not (len(recs) == 1 and recs[0]["calls"] == 9 and recs[0]["patched"]):
        fails.append(f"2 wrap mode: expected 9 patched calls, got {recs}")

    # 3. both together, and TT_BIO_* env is carried so a run records the config it scored
    recs = run("import faketarget as f\n[f.decide(True) for _ in range(3)]\n",
               {"C14_GI_WRAP": "faketarget:decide",
                "C14_GI_COUNTER": "faketarget:STATS",
                "TT_BIO_APB_CONCAT_HEADS": "1"}, tmp)
    if not (len(recs) == 1 and recs[0]["calls"] == 3 and recs[0]["counter"] == [3, 0]
            and recs[0]["env_flags"].get("TT_BIO_APB_CONCAT_HEADS") == "1"):
        fails.append(f"3 combined: got {recs}")

    # 4. NEGATIVE CONTROL: module never imported -> counter is null, NOT [0, 0].
    #    A zero here would read as "the lever was reached and declined everything", which is the
    #    opposite conclusion from "this process never loaded the lever at all".
    recs = run("pass\n", {"C14_GI_COUNTER": "faketarget:STATS"}, tmp)
    if not (len(recs) == 1 and recs[0]["counter"] is None):
        fails.append(f"4 blind counter: expected null, got {[r.get('counter') for r in recs]}")

    # 5. NEGATIVE CONTROL: neither variable set -> the shim writes nothing.
    #    Without this, an empty result set would be indistinguishable from a clean run.
    recs = run("import faketarget as f\nf.decide(True)\n", {}, tmp)
    if recs:
        fails.append(f"5 disabled: expected no records, got {len(recs)}")

    for f in fails:
        print("FAIL", f)
    print(f"{5 - len(fails)} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
