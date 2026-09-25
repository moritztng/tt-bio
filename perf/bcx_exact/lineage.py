#!/usr/bin/env python3
"""Which trees carry exact_training, and therefore which published numbers were measured with it.

Card-free, and it is the leg that resolved this row's contradiction. `ladder.py` measured 53.7 s
of host float64 arithmetic per 48-block forward at n=224 against `bcx-round`'s measured 1.13 s
taped forward at the same n. Both numbers are right. They are about different trees.

`_EXACT_TRAINING = [True]` landed in 502ed112e, 2026-09-23 03:59:45 +0000, on the main lineage.
BindCraft 2's whole device-measurement lineage -- bcx-round, bcx-predictor, bcx-tracewire,
bcx-stack, bcx-nan, bcx-maskbias -- does not descend from it and contains no exact_training at
all, so no BindCraft 2 number this campaign published was measured with the instrument in the
loop. origin/main does, and `tt_bio/bindcraft2.py` opens `tape()` and calls `ag.backward` at
the module default with no `exact_training(False)` anywhere, so BindCraft 2 running on main is
armed.

Run it and it reprints the table from live git rather than from this docstring.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

DEFAULT_ON = "502ed112e"        # 'training tape: exact softmax and layer norm on by default'
REFS = ["main", "wk/bcx-round", "wk/bcx-predictor", "wk/bcx-tracewire", "wk/bcx-stack",
        "wk/bcx-nan", "wk/bcx-maskbias", "wk/bcx-throughput", "wk/bcx-exact",
        "wk/of3t-bwattrib"]


def git(*a):
    r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
    return r.stdout.strip(), r.returncode


def resolve(ref):
    for cand in (f"origin/{ref}", ref):
        sha, rc = git("rev-parse", "-q", "--verify", cand)
        if rc == 0:
            return cand, sha
    return None, None


def row(ref):
    name, sha = resolve(ref)
    if name is None:
        return {"ref": ref, "resolved": None}
    _, rc = git("merge-base", "--is-ancestor", DEFAULT_ON, name)
    src, _ = git("show", f"{name}:tt_bio/autograd.py")
    bc, _ = git("show", f"{name}:tt_bio/bindcraft2.py")
    when, _ = git("log", "-1", "--format=%ci", name)
    return {"ref": ref, "resolved": name, "sha": sha[:9], "committed": when,
            "descends_from_default_on": rc == 0,
            "EXACT_TRAINING_occurrences": src.count("EXACT_TRAINING"),
            "default_on_line": "_EXACT_TRAINING = [True]" in src,
            "bindcraft2_py": bool(bc),
            "bindcraft2_opens_tape": "taped.tape()" in bc,
            "bindcraft2_turns_it_off": "exact_training" in bc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "LINEAGE.json"))
    a = ap.parse_args()
    when, _ = git("log", "-1", "--format=%H %ci %s", DEFAULT_ON)
    rows = [row(r) for r in REFS]
    blob = {"host": os.uname().nodename, "when_utc": time.strftime("%FT%TZ", time.gmtime()),
            "opened_a_device": False, "default_on_commit": when, "rows": rows}
    pathlib.Path(a.out).write_text(json.dumps(blob, indent=1))
    print(when)
    for r in rows:
        print(json.dumps(r))
    print(f"-> {a.out}")
    armed = [r for r in rows if r.get("default_on_line")]
    print(f"armed trees: {len(armed)} of {len(rows)}")


if __name__ == "__main__":
    main()
