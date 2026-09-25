#!/usr/bin/env python3
"""`of3t-stepqb2`'s `steprun.py`, pointed at this row's namespace and given the one knob it
does not have: `TRIM_HOST_HEAP`.

Nothing here is a harness. `steprun.py` is imported and its `main()` is called; this file
sets `OUT`, sets `tt_bio.autograd.TRIM_HOST_HEAP` before the first taped call, and records
which way the knob was set into the artifact. The knob is a module constant on the engine
(`of3t-tapemem`'s fix, unmerged), so the only way to take a pair is to set it in-process --
an env var would be an engine edit and this row lands nothing.

The venue moved from qb1 to pc when qb1 went dark, so every number here is pc card 0, p150a,
PCIe Gen4 x8, and `usable = MemTotal - Hugetlb` = 27.502 GiB rather than 30.5.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    argv = sys.argv[1:]
    trim = True
    if "--trim" in argv:
        i = argv.index("--trim")
        trim = argv[i + 1].lower() in ("1", "on", "true", "yes")
        del argv[i:i + 2]

    from tt_bio import autograd as ag
    if not hasattr(ag, "TRIM_HOST_HEAP"):
        raise SystemExit("this tree has no TRIM_HOST_HEAP: apply of3t-tapemem's 33e5f1eb0 first")
    ag.TRIM_HOST_HEAP = trim
    print(f"[run1] TRIM_HOST_HEAP={ag.TRIM_HOST_HEAP} "
          f"malloc_trim_available={ag._trim_host_heap()!r}", flush=True)

    import perf.of3t_stepqb2.steprun as SR
    SR.OUT = REPO / "perf" / "of3t_stepqb1" / "out"
    SR.OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = ["run1.py"] + argv
    t0 = time.perf_counter()
    rc = SR.main()
    wall = round(time.perf_counter() - t0, 3)

    # stamp the knob and the tree onto every artifact this invocation wrote
    tag = None
    for i, a in enumerate(argv):
        if a == "--tag":
            tag = argv[i + 1]
    if tag is None:
        tag = "384"
        for i, a in enumerate(argv):
            if a == "--tokens":
                tag = argv[i + 1]
    import hashlib
    stamp = {"row": "of3t-stepqb1",
             "trim_host_heap": trim,
             "autograd_sha256": hashlib.sha256(
                 (REPO / "tt_bio" / "autograd.py").read_bytes()).hexdigest(),
             "tree": os.popen("git -C %s rev-parse HEAD" % REPO).read().strip(),
             "tree_note": "origin/main + of3t-tapemem 33e5f1eb0's autograd.py hunk, "
                          "cherry-picked, uncommitted (measurement row lands no engine file)",
             "wall_s": wall,
             "meminfo_at_end": {k: v for k, v in (
                 (l.split(":")[0], l.split(":")[1].strip())
                 for l in Path("/proc/meminfo").read_text().splitlines())
                 if k in ("MemTotal", "MemAvailable", "Hugetlb")}}
    p = SR.OUT / f"RUN_{tag}.json"
    if p.is_file():
        d = json.loads(p.read_text())
        d["stepqb1"] = stamp
        p.write_text(json.dumps(d, indent=1, default=str))
    (SR.OUT / f"STAMP_{tag}.json").write_text(json.dumps(stamp, indent=1))
    print(f"[run1] rc={rc} trim={trim} wall={wall}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
