"""Device kernel time of one census fold, by module path and by call site.

perf/spd_census/analyze.py splits a fold into Protenix-v2's phases, which put OpenDDE's expander, refiner and
structural-axis diffusion under "other". This joins the same records (progs_<fold> to ops_<fold> by runtime id)
and sums kernel seconds per registry path prefix and per innermost tt_bio call site, unscaled: the fold as profiled.

usage: census_regs.py RUN_DIR [FOLD] [DEPTH] [TOP]
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1])
FOLD = sys.argv[2] if len(sys.argv) > 2 else "full"
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TOP = int(sys.argv[4]) if len(sys.argv) > 4 else 40

sigs = {}
for line in open(RUN / f"sig_{FOLD}.jsonl"):
    e = json.loads(line)
    if "op" in e:
        sigs[e["sig"]] = e
calls = np.array([json.loads(line) for line in open(RUN / f"ops_{FOLD}.jsonl")], dtype=np.int64)
calls = calls[np.argsort(calls[:, 1], kind="stable")]
P = np.array([[x if x is not None else -1 for x in json.loads(line)][:5]
              for line in open(RUN / f"progs_{FOLD}.jsonl")], dtype=np.float64)
rid = np.floor(P[:, 0] / 1024)
idx = np.searchsorted(calls[:, 1], rid, side="right") - 1
ok = (idx >= 0) & (rid < calls[np.clip(idx, 0, None), 2])
k = P[ok, 3] / 1e9
sig_of = calls[idx[ok], 0]
per_sig = np.bincount(sig_of, weights=k, minlength=max(sigs) + 1)
ncall = np.bincount(calls[:, 0], minlength=max(sigs) + 1)
total = per_sig.sum()
print(f"{RUN} {FOLD}: kernel {total:.1f} s over {ok.sum()} programs ({(~ok).sum()} unattributed)")


def site(e):
    own = [s for s in e["site"] if not s.startswith(("ops.py", "dispatch.py"))]
    return own[-1] if own else e["site"][-1]


by_reg, by_site = collections.Counter(), collections.Counter()
n_site = collections.Counter()
for s, t in enumerate(per_sig):
    if t <= 0 or s not in sigs:
        continue
    e = sigs[s]
    by_reg["/".join(e["reg"].split("/")[:DEPTH]) or "-"] += t
    key = (site(e), e["op"])
    by_site[key] += t
    n_site[key] += int(ncall[s])
print(f"\nby module path (depth {DEPTH}):")
for r, t in by_reg.most_common():
    print(f"  {t:8.1f} s {100 * t / total:5.1f} %  {r}")
print(f"\ntop {TOP} call sites:")
for (st, op), t in by_site.most_common(TOP):
    print(f"  {t:8.1f} s {100 * t / total:5.1f} %  {n_site[(st, op)]:7d} calls  {op:28s} {st}")
