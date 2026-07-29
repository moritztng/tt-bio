#!/usr/bin/env python3
"""Which (target, generator) pairs are not going to finish on their own?

The publish preflight requires every generator to have the same fold count, so one pair that never
succeeds blocks the release. Drivers walk their slice once and exit; the supervisor relaunches only
when the host is fully idle, and retries are capped. So a pair that has failed repeatedly needs a
named reason now, not at assembly time.

Reports, for every pair that has NO ok record: how many attempts, the statuses seen, and the last
error -- so the difference between "not attempted yet" and "attempted and always fails" is explicit.
"""
import json
import socket
from collections import Counter, defaultdict
from pathlib import Path

TIER = Path.home() / "abag_xm" / "tier_a"
recs = [json.loads(l) for l in open(TIER / "progress.jsonl") if l.strip()]

by_pair = defaultdict(list)
for r in recs:
    by_pair[(r.get("target"), r.get("model"))].append(r)

ok = {k for k, v in by_pair.items() if any(x.get("status") == "ok" for x in v)}
never = {k: v for k, v in by_pair.items() if k not in ok}

print(f"host {socket.gethostname()}: {len(by_pair)} pairs attempted, {len(ok)} ok, "
      f"{len(never)} never ok")
if not never:
    print("nothing outstanding among attempted pairs")
else:
    print(f"\n{'target':7}{'generator':15}{'tries':>6}  statuses / last error")
    for (t, g), v in sorted(never.items(), key=lambda kv: -len(kv[1])):
        c = Counter(x.get("status") for x in v)
        last = v[-1]
        err = (last.get("stderr") or "")
        lines = [ln for ln in err.splitlines() if ln.strip()]
        tail = lines[-1][:90] if lines else "(no stderr captured)"
        print(f"{t:7}{g:15}{len(v):>6}  {dict(c)}")
        print(f"{'':28}  {tail}")

# Repeated failures among pairs that DID eventually succeed are worth seeing too: they cost card time
# and indicate a flaky target rather than a broken one.
flaky = {k: v for k, v in by_pair.items()
         if k in ok and sum(1 for x in v if x.get("status") != "ok") >= 2}
print(f"\n{len(flaky)} pairs succeeded only after >=2 failures (cost card time, not blocking):")
for (t, g), v in sorted(flaky.items(), key=lambda kv: -len(kv[1]))[:8]:
    c = Counter(x.get("status") for x in v)
    print(f"  {t:7}{g:15} {dict(c)}")
