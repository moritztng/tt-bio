"""Which FAIL cells in a capacity report would 8e3a3e9e's guard actually change?

The guard (capacity_gate.py:1308) rewrites a bisect's verdict only on the path where no
rung allocated AND no screen leg at a walked rung reported an allocator mechanism. A cell
that already carries a real allocator refusal gets the identical note from the fixed code,
so re-measuring it reproduces its own result at full cost. opendde's bisect walks at the
RESIDENCY tier at roughly 12 min a rung, so the difference is hours.

Prints one model per line, the ones that need re-measuring on the fixed tree.
"""
import json
import sys

ALLOC_MECHANISMS = ("dram", "l1")
STALE_NOTE = "shapes do not allocate at any size walked"

report = json.load(open(sys.argv[1]))
bar = report["bar_tokens"]

for r in report["results"]:
    if r.get("verdict") != "FAIL":
        continue
    note = r.get("alloc_ceiling_note") or ""
    if not note.startswith(STALE_NOTE):
        # Either a real ceiling was found, or no rung was walked at all. The fixed code
        # takes the same branch and writes the same thing.
        continue
    # Mirror the guard's own test over the bisect rungs, which are the screen legs below
    # the bar. If any of them saw the allocator refuse, the note it has is the note the
    # fixed code would write.
    screens = [l for l in (r.get("legs") or [])
               if l.get("tier") == "screen" and (l.get("tokens") or bar) < bar]
    if any(l.get("mechanism") in ALLOC_MECHANISMS for l in screens):
        continue
    print(r["model"])
