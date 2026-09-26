"""Print the trunk backward's op-class table from a fullstep --node-prof artifact."""
import json
import sys
from collections import defaultdict

d = json.load(open(sys.argv[1]))
want = [int(x) for x in sys.argv[2:]] or None

print("CLOCK:", d["env"].get("aiclk_during"))
for i, r in enumerate(d.get("reps") or d.get("rows", [])):
    if want and i not in want:
        continue
    np_ = r.get("node_prof")
    print(f"\n=== rep {i}  step {r.get('step_s')} s  trunk_bwd {r.get('trunk_backward_s')} s  "
          f"nodes {r.get('trunk_tape_nodes')}  chunk_bwd {r.get('chunk_backward_s')} s ===")
    if not np_:
        continue
    for k in ("nodes_wrapped", "groups_wrapped", "nested_backwards", "synced_per_node",
              "billed_self_s", "wall_s", "unbilled_s", "by_depth"):
        print(f"  {k}: {np_[k]}")
    print(f"\n  {'op':40s} {'calls':>8s} {'self_s':>9s} {'%':>6s} {'total_s':>9s}")
    wall = np_.get("wall_s") or 1.0
    for o in np_["ops"]:
        if o["self_s"] < 0.0005:
            continue
        print(f"  {o['op']:40s} {o['calls']:8d} {o['self_s']:9.3f} "
              f"{100 * o['self_s'] / wall:5.1f}% {o['total_s']:9.3f}")
    print(f"  {'op classes':40s} {len(np_['ops']):8d}")

    # collapsed over depth: the op, wherever it fired
    agg = defaultdict(lambda: [0, 0.0])
    for o in np_["ops"]:
        k = o["op"].split(":", 1)[1]
        agg[k][0] += o["calls"]
        agg[k][1] += o["self_s"]
    print(f"\n  --- collapsed over depth ---")
    for k, (c, s) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        if s < 0.0005:
            continue
        print(f"  {k:40s} {c:8d} {s:9.3f} {100 * s / wall:5.1f}%")
