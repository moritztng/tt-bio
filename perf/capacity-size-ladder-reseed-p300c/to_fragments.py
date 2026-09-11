"""Move this branch's p300c rows out of the monolith into per-model fragments.

main converted the size ladder to one fragment per model (1358a695). A fragment REPLACES the
monolith's model entry at read time (_size_ladder_merge_models ends in an unconditional
.update), so a monolith row loses to a fragment for the same (card, model) whatever its date.
This branch recorded four models into the monolith, and main carries stale 4-rung p300c
fragments for three of them, so a merge would drop those measurements without a conflict.

Re-measures nothing: each entry is read from this branch's own monolith and written through the
gate's own writer, then popped from the monolith, which is exactly what record mode does when
fragment=True (release_gate.py:3444-3448).
"""
import importlib.util, json, sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1
            else "/home/ttuser/.coworker/wt/capacity-size-ladder-reseed-p300c")
CARD = "p300c"
spec = importlib.util.spec_from_file_location("rg", REPO / "scripts" / "release_gate.py")
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)

baseline = REPO / "docs" / "size_ladder_baseline.json"
data = json.loads(baseline.read_text())
block = data["cards"][CARD]
stamp = {k: v for k, v in block.items() if k != "models"}
want = [m for m, e in block["models"].items()
        if set(str(r) for r in rg._size_ladder_model_rungs(m))
        <= (set(e.get("runtime_s") or {}) | set(e.get("refused") or {}))]
if not want:
    print("nothing to move: no complete p300c monolith row"); sys.exit(0)
for m in sorted(want):
    entry = block["models"][m]
    frag = rg._size_ladder_write_fragment(baseline, CARD, stamp, m, entry)
    block["models"].pop(m, None)
    print(f"  moved {CARD}/{m} -> {frag.name} "
          f"({len(entry.get('runtime_s') or {})} rungs, recorded {entry.get('recorded')})")
baseline.write_text(json.dumps(data, indent=2) + "\n")
print(f"moved {len(want)} row(s); monolith p300c now holds {sorted(block['models'])}")
