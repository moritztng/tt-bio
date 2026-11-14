"""Exit 0 when SPRE beat GROW at 512 by >= 0.03 ms in every round of a run2 tail.jsonl."""
import json
import sys

r = [json.loads(l) for l in open(sys.argv[1]) if '"ms"' in l]
ms = {(x["round"], x["leg"]): x["ms"] for x in r if x["n"] == 512}
d = [round(ms[(k, "grow=1")] - ms[(k, "grow=1,spre=1")], 4) for k in sorted({k for k, _ in ms})]
print("spre gain at 512 by round", d)
sys.exit(0 if d and min(d) >= 0.03 else 1)
