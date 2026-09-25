"""Score a tap_gate --device stdout against <tree>'s committed af2ig record (log lines before the JSON dropped)."""
import json, sys
from pathlib import Path
tree, *reports = sys.argv[1:]
sys.path.insert(0, str(Path(tree) / "scripts" / "af2_port"))
import device_floor as df
committed = json.loads((Path(tree) / "docs/implementation-parity-data/af2ig-trunk-device.json").read_text())
for r in reports:
    t = Path(r).read_text()
    rep = json.loads(t[t.index("\n{") + 1:] if not t.startswith("{") else t)
    v = df.af2ig_device_floor_verdict(rep, committed)
    fails = [x["tap"] for x in rep["rows"] if x.get("verdict") not in ("PASS", None)]
    pmin = min(x["pcc"] for x in rep["rows"] if x.get("pcc") is not None)
    print(f"{Path(r).stem:22s} {v[0]:5s} failing_taps={len(fails)}/{len(rep['rows'])} pcc_min={pmin:.6f} | {str(v[1])[:160]}")
