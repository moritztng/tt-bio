"""The S2b kill rule, applied by the runner so the fold A/B cannot run on a failed screen.

Pre-committed in state/boltz2-sizes-perf.md S2: at (768,4,768,32), under 1.10x off-fold or
torch.equal false is NO-GO and nothing gets built on top of it.
"""
import json
import sys

try:
    d = json.load(open("perf/b2sizes/s2b_screen.json"))
except Exception as e:                                                       # noqa: BLE001
    print(f"NOFILE {type(e).__name__}")
    sys.exit(0)

r = [x for x in d.get("runs", []) if x.get("N") == 768]
if not r:
    print("NO768")
    sys.exit(0)
r = r[0]
ok = r["speedup"] >= 1.10 and r["torch_equal"]
print(f"{'GO' if ok else 'NOGO'} {r['speedup']:.4f} equal={r['torch_equal']}")
