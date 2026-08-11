"""A 512 aa protenix-v2 fold through tt_baseline, with reblock_permute's own accounting printed.

Exec step 2's accept criterion is not "the gate opens" -- it is that a live fold actually SERVES the
channel moves the widened window is supposed to catch. A window that opens and still declines every
call is the failure this exists to detect, so STATS (served, declined) and REJECTS come out of the
same process that produced the fold time.
"""
import argparse, importlib.util, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tt_bio.reblock_permute as rp   # noqa: E402

spec = importlib.util.spec_from_file_location("tt_baseline", REPO / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--l1-nmax", type=int, default=None,
                help="override reblock_permute.L1_N_MAX for this arm (the A/B knob)")
a = ap.parse_args()
if a.l1_nmax is not None:
    rp.L1_N_MAX = a.l1_nmax

fix = REPO / "perf" / "size512" / "fixtures"
res = tb.measure("protenix-v2", a.repeat, REPO / ".msa_s512_512", Path(a.out),
                 fix / "cdk2x2_512.yaml", fix / "cdk2x2_512.a3m", "512 aa")

served, declined = rp.STATS
rej = {f"{k[0]}|{list(k[1])}": v for k, v in rp.REJECTS.items()}
extra = {"l1_n_max": rp.L1_N_MAX, "l1_n_min": rp.L1_N_MIN, "reblock_served": served, "reblock_declined": declined, "reblock_rejects": rej,
         "reblock_enabled": rp._ENABLED}
d = json.load(open(a.out)); d.update(extra); json.dump(d, open(a.out, "w"), indent=2)
print(f"\nreblock_permute: served {served}  declined {declined}")
print("rejects:", json.dumps(rej, indent=1))
