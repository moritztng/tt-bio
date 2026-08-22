#!/usr/bin/env python3
"""Why does the fused SDPA decline OpenFold3 at some token counts and serve it at others?

`_tri_att_sdpa_hifi` counts a decline but not a reason, and `triatt_sdpa._reject` records only the
reason name -- `fill_preconditions` covers six separate terms. This wraps `SG.plan` and records
the plan dict next to the shape, so the census says which term failed rather than that one did.

    of3_gate_probe.py <fixture> <card>

Runs one fold at 5 sampling steps, which is legitimate here because nothing structural is scored:
the only outputs are the plan census and the served/declined counters, and both are settled in the
trunk before the sampler runs.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

FIX, CARD = sys.argv[1], sys.argv[2]
os.environ["TT_VISIBLE_DEVICES"] = CARD
os.environ["TT_BIO_LEASE_CARDS"] = CARD
os.environ["TT_BIO_LEASE_HOLDER"] = "worker:fused-sdpa-adopt-of3-p2"
os.environ["TT_BIO_TRIATT_FUSED_HIFI"] = "1"

import tt_bio.triatt_sdpa as SD                                    # noqa: E402

KEYS = ("nh_per_core", "q_per_core", "bcast_batch", "use_padded_mask", "NKH", "NVH",
        "k_num_chunks", "Sq_chunk_t", "Sk_chunk_t")
LOG: dict = {}
_orig_plan = SD.SG.plan


def plan(q, k, v, bias, out, q_chunk, k_chunk, grid, ckc, scale, split):
    p = _orig_plan(q, k, v, bias, out, q_chunk, k_chunk, grid, ckc, scale, split)
    key = f"shape={[int(d) for d in q.shape]} q_chunk={q_chunk} k_chunk={k_chunk} " \
          f"grid={tuple(grid)} split={tuple(split)}"
    if key not in LOG:
        H = int(q.shape[1])
        d = {k2: (int(p[k2]) if not isinstance(p[k2], bool) else p[k2]) for k2 in KEYS if k2 in p}
        d["FAILS"] = [t for t, okv in (
            ("nh_per_core==1", p["nh_per_core"] == 1),
            ("q_per_core==1", p["q_per_core"] == 1),
            ("bcast_batch", bool(p["bcast_batch"])),
            ("not use_padded_mask", not p["use_padded_mask"]),
            ("NKH==H", p["NKH"] == H),
            ("NVH==H", p["NVH"] == H)) if not okv]
        LOG[key] = d
    return p


SD.SG.plan = plan

sys.argv = ["fold_fix_ab.py", "--model", "openfold3", "--fix", FIX, "--label", "probe",
            "--seeds", "0", "--sampling-steps", "5", "--dump-distogram",
            "--outdir", f"/tmp/gateprobe/{FIX}"]
import runpy                                                        # noqa: E402
try:
    runpy.run_path(str(ROOT / "perf" / "rf3" / "fold_fix_ab.py"), run_name="__main__")
finally:
    import tt_bio.tenstorrent as T
    out = {"fixture": FIX,
           "hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS),
           "rejects": {f"{r}|{list(s)}": n for (r, s), n in SD.REJECTS.items()},
           "plans": LOG}
    Path(f"/tmp/gateprobe_{FIX}.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n===== GATE CENSUS", FIX, "=====")
    print(json.dumps(out, indent=1))
