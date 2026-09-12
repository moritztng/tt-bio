#!/usr/bin/env python3
"""Is `outbuf=L1` bit-exact against the shipped DRAM output, on every instance it wins on?

The whole value of this lever is that it is pure placement: the same kernel, the same accumulation
order, a different destination buffer. If that is true the outputs are identical bit for bit, and
the lever needs no accuracy argument at all. If it is not true, it is a numerics change and joins
the fidelity arms behind the cdk2x2_298 control.

Timing is deliberately absent here. This compares outputs only, on the same seeded operands.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import op_replay as R                                                     # noqa: E402
from sweep import run_one                                                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "bench_manifest.json"))
    ap.add_argument("--renoise", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ratio", type=float, default=1.03)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    R._init_tables(ttnn)

    recs = {r["id"]: r for r in json.load(open(a.manifest))}
    inst = json.load(open(a.renoise))["instances"]
    todo = []
    for i in inst:
        arm = next((x for x in i["arms"] if x["raw_knob"] == "outbuf=L1"), None)
        if arm and arm.get("ratio", 0) >= a.min_ratio:
            todo.append((i["id"], arm["ratio"]))

    device = R.open_device_retry(ttnn)
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "arch": str(device.arch())}, "rows": []}
    print(f"# checking {len(todo)} instances where outbuf=L1 wins", flush=True)
    for iid, ratio in todo:
        rec = recs[iid]
        row = {"id": iid, "ratio": ratio,
               "shipped_out_buffer": rec["out_mem"]["buffer"],
               "shapes": [",".join(str(d) for d in s["shape"]) for s in rec["inputs"]]}
        try:
            _, _, base = run_one(ttnn, torch, rec, device, "", 2, 1, True)
            _, _, arm = run_one(ttnn, torch, rec, device, "outbuf=L1", 2, 1, True)
            d = (arm.float() - base.float()).abs()
            row["max_abs"] = float(f"{float(d.max()):.3e}")
            row["bit_exact"] = bool(float(d.max()) == 0.0)
        except Exception as ex:                                           # noqa: BLE001
            row["error"] = f"{type(ex).__name__}: {str(ex)[:140]}"
        out["rows"].append(row)
        print(f"   {iid:22s} {ratio:6.3f}x  "
              f"{row.get('error') or ('BIT-EXACT' if row['bit_exact'] else 'max_abs ' + str(row['max_abs']))}",
              flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    ttnn.close_device(device)
    n = sum(1 for r in out["rows"] if r.get("bit_exact"))
    print(f"DONE {n}/{len(out['rows'])} bit-exact -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
