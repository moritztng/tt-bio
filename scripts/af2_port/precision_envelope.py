"""Step 6A: is the device arm's per-design error inside the reference's own precision freedom?

`device_floor.py` adjudicates a tap by comparing the device delta against the same quantity measured
between two precisions of the reference implementation. This does the same adjudication on the four
`af2_easy` criteria of a whole design, which is the level a filter decision is actually taken at.

The reference arm at bfloat16 is the baseline both other arms are measured from: float32 is the same
implementation inside its own precision freedom, device is ours. A ratio at or below 1 says the port
is no further from the reference than the reference is from itself.

n is 5, so this reports per design and does not pool.

    PYTHONPATH=. python3 scripts/af2_port/precision_envelope.py \\
        --pop scripts/af2_port/parity_artifacts/designpop_pxd196 --out /tmp/envelope.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCALARS = ("plddt", "i_ptm", "i_pae")     # the three af2_easy confidence criteria
RMSD_KEY = "bound_unbound_rmsd"


def rows(path: Path, field: str = "id") -> dict:
    if not path.exists():
        return {}
    return {json.loads(l)[field]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True, help="a designpop_* artifact directory")
    ap.add_argument("--bf16", default="scores_host.jsonl")
    ap.add_argument("--fp32", default="scores_reference_fp32_complex.jsonl")
    ap.add_argument("--device", default="scores_device.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A = Path(a.pop)

    bf16, fp32, dev = rows(A / a.bf16), rows(A / a.fp32), rows(A / a.device)
    r_bf16, r_fp32, r_dev = (rows(A / f) for f in
                             ("rmsd_reference.jsonl", "rmsd_reference_fp32.jsonl",
                              "rmsd_device.jsonl"))

    out = {"population": A.name, "designs": {}}
    for rid in sorted(fp32):
        if rid not in bf16 or rid not in dev:
            continue
        d = {}
        for k in SCALARS:
            base = bf16[rid]["ref"][k]
            df, dd = abs(fp32[rid]["ref"][k] - base), abs(dev[rid]["ref"][k] - base)
            d[k] = {"bf16": base, "fp32": fp32[rid]["ref"][k], "device": dev[rid]["ref"][k],
                    "fp32_delta": round(df, 6), "device_delta": round(dd, 6),
                    "ratio": (round(dd / df, 3) if df else None)}
        if rid in r_bf16 and r_bf16[rid].get(RMSD_KEY) is not None:
            base = r_bf16[rid][RMSD_KEY]
            e = {"bf16": base}
            for name, src in (("fp32", r_fp32), ("device", r_dev)):
                if rid in src and src[rid].get(RMSD_KEY) is not None:
                    e[name] = src[rid][RMSD_KEY]
                    e[name + "_delta"] = round(abs(src[rid][RMSD_KEY] - base), 6)
            if "fp32_delta" in e and "device_delta" in e:
                e["ratio"] = (round(e["device_delta"] / e["fp32_delta"], 3)
                              if e["fp32_delta"] else None)
            d[RMSD_KEY] = e
        out["designs"][rid] = d

    ratios = [v["ratio"] for d in out["designs"].values() for v in d.values()
              if v.get("ratio") is not None]
    out["n_designs"] = len(out["designs"])
    out["ratio_max"] = max(ratios) if ratios else None
    out["ratio_median"] = sorted(ratios)[len(ratios) // 2] if ratios else None
    out["inside_reference_freedom"] = all(r <= 1.0 for r in ratios) if ratios else None
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("population", "n_designs", "ratio_max", "ratio_median",
                       "inside_reference_freedom")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
