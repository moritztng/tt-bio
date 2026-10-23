"""Summarise a gap512 chain: per-leg split summary and byte identity of designs.pt against arm A.

    python compare.py <chain dir>
"""
import json
import sys
from pathlib import Path

import torch


def flat(x, prefix=""):
    if torch.is_tensor(x):
        yield prefix, x
    elif isinstance(x, dict):
        for k in sorted(x):
            yield from flat(x[k], f"{prefix}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from flat(v, f"{prefix}[{i}]")
    elif hasattr(x, "__dict__"):
        yield from flat(vars(x), prefix)


def main(root):
    root = Path(root)
    legs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "timeline.json").exists())
    ref = {}
    for leg in legs:
        s = json.loads((leg / "timeline.json").read_text())["summary"]
        st = s["stage_s_per_design"]
        ident = ""
        pt = leg / "designs.pt"
        if pt.exists():
            t = dict(flat(torch.load(pt, weights_only=False)))
            if not ref:
                ref = t
            ident = "identical" if ref.keys() == t.keys() and all(torch.equal(ref[k], t[k]) for k in ref) else "DIFFERS"
        print(f"{leg.name:6s} s/design {s['s_per_design']:7.3f}  gap {s['gap_s_per_design']:.3f} "
              f"({s['gap_ms_per_call_median']} ms/call)  wait {s['wait_s_per_design']:.3f}  "
              f"launch {st.get('launch', 0):.3f} enqueue {st.get('enqueue', 0):.3f} heads {st.get('heads', 0):.3f}  "
              f"AICLK {s['aiclk_mhz']} (min {s['aiclk_min_mhz']})  load {s['load1']}  {ident} vs {legs[0].name}")


if __name__ == "__main__":
    main(sys.argv[1])
