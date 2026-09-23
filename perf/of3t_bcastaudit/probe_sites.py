#!/usr/bin/env python3
"""Re-probe every mixed-dtype form the audit saw, at the exact shapes and operand order a fold used.

    probe_sites.py AUDIT.json --out F.json [--reps R]

The envelope says which (op, dtype, axis) classes fail on a few shapes. This closes the gap between
"the class is clean" and "the call this fold makes is clean": for every distinct
(op, dtype_a, dtype_b, shape_a, shape_b) in AUDIT.json's mixed_sites it runs the op on a 0/1 second
operand after poisoning device memory, against a float64 reference, and compares every rep with rep 0.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

SITE = re.compile(r"^(\w+):(mixed_\w+) (\S+) (\w+)x(\w+) (\[[^\]]*\])x(\[[^\]]*\])$")


def main() -> int:
    argv = sys.argv[1:]
    audit = json.loads(Path(argv[0]).read_text())
    out = Path(argv[argv.index("--out") + 1])
    reps = int(argv[argv.index("--reps") + 1]) if "--reps" in argv else 8
    import ttnn
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    up = lambda t, d: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=d)  # noqa
    forms = {}
    for model, row in audit.items():
        for s in row["mixed_sites"]:
            m = SITE.match(s)
            if m:
                op, _, site, da, db, sa, sb = m.groups()
                forms.setdefault((op, da, db, sa, sb), []).append(f"{model} {site}")
    torch_op = {"multiply": torch.mul, "add": torch.add, "subtract": torch.sub}
    g = torch.Generator().manual_seed(0)
    res = {}
    for (op, da, db, sa, sb), where in sorted(forms.items()):
        base = {"mul": "multiply", "sub": "subtract"}.get(op.rstrip("_"), op.rstrip("_"))
        a = torch.randn(*json.loads(sa), generator=g)
        b = (torch.rand(*json.loads(sb), generator=g) > 0.2).float()
        adev = up(a, getattr(ttnn, da.lower()))
        ref = torch_op[base](ttnn.to_torch(adev).double(), b.double())
        counts, y0 = [], None
        for _ in range(reps):
            for t in [up(torch.full((1, 64, 64, 128), 3.0e4), ttnn.float32) for _ in range(8)]:
                ttnn.deallocate(t)
            bdev = up(b, getattr(ttnn, db.lower()))
            y = getattr(ttnn, base)(adev, bdev)
            yh = ttnn.to_torch(y).double()
            ulp = 2.0 ** -7 if y.dtype == ttnn.bfloat16 else 2.0 ** -14
            y0 = yh if y0 is None else y0
            counts.append([int(((yh - ref).abs() > ref.abs() * ulp + 1e-6).sum()),
                           int((yh != y0).sum())])
            ttnn.deallocate(y)
            ttnn.deallocate(bdev)
        key = f"{base} {da}x{db} {sa}x{sb}"
        res[key] = {"sites": where, "per_rep_wrong_unlike_rep0": counts,
                    "clean": all(max(c) == 0 for c in counts)}
        print(key, "clean" if res[key]["clean"] else "FAIL", counts, where, flush=True)
    out.write_text(json.dumps({"reps": reps, "forms": res}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
