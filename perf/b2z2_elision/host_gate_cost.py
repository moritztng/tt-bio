#!/usr/bin/env python3
"""What the shift gather's host-side matrix check costs, per fold, at the production shape.

The elision is safe because it reads the selection matrix instead of its shape, and reading it
is host work the matmul path does not do: one reconstruction of the matrix a centred window
would produce, one `torch.equal`. That cost is inside the fold wall the cell publishes, but the
page also publishes a host/device split, so it is worth a number rather than an assurance.

Once per fold, in `_populate_diffusion_cache`. No device.
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from tt_bio.boltz2 import get_indexing_matrix  # noqa: E402
from tt_bio.tenstorrent import ATOM_WINDOW, ATOM_DIM, _atom_gather_shift_windows  # noqa: E402

PIECES = ATOM_DIM // ATOM_WINDOW
OUT = {"doc": __doc__, "shapes": []}
for k_real, k_bucket in ((129, 140), (129, 140), (140, 140)):
    ki = get_indexing_matrix(k_real, ATOM_WINDOW, ATOM_DIM, "cpu")
    ki = torch.nn.functional.pad(
        ki, (0, PIECES * 2 * k_bucket - ki.shape[1], 0, 2 * k_bucket - ki.shape[0]))
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        got = _atom_gather_shift_windows(ki)
        ts.append((time.perf_counter() - t0) * 1e3)
    assert got == k_real, (got, k_real)
    OUT["shapes"].append({"windows_real": k_real, "windows_bucket": k_bucket,
                          "matrix": list(ki.shape), "n": len(ts),
                          "median_ms": round(st.median(ts), 4),
                          "max_ms": round(max(ts), 4)})
    print(f"windows {k_real}/{k_bucket} matrix {tuple(ki.shape)} "
          f"median {st.median(ts):.4f} ms max {max(ts):.4f} ms", flush=True)

Path(sys.argv[1]).write_text(json.dumps(OUT, indent=1))
