#!/usr/bin/env python3
"""Cross-key parity for the E6 gated move: the re-keyed kernel against the one main ships.

The control arm is not a reimplementation. `origin/main:tt_bio/reblock_permute.py` is exec'd into
its own namespace with `__file__` pointed at a checkout of main's kernel sources, so the old key
runs its own reader, its own writer and its own program cache inside this process. Both arms then
move the SAME draw and the two results are torch.equal'd.

N=298 is in the list on purpose: D1 is not a multiple of 32 there, so the last row tile is partly
padding. The re-key moved the writer's staging zero from once per (it, jt) to once per
(it, jt, ct), which is exactly the path that is invisible at 512.

Cw = 4*C is the trimul's fused projection layout and the slices are the ones both production
callers pass (chunk(xw, 4, -1)[2] gated by [0]).
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RP


def old_module():
    """main's reblock_permute, with main's kernel sources, as a live module namespace."""
    d = Path(tempfile.mkdtemp(prefix="e6oldkey."))
    tar = subprocess.run(
        ["git", "archive", "origin/main", "tt_bio/kernels/reblock_permute",
         "tt_bio/kernels/reblock_permute_gated"], cwd=ROOT, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(d)], input=tar, check=True)
    src = subprocess.run(["git", "show", "origin/main:tt_bio/reblock_permute.py"], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout
    ns = {"__file__": str(d / "tt_bio" / "reblock_permute.py"), "__name__": "_rp_oldkey"}
    exec(compile(src, "origin/main:tt_bio/reblock_permute.py", "exec"), ns)
    return ns


CASES = [(128, 256), (256, 256), (298, 256), (320, 256), (512, 256),
         (384, 128), (256, 384)]

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    m = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if m:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = m
dev = T.get_device()
g = dev.compute_with_storage_grid_size()

OLD = old_module()
import importlib.metadata as im
res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "grid": [g.x, g.y], "ttnn": im.version("ttnn"), "cases": {}}

ok = True
for N, C in CASES:
    torch.manual_seed(N * 1000 + C)
    t = torch.randn(1, N, N, 4 * C)
    xw = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    new = ttnn.to_torch(RP.reblock_permute_gated(xw, 2 * C, 0, C))
    old = ttnn.to_torch(OLD["reblock_permute_gated"](xw, 2 * C, 0, C))
    eq = bool(torch.equal(old, new))
    # the two-op sequence the kernel replaces, as an absolute anchor rather than a relative one
    p, gsl = t[..., 2 * C:3 * C], t[..., 0:C]
    ref = torch.permute(
        (p.to(torch.bfloat16).float() * torch.sigmoid(gsl.to(torch.bfloat16).float())
         ).to(torch.bfloat16), (0, 3, 1, 2))
    md = float((new.float() - ref.float()).abs().max())
    res["cases"][f"N{N}_C{C}"] = {"torch_equal_vs_main_key": eq, "max_abs_vs_torch_ref": md,
                                  "shape": list(new.shape)}
    ok &= eq
    print("  N=%4d C=%3d  torch.equal vs main key: %s   max|new-torchref|=%.4g"
          % (N, C, eq, md), flush=True)
    ttnn.deallocate(xw)

res["all_bit_exact"] = ok
a.out.write_text(json.dumps(res, indent=1))
print("ALL BIT-EXACT" if ok else "FAILURES", flush=True)
sys.exit(0 if ok else 1)
