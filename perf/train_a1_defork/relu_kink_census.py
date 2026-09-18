#!/usr/bin/env python3
"""Why the relu arm of gradcheck_dispatch misses the 1.0e-2 bar: it is the kink, measured.

relu's gradient is discontinuous, so the mask the backward gates on is a SIGN TEST on the
forward. Any forward error at all flips the sign of the coordinates that sit inside it, and
each flipped coordinate contributes its whole gradient rather than a rounded one. So the
gradient error of a kinked op scales as sqrt(flipped fraction), not as the forward error --
and REL_L2_BAR = 1.0e-2 was derived from the bf16 mantissa for a SMOOTH op.

This counts the flips instead of arguing about them, at three precisions, and puts the
count next to the measured error.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_s = importlib.util.spec_from_file_location("_gc", REPO / "perf" / "hallgrad" / "gradcheck.py")
GC = importlib.util.module_from_spec(_s); _s.loader.exec_module(GC)

import ttnn
from tt_bio import tenstorrent as tt
from tt_bio import ops

ARMS = [("bfloat16", "HiFi2"), ("float32", "HiFi2"), ("float32", "HiFi4")]
rows = []
tt.get_device()
for dts, fid in ARMS:
    dt = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[dts]
    tdt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dts]
    rng = np.random.default_rng(7)
    t = GC.case_linear(rng)
    rounded = {k: torch.from_numpy(v).to(tdt).to(torch.float64) for k, v in t.items()}
    ref_pre = rounded["x"] @ rounded["w"] + rounded["b"]
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
        fp32_dest_acc_en=(dts == "float32"), packer_l1_acc=True)
    dev = {k: ttnn.from_torch(v.to(tdt), dtype=dt, layout=ttnn.TILE_LAYOUT,
                              device=tt.get_device()) for k, v in rounded.items()}
    got_pre = ttnn.to_torch(ops.shipped_linear(dev["x"], dev["w"], dev["b"],
                                               compute_kernel_config=ckc,
                                               core_grid=tt.CORE_GRID_MAIN)).to(torch.float64)
    n = ref_pre.numel()
    flips = int(((got_pre > 0) != (ref_pre > 0)).sum())
    live = int((ref_pre > 0).sum())
    fwd_rel = float(torch.linalg.vector_norm(got_pre - ref_pre) / torch.linalg.vector_norm(ref_pre))
    rows.append({"dtype": dts, "fidelity": fid, "n": n, "live_mask": live,
                 "sign_flips": flips, "flip_frac_of_live": round(flips / live, 6),
                 "forward_rel_l2": fwd_rel,
                 "predicted_grad_rel_l2": round(float(np.sqrt(flips / live)), 5)})
print(json.dumps(rows, indent=1))
Path(REPO / "perf/train_a1_defork/out/relu_kink_census.json").write_text(json.dumps(rows, indent=1))
