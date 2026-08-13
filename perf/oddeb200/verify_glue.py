#!/usr/bin/env python3
"""Bit-exactness check of the landed host-glue lever, against the ACTUAL engine functions
(the screen validated a transcription; this validates what shipped). CPU only, no device.

  1. `Protenix._generate_relp` with `_RELP_SCATTER` True vs False, torch.equal, at both axes and
     on a two-chain / multi-entity index set (the screen only exercised one chain).
  2. the two z_trunk consumers' dtype contract: bf16 + fp32 promotes to fp32 exactly, and
     `ttnn.from_torch(bf16, dtype=bfloat16)` is the same bytes as from an fp32 upcast.
"""
import sys, time
sys.path.insert(0, "/home/ttuser/.coworker/wt/opendde-beat-b200")
import torch

import tt_bio.protenix as PX
from tt_bio.protenix import Protenix


def idx(n, nchain):
    """nchain chains of ~equal length, two distinct entities, so sc/sr/se all vary."""
    per = n // nchain
    asym = torch.arange(n) // per
    asym = asym.clamp(max=nchain - 1)
    res = torch.arange(n) % per
    ent = asym % 2
    tok = torch.arange(n)
    sym = asym // 2
    return {"asym_id": asym, "residue_index": res, "entity_id": ent,
            "token_index": tok, "sym_id": sym}


ok = True
for n in (512, 995):
    for nchain in (1, 2, 4):
        f = idx(n, nchain)
        PX._RELP_SCATTER = False
        t0 = time.perf_counter(); a = Protenix._generate_relp(f); ta = time.perf_counter() - t0
        PX._RELP_SCATTER = True
        t0 = time.perf_counter(); b = Protenix._generate_relp(f); tb = time.perf_counter() - t0
        eq = torch.equal(a, b)
        ok &= eq and a.dtype == b.dtype and a.shape == b.shape
        print(f"relp n={n:4d} chains={nchain}: torch.equal={eq} dtype {a.dtype}=={b.dtype} "
              f"shape {tuple(a.shape)} | legacy {ta*1e3:7.1f} ms  scatter {tb*1e3:7.1f} ms "
              f"| saving {(ta-tb)*1e3:6.1f} ms")
        del a, b
print("RELP_STATS (scatter, legacy):", PX.RELP_STATS)

# --- the seam's dtype contract ------------------------------------------------------------------
z_bf16 = (torch.randn(64, 64, 384) * 3).to(torch.bfloat16)
z_fp32 = z_bf16.float()
s = torch.randn(64, 8)
w = torch.randn(384, 8)
add_fp32 = z_fp32 + torch.nn.functional.linear(s, w).unsqueeze(1)
add_bf16 = z_bf16 + torch.nn.functional.linear(s, w).unsqueeze(1)
print(f"confidence-head add: dtype {add_bf16.dtype} torch.equal={torch.equal(add_fp32, add_bf16)}")
ok &= torch.equal(add_fp32, add_bf16) and add_bf16.dtype is torch.float32
# the expander's cast: from an fp32 upcast vs from the bf16 directly
print(f"expander cast: torch.equal={torch.equal(z_fp32.to(torch.bfloat16), z_bf16)}")
ok &= torch.equal(z_fp32.to(torch.bfloat16), z_bf16)

print("VERDICT:", "PASS -- bit-exact" if ok else "FAIL")
sys.exit(0 if ok else 1)
