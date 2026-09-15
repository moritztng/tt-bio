#!/usr/bin/env python3
"""Control for the one behaviour change in batched_matmul: which exceptions it still absorbs.

The branch narrowed `batched_matmul`'s `except` from bare `Exception` to "an L1 refusal or a DRAM
OOM, anything else re-raises". Every other claim on this row was checked against a device; this one
was only read, and it is the only change that can turn a fold that used to finish into a crash. The
config factory budgets against `_matmul_cb_budget()` before it hands a config back, so the branch
almost never fires in a real fold (0 refusals in 166278 and 442226 served RF3 calls), which is
exactly why it needs forcing rather than waiting for.

Both directions, on real operands, with the result checked:

  absorb  an oversized config whose circular buffers do not fit L1. Must be absorbed, must fall
          back to ttnn's own planner, and the fallback must equal the untuned reference -- an
          absorbed refusal that returns a WRONG answer is worse than a crash.
  raise   a structurally illegal config (out_subblock_h * out_subblock_w past the dest register
          file). Must propagate. Under the old bare `except` this read as "ttnn was slower today"
          with nothing in the log.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 python3 bmm_absorb_ctl.py --out out/bmm_ctl.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    import tt_bio.tenstorrent as TT
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"

    dev = TT.get_device()
    g = dev.compute_with_storage_grid_size()
    rec = {"doc": __doc__, "host": socket.gethostname(), "grid": [g.x, g.y],
           "arch": str(dev.arch()), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "l1_budget": TT._matmul_cb_budget(), "arms": {}}

    # Both-sides-batched, DRAM-interleaved, equal dtype: the gate batched_matmul itself checks.
    B, M, K, N = 4, 1024, 1024, 1024
    ta = torch.randn(1, B, M, K, dtype=torch.bfloat16)
    tb = torch.randn(1, B, K, N, dtype=torch.bfloat16)
    mk = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    a, b = mk(ta), mk(tb)
    rec["shape"] = {"a": list(a.shape), "b": list(b.shape), "dtype": str(a.dtype)}

    # The reference: ttnn's own planner, which is exactly what the absorbing path falls back to.
    ref = ttnn.to_torch(ttnn.matmul(a, b))

    tiles = lambda n: -(-n // 32)
    grid = tuple(TT.COMPUTE_GRID_MAIN)

    def cfg(sub_h, sub_w, in0_bw, pcm):
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid, in0_block_w=in0_bw,
            out_subblock_h=sub_h, out_subblock_w=sub_w,
            per_core_M=pcm, per_core_N=tiles(N))

    def run_arm(name, bad_cfg, expect):
        """Force `bad_cfg` through batched_matmul's tuned slot and record what came back."""
        orig = TT._batched_matmul_config
        TT._batched_matmul_config = lambda *a_, **k_: bad_cfg
        TT._BMM_CFG_RUNG.clear()
        arm = {"expect": expect, "cfg": TT._bmm_cfg_fields(bad_cfg)}
        t = time.perf_counter()
        try:
            out = TT.batched_matmul(a, b)
            arm["outcome"] = "absorbed"
            got = ttnn.to_torch(out)
            arm["equals_reference"] = bool(torch.equal(got, ref))
            arm["max_abs_vs_reference"] = float((got.float() - ref.float()).abs().max())
        except Exception as e:                                              # noqa: BLE001
            arm["outcome"] = "raised"
            arm["error"] = str(e)[:400]
            arm["error_type"] = type(e).__name__
            arm["is_l1_text"] = "circular buffers" in str(e)
            arm["is_dram_oom_text"] = "Out of Memory" in str(e)
        finally:
            TT._batched_matmul_config = orig
        arm["wall_s"] = round(time.perf_counter() - t, 3)
        arm["pass"] = arm["outcome"] == expect and arm.get("equals_reference", True)
        rec["arms"][name] = arm
        print(f"[bmm-ctl] {name}: expect={expect} got={arm['outcome']} "
              f"equal={arm.get('equals_reference')} pass={arm['pass']}", flush=True)
        return arm["pass"]

    # in0 alone is double-buffered at in0_block_w x per_core_M tiles: 2*32*32*2048 B = 4.2 MB per
    # core against a 1.5 MB L1. Structurally legal otherwise, so the only thing wrong is the bytes.
    ok_absorb = run_arm("l1_overflow", cfg(2, 2, tiles(K), tiles(M)), "absorbed")
    # out_subblock_h * out_subblock_w = 16, past the dest register file. Not an L1 refusal.
    ok_raise = run_arm("illegal_subblock", cfg(4, 4, 2, tiles(M)), "raised")
    if rec["arms"]["illegal_subblock"]["outcome"] == "raised":
        bad = rec["arms"]["illegal_subblock"]
        # It must re-raise for the RIGHT reason: not matched by either absorbing predicate.
        ok_raise = ok_raise and not bad["is_l1_text"] and not bad["is_dram_oom_text"]
        rec["arms"]["illegal_subblock"]["pass"] = ok_raise

    rec["verdict"] = "PASS" if (ok_absorb and ok_raise) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1))
    print(f"BMMCTL {rec['verdict']}", flush=True)
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(2)
