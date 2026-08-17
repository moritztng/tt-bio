#!/usr/bin/env python3
"""Does each of the four p3 levers fire on a Wormhole grid, and is it still bit-exact there?

The four levers (E, C-in, F, G) were measured only on Blackhole. All four ride inside
`SwiGLUFFN._row_blocked`, which is reached whenever `_split_plan` returns rows -- and on Wormhole
it does, because `SPLIT_SWIGLU_SMALL_GRID` is True in the tree JapanFold already serves. So they
are live on the Galaxy the moment the sync lands, and "Blackhole-only code" is not the answer.

What this script asks, per size, without folding anything:

  * does the lever SERVE or does the refusal ladder decline it (the [served, declined] censuses);
  * is the arm `torch.equal` to the all-off arm (a re-blocked matmul is NOT bit-exact for free --
    state/japanfold-esmfold2-wh-msa-cap-p2.md section 3 measured a 192.0 max diff from exactly
    that on this model);
  * does it raise, and with what.

Run it through `SwiGLUFFN.residual`, not `__call__`: F and G only exist on the residual path, and
`PairUpdateBlock` is the only caller.

Fast mode is forced on by default because `main.py` forces `--fast` for ESMFold2 on Wormhole, and
that decides the fc1/fc2 weight dtype (`bfloat8_b`), which decides both the L1 footprint and
whether `_pair_proj_config` declines. An arm measured in bf16 is not the arm the Galaxy runs.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import esmc as EC

C_Z, D_FF = 256, 1024

# Each arm names the levers that are ON. `off` is the pre-p3 code path, `ship` is the tree's own
# defaults. Every arm writes all four gates every round: an arm that leaves a gate where the
# previous arm left it is not the arm it is named after (the trap fold_ab.py documents).
ARMS = {
    "off":  dict(ln=False, slice=False, resid=False, fill=False),
    "E":    dict(ln=True,  slice=False, resid=False, fill=False),
    "Cin":  dict(ln=False, slice=True,  resid=False, fill=False),
    "F":    dict(ln=False, slice=False, resid=True,  fill=False),
    "G":    dict(ln=False, slice=False, resid=True,  fill=True),   # G rides on F
    "ship": dict(ln=True,  slice=True,  resid=True,  fill=True),
}


def ckc():
    """The compute-kernel-config class for the part this process opened; `BlackholeCompute...`
    throws on a wormhole_b0. Same dispatch tt-bio does at tenstorrent.py:5763."""
    cls = (ttnn.types.WormholeComputeKernelConfig if T.is_wormhole()
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def clear_refusals():
    """A refusal cached by one arm silently disables that lever for every later arm, which would
    read as 'the lever declined' when it was never asked. Reset between arms."""
    for name in ("_L1_LN_REFUSED", "_L1_SLICE_REFUSED", "_FUSED_RESID_REFUSED",
                 "_FILL_ASSEMBLY_REFUSED"):
        getattr(EC, name).clear()


def zero_stats():
    for name in ("L1_LN_STATS", "L1_SLICE_STATS", "FUSED_RESID_STATS", "FILL_ASSEMBLY_STATS",
                 "L1_FC1_STATS", "SPLIT_STATS"):
        s = getattr(EC, name)
        s[0] = s[1] = 0


def timed(fn, dev, reps=4, batches=5, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / reps)
    return st.median(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="320,512,640,768,1024")
    ap.add_argument("--arms", default="off,E,Cin,F,G,ship")
    ap.add_argument("--fast", type=int, default=1,
                    help="1 mirrors the --fast ESMFold2 is forced into on Wormhole (main.py)")
    ap.add_argument("--no-timing", action="store_true",
                    help="census + parity only; skip the ms/call loop")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    T.set_fast_mode(bool(a.fast))
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    torch.manual_seed(0)
    sd = {"0.weight": torch.randn(C_Z) * 0.05 + 1.0,
          "0.bias": torch.randn(C_Z) * 0.02,
          "1.weight": torch.randn(2 * D_FF, C_Z) * 0.02,
          "3.weight": torch.randn(C_Z, D_FF) * 0.02}
    ffn = EC.SwiGLUFFN(sd, ckc(), fuse_swiglu=True)

    lo, hi = EC.PAIR_FFN_ROW_BLOCK_SEQ
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "git_head": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
           "arch": "wormhole" if T.is_wormhole() else "blackhole",
           "grid": [g.x, g.y], "cores": g.x * g.y,
           "l1_unreserved_per_core": int(ttnn.get_max_worker_l1_unreserved_size()),
           "is_small_grid": bool(T._IS_SMALL_GRID), "fast_mode": bool(a.fast),
           "rows": EC._PAIR_FFN_ROW_BLOCK, "window": [lo, hi],
           # If `fuse_swiglu` resolved True the wheel has a fused kernel, `split_swiglu` is False
           # and NONE of the four levers can run on this machine. That is a legitimate verdict and
           # the first thing to read out of this file.
           "split_swiglu": bool(ffn.split_swiglu), "fuse_swiglu": bool(ffn.fuse_swiglu),
           "sizes": {}}
    print(json.dumps({k: v for k, v in res.items() if k != "sizes"}), flush=True)

    for L in [int(s) for s in a.sizes.split(",")]:
        xt = torch.randn(1, L, L, C_Z) * 0.5
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        del xt
        row = {"in_window": bool(lo <= L <= hi)}
        ref = None
        for arm in [s for s in a.arms.split(",") if s]:
            cfg = ARMS[arm]
            clear_refusals()
            EC.set_pair_ffn_l1_ln(cfg["ln"])
            EC.set_pair_ffn_l1_slice(cfg["slice"])
            EC.set_pair_ffn_fused_residual(cfg["resid"])
            EC.set_pair_ffn_fill_assembly(cfg["fill"])
            zero_stats()
            try:
                r = ffn.residual(x)
                got = ttnn.to_torch(r)
                ttnn.deallocate(r)
                row[arm] = {"ln": list(EC.L1_LN_STATS), "slice": list(EC.L1_SLICE_STATS),
                            "resid": list(EC.FUSED_RESID_STATS),
                            "fill": list(EC.FILL_ASSEMBLY_STATS),
                            "l1_fc1": list(EC.L1_FC1_STATS), "split": list(EC.SPLIT_STATS),
                            "refused": {"ln": len(EC._L1_LN_REFUSED),
                                        "slice": len(EC._L1_SLICE_REFUSED),
                                        "resid": len(EC._FUSED_RESID_REFUSED),
                                        "fill": len(EC._FILL_ASSEMBLY_REFUSED)}}
                if arm == "off":
                    ref = got
                elif ref is not None:
                    row[arm]["torch_equal"] = bool(torch.equal(ref, got))
                    row[arm]["max_abs_diff"] = float((ref - got).abs().max())
                del got
                if not a.no_timing:
                    zero_stats()
                    m, raw = timed(lambda: ttnn.deallocate(ffn.residual(x)), dev)
                    row[arm]["ms"] = round(m, 4)
                    row[arm]["ms_raw"] = [round(v, 4) for v in raw]
            except Exception as e:  # noqa: BLE001 -- an OOM at 1024 aa is a result, not a crash
                row[arm] = {"error": "%s: %s" % (type(e).__name__, e)[:600]}
            print(L, arm, json.dumps({k: v for k, v in row[arm].items() if k != "ms_raw"}),
                  flush=True)
        if ref is not None and "ship" in row and "ms" in row.get("off", {}) \
                and "ms" in row.get("ship", {}):
            row["ship_vs_off_ms"] = round(row["ship"]["ms"] - row["off"]["ms"], 4)
        res["sizes"][L] = row
        ttnn.deallocate(x)
        del ref
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))  # written per size: an OOM later keeps this

    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
