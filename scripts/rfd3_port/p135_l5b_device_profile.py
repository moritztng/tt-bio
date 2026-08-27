#!/usr/bin/env python3
"""p135 -- the L5b device profile: where §12.4's 2.6x cost-model gap went.

§12.4 priced L5b at 22.1 ms/step of deletable DRAM traffic, applied the lineage's 75 % exchange
rate for a predicted -3.32 s/design, and the fold paid -1.255 to -1.322 s: **6.3 ms/step against
a predicted 16.6**. The candidate mechanism written down there -- the fused kernel runs the PV
MAC inside the normalise pass, so it adds arithmetic to a kernel whose cost the deleted write was
overlapping (`fusion-into-compute-bound-kernel-unhides-arithmetic`) -- was explicitly NOT a
finding, because nothing measured it. This measures it.

Six arms per shape, each amortised over N calls with ONE `synchronize_device`, because a
per-call sync on a 5 ms kernel inflates it about 2x
(`tt-bio-isolated-op-timing-oversync-inflates-cost`). Every big buffer is preallocated and
reused, so the allocator is not inside arms A, B, C or D; `ttnn.empty` is its own arm instead.

  A  softmax_into -> bf16 out       the shipped normalise, writing 294.3 MB at the R4 shape
  B  softmax_into -> fp32 out       the same normalise and the same compute, writing 588.6 MB.
                                    B-A is the MEASURED marginal price of 294.3 MB of writes on
                                    this card at this access pattern -- the quantity the cost
                                    model assumed it was deleting at the DRAM roof
  C  softmax_into(vv=...)           the fused L5b program: no big write, plus the PV MAC
  D  attn_value_matmul              the PV matmul the fold deletes, reading 294.3 MB back
  E  typecast fp32 -> bf16          the roof probe, 882.9 MB moved by an op that does nothing
                                    else (`roofline-roof-must-be-measured-not-asserted`)
  F  ttnn.empty + deallocate        the allocation p122 wraps, so the chain total is complete
                                    and E's in-loop allocator share is subtractable

Pre-registered readings, so the artifact is not interpreted after the fact:

  1. **isolated prize** = (A + D + F) - C, x9 calls/step. If it lands on the fold's 6.3 ms/step,
     the isolated screen agrees with the fold and the whole 2.6x is in the COST MODEL. If it is
     much larger, the fold does not collect what the screen shows and the gap is realisation
     (`rfd3-isolated-screen-underprices-residency-lever`, inverted).
  2. **the write's real price** = B - A. If it is near zero, the 294.3 MB write was already
     overlapped by the normalise, deleting it was never worth 2.5 ms, and that is the cost
     model's root error rather than the kernel's fault.
  3. **the MAC's added cost** = C - (A - (B - A)): what the fused program costs above a
     write-free normalise. This assumes the write's cost is linear in bytes, which B-A measures
     at one point only. Stated here rather than hidden in the arithmetic.

Both atom shapes, always. §12.4's 6.3 ms/step was measured at R3 (4576-wide atom axis) and the
census fixture is R4 (6080-wide), so a profile at R4 alone cannot be compared with the fold it
exists to explain.

    ~/.coworker/scripts/benchlock.sh rfd3-fusion-programme-p6 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:rfd3-fusion-programme-p6 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio/env/bin/python3 -u scripts/rfd3_port/p135_l5b_device_profile.py \
      perf/p135/l5b_device_profile.json
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio import softmax_generic as SG                                 # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p135/l5b_device_profile.json")
N = int(os.environ.get("P135_N", 10))          # calls per timed loop, amortising the one sync
REPS = int(os.environ.get("P135_REPS", 5))     # timed loops per arm; the spread is the noise floor
CALLS_PER_STEP = 9                             # 6 decoder + 3 atom encoder, confirmed by perf/p49x
STEPS_PER_DESIGN = 199

# (label, rows, key width). Rows are the design's atom count, the key width is the padded atom
# axis the fold's own census reports as served: 4576 at R3 (perf/p126/ab_R3_rerun.json), 6080 at
# R4 (perf/p123). Both are read off runs, not scaled from each other (§12.1).
SHAPES = [("R3", 4558, 4576), ("R4", 6051, 6080)]
if os.environ.get("P135_SMOKE"):
    SHAPES = [("smoke", 320, 3520)]
HEADS, HEAD_DIM = 4, 32
MB = 1024.0 * 1024.0


def med_ms(fn, dev, n=None, reps=None):
    """Median ms per CALL, amortising one device sync over `n` calls.

    Not one sync per call: at 5 ms/call the sync itself is a measurable fraction of the arm, and
    it lands on every arm differently because they enqueue different numbers of programs.
    """
    n = n or N
    reps = reps or REPS
    fn()                                        # compile + program-cache warm, never timed
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        ttnn.synchronize_device(dev)
        ts.append(1000.0 * (time.perf_counter() - t0) / n)
    return {"ms": round(statistics.median(ts), 4), "min": round(min(ts), 4),
            "max": round(max(ts), 4),
            "spread": round((max(ts) - min(ts)) / statistics.median(ts), 4)}


def profile(dev, label, rows, key_w):
    ckc_obj = M._default_compute_kernel_config()
    ckc_tuple = (ttnn.MathFidelity.HiFi4, True, True, False)   # what softmax_pv_fused passes
    torch.manual_seed(42)
    x_t = torch.randn(1, HEADS, rows, key_w, dtype=torch.float32) * 4.0
    v_t = torch.randn(1, HEADS, key_w, HEAD_DIM, dtype=torch.bfloat16)
    x = ttnn.from_torch(x_t, ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    vv = ttnn.from_torch(v_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)

    cls = SG.pv_classify(x, vv, ttnn.bfloat16, ckc_obj)
    print("[%s] rows=%d key=%d  pv_classify: %s" % (label, rows, key_w, cls), flush=True)
    if not cls.get("ok"):
        # Timing a fused program at a shape production declines at would be a number about
        # nothing. Report the refusal and move on rather than print a delta.
        return {"shape": [1, HEADS, rows, key_w, HEAD_DIM], "label": label,
                "pv_classify": {k: str(v) for k, v in cls.items()},
                "skipped": "pv_classify declined at this shape"}

    full = [1, HEADS, rows, key_w]
    out_bf16 = ttnn.empty(full, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, x.memory_config())
    out_fp32 = ttnn.empty(full, ttnn.float32, ttnn.TILE_LAYOUT, dev, x.memory_config())
    out_pv = ttnn.empty([1, HEADS, rows, HEAD_DIM], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                        x.memory_config())

    arms = {}
    arms["A_softmax_write_bf16"] = med_ms(
        lambda: SG.softmax_into(dev, x, out_bf16, ckc=ckc_tuple), dev)
    arms["B_softmax_write_fp32"] = med_ms(
        lambda: SG.softmax_into(dev, x, out_fp32, ckc=ckc_tuple), dev)
    arms["C_fused_softmax_pv"] = med_ms(
        lambda: SG.softmax_into(dev, x, out_pv, ckc=ckc_tuple, vv=vv), dev)
    arms["D_attn_value_matmul"] = med_ms(
        lambda: _pv_into(out_bf16, vv, ckc_obj), dev)

    def _roof():
        t = ttnn.typecast(x, ttnn.bfloat16)
        ttnn.deallocate(t)
    arms["E_roof_typecast"] = med_ms(_roof, dev)

    def _alloc():
        t = ttnn.empty(full, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, x.memory_config())
        ttnn.deallocate(t)
    arms["F_ttnn_empty"] = med_ms(_alloc, dev)

    # Bytes each arm moves, counted here rather than asserted, so an achieved GB/s is derivable
    # for every row and the roof is one of them.
    elems = HEADS * rows * key_w
    bf16_mb = 2.0 * elems / MB
    fp32_mb = 4.0 * elems / MB
    v_mb = 2.0 * HEADS * key_w * HEAD_DIM / MB
    pv_out_mb = 2.0 * HEADS * rows * HEAD_DIM / MB
    traffic = {"A_softmax_write_bf16": fp32_mb + bf16_mb,
               "B_softmax_write_fp32": fp32_mb + fp32_mb,
               "C_fused_softmax_pv": fp32_mb + v_mb + pv_out_mb,
               "D_attn_value_matmul": bf16_mb + v_mb + pv_out_mb,
               "E_roof_typecast": fp32_mb + bf16_mb,
               "F_ttnn_empty": 0.0}
    for k, a in arms.items():
        a["mb_moved"] = round(traffic[k], 1)
        a["achieved_gb_s"] = (round(traffic[k] / 1024.0 / (a["ms"] / 1000.0), 1)
                              if a["ms"] > 0 and traffic[k] else None)

    A, B, C, D, F = (arms[k]["ms"] for k in
                     ("A_softmax_write_bf16", "B_softmax_write_fp32", "C_fused_softmax_pv",
                      "D_attn_value_matmul", "F_ttnn_empty"))
    write_price = B - A                       # reading 2, above
    macfree = A - write_price                 # a normalise that wrote nothing, if writes are linear
    shipped_chain = A + D + F
    prize_call = shipped_chain - C
    rec = {"label": label, "shape": [1, HEADS, rows, key_w, HEAD_DIM],
           "pv_classify_ok": True, "n_per_loop": N, "reps": REPS, "arms": arms,
           "shipped_chain_ms_per_call": round(shipped_chain, 4),
           "fused_ms_per_call": round(C, 4),
           "isolated_prize_ms_per_call": round(prize_call, 4),
           "isolated_prize_ms_per_step": round(prize_call * CALLS_PER_STEP, 3),
           "isolated_prize_s_per_design": round(prize_call * CALLS_PER_STEP
                                                * STEPS_PER_DESIGN / 1000.0, 3),
           "write_price_ms": round(write_price, 4),
           "write_price_mb": round(bf16_mb, 1),
           "write_price_gb_s": (round(bf16_mb / 1024.0 / (write_price / 1000.0), 1)
                                if write_price > 0 else None),
           "roof_gb_s": arms["E_roof_typecast"]["achieved_gb_s"],
           "implied_macfree_normalise_ms": round(macfree, 4),
           "mac_added_cost_ms": round(C - macfree, 4),
           "mac_added_cost_note": ("C - (A - (B-A)); assumes the write's price is linear in "
                                   "bytes, which B-A measures at one point only")}
    print("[%s] shipped chain %.3f ms/call (A %.3f + D %.3f + F %.3f), fused %.3f  ->  "
          "isolated prize %.3f ms/call = %.2f ms/step = %.3f s/design"
          % (label, shipped_chain, A, D, F, C, prize_call,
             rec["isolated_prize_ms_per_step"], rec["isolated_prize_s_per_design"]), flush=True)
    print("[%s] the write's measured price for %.1f MB: %.3f ms (%s GB/s) against a roof of "
          "%s GB/s" % (label, bf16_mb, write_price, rec["write_price_gb_s"], rec["roof_gb_s"]),
          flush=True)
    print("[%s] MAC added over a write-free normalise: %.3f ms/call" % (label, C - macfree),
          flush=True)

    for t in (x, vv, out_bf16, out_fp32, out_pv):
        ttnn.deallocate(t)
    return rec


def _pv_into(attn, vv, ckc_obj):
    r = M.attn_value_matmul(attn, vv, ckc_obj, ttnn.bfloat16)
    ttnn.deallocate(r)


def main():
    SG.set_pv_enabled(True)      # script-local: pv_classify reads it, the model is not running
    dev = M.get_device()
    g = dev.compute_with_storage_grid_size()
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES", "?"),
           "grid": [g.x, g.y], "torch": torch.__version__,
           "calls_per_step": CALLS_PER_STEP, "steps_per_design": STEPS_PER_DESIGN,
           "fold_measured_ms_per_step_R3": 6.3,      # §12.4, the number this run has to explain
           "cost_model_predicted_ms_per_step_R3": 16.6,
           "shapes": []}
    for label, rows, key_w in SHAPES:
        res["shapes"].append(profile(dev, label, rows, key_w))

    r3 = next((s for s in res["shapes"] if s.get("label") == "R3" and not s.get("skipped")), None)
    if r3:
        got = r3["isolated_prize_ms_per_step"]
        res["verdict"] = (
            "isolated screen at R3 pays %.2f ms/step against the fold's 6.3 and the cost model's "
            "16.6 -- %s" % (got, "the gap is in the COST MODEL, the screen and the fold agree"
                            if abs(got - 6.3) <= 2.0 else
                            "the screen and the fold DISAGREE, so the gap is realisation"))
        print("\nVERDICT: " + res["verdict"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
