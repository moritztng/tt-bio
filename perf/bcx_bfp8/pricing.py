"""bcx-bfp8: what bfp8 can be worth on the CURRENT branch, priced against a stamped census.

`bcx-plan` prices bfp8 as a ROOF DISTANCE: "reaching the measured DRAM roof at bf16 is 8.73x;
reaching the same roof at bfp8, one byte per element, is 17.5x", and concludes that the 3x bar
"sits between two measured roofs, and bfp8 is the only one of the two that contains the target".

A roof distance is not what a conversion buys. This script computes what it buys, and the answer
does not depend on measuring bfp8 at all: it is bounded above by the host, which a dtype change
does not touch. bfp8 removes BYTES. It removes no ttnn calls, so it removes no enqueue work.

Inputs are `bcx-realcensus`'s n=256 numbers on the stack arm, qb1 card 0, AICLK 1343-1350 MHz
sampled inside every window -- properly stamped, which this row's own 800 MHz table is not.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- bcx-realcensus, quoted
DEV_MS = 210.6          # Evoformer block backward, device kernel time, 1163 ops
WALL_MS = 230.7         # same block, unprofiled wheel, K=1
HOST_MS = 178.0         # host hands control back after 0.178 s of the 0.231 s block
STEP_S = 13.377         # whole 4+48 checkpointed gradient step at n=256
STEP_BWD_S = 11.69
STEP_FWD_S = 1.681
STEP_HOST_S = 9.2       # 48 x 0.178 + 4 x 0.164

# Every class in that census is graded against a BYTE roof -- "Every backward matmul in this
# block is bytes-bound by that rule, so none sits against its FLOP roof" -- so the byte-bound
# share of device time is the whole of it, and bfp8's device ceiling is the byte ratio itself.
CLASSES = {"layout": 79.2, "matmul": 65.4, "eltwise": 54.8,
           "reduction": 7.3, "softmax": 2.9, "layernorm": 0.9, "other": 0.1}

NOMINAL = 2.0 / (1088.0 / 1024.0)   # bf16 bytes / bfp8_b bytes = 1.882
H200_STEP_S = 0.511
BAR_X = 3.0
BAR_S = BAR_X * H200_STEP_S


def block(s):
    """One Evoformer block backward if bfp8 gives the device a factor `s` on byte-bound work."""
    dev = DEV_MS / s
    naive = dev + (WALL_MS - DEV_MS)     # wall the device does not account for, unchanged
    return dev, naive, max(naive, HOST_MS)


def step(s):
    _, _, w = block(s)
    bwd = max(STEP_BWD_S / (WALL_MS / w), STEP_HOST_S)
    return STEP_FWD_S + bwd


def main():
    rows = []
    print(f"byte-bound share of device time: {sum(CLASSES.values()):.1f} of {DEV_MS} ms = "
          f"{sum(CLASSES.values())/DEV_MS*100:.0f} %  (every class graded against a byte roof)")
    print(f"nominal bfp8_b/bf16 byte ratio 0.5313 -> device-side ceiling {NOMINAL:.3f}x\n")
    print(f"{'device x':>9} {'block dev':>10} {'block wall':>11} {'block x':>8} "
          f"{'step s':>8} {'step x':>7} {'x off H200':>11} {'left to bar':>12}")
    for s in [1.0, 1.2, 1.4, 1.6, NOMINAL, 2.5, 4.0, 1e9]:
        dev, naive, w = block(s)
        st = step(s)
        rows.append({"device_x": s, "block_dev_ms": dev, "block_wall_ms": w,
                     "block_x": WALL_MS / w, "step_s": st, "step_x": STEP_S / st,
                     "x_off_h200": st / H200_STEP_S, "left_to_bar_x": st / BAR_S})
        tag = "  <- bfp8 nominal" if abs(s - NOMINAL) < 1e-9 else (
              "  <- free device" if s > 1e8 else "")
        print(f"{min(s,99999):9.3f} {dev:10.1f} {w:11.1f} {WALL_MS/w:8.3f} "
              f"{st:8.3f} {STEP_S/st:7.3f} {st/H200_STEP_S:11.1f} {st/BAR_S:12.2f}{tag}")

    dev, naive, w = block(NOMINAL)
    print(f"\nAt the nominal ratio the device half of the block falls {DEV_MS:.1f} -> {dev:.1f} ms,")
    print(f"which would be {WALL_MS/naive:.2f}x on wall if the host were free. The host is not free:")
    print(f"it is busy {HOST_MS:.1f} ms of the block's {WALL_MS:.1f} ms and a dtype change removes")
    print(f"no ttnn calls, so wall stops at {w:.1f} ms -- {WALL_MS/w:.2f}x, not {WALL_MS/naive:.2f}x.")
    st = step(NOMINAL)
    print(f"\nWhole 4+48 step at n=256: {STEP_S:.3f} s -> {st:.3f} s ({STEP_S/st:.2f}x).")
    print(f"Against one H200 at {H200_STEP_S} s: {STEP_S/H200_STEP_S:.1f}x today -> "
          f"{st/H200_STEP_S:.1f}x. The {BAR_X:.0f}x bar is {BAR_S:.3f} s, so bfp8 at its own")
    print(f"nominal ceiling leaves {st/BAR_S:.1f}x still to find.")
    free = step(1e9)
    print(f"\nEven an INFINITELY fast device leaves {free:.2f} s ({STEP_S/free:.2f}x, "
          f"{free/H200_STEP_S:.1f}x off one H200, {free/BAR_S:.1f}x over the bar):")
    print(f"that is the host-enqueue floor, and it is what bounds every device-side lever on this")
    print(f"branch, bfp8 included. Trace capture is the lever that moves it, and it has to land")
    print(f"BEFORE bfp8 is worth converting, not after.")

    R = roofs()
    (HERE / "pricing.json").write_text(json.dumps(
        {"source": "bcx-realcensus n=256 stack arm, qb1 card 0, AICLK 1343-1350 sampled during",
         "inputs": {"block_device_ms": DEV_MS, "block_wall_ms": WALL_MS, "block_host_ms": HOST_MS,
                    "step_s": STEP_S, "step_bwd_s": STEP_BWD_S, "step_fwd_s": STEP_FWD_S,
                    "step_host_enqueue_s": STEP_HOST_S, "classes_ms": CLASSES},
         "nominal_device_ceiling_x": NOMINAL, "h200_step_s": H200_STEP_S, "bar_s": BAR_S,
         "sweep": rows,
         "bfp8_nominal": {"block_wall_x": WALL_MS / w, "step_s": st, "step_x": STEP_S / st,
                          "x_off_h200": st / H200_STEP_S, "left_to_bar_x": st / BAR_S},
         "roof_rederived": R,
         "host_floor": {"step_s": free, "step_x": STEP_S / free,
                        "x_off_h200": free / H200_STEP_S, "left_to_bar_x": free / BAR_S}},
        indent=1))
    print("\nwrote", HERE / "pricing.json")




# ---------------------------------------------------------------- ROOF re-derivation
# bcx-plan's bracket (b needs 11.1x; bf16 is 8.73x from the DRAM roof, bfp8 17.5x) was computed
# on the `perf/hallgrad` census PROXY unit, which the CENSUS ruling superseded. The real block is
# 12.7x that unit with a different op mix, so the bracket has to be re-derived on the real block.
# Per-class utilisation against that class's OWN roof, from bcx-realcensus.
UTIL = {"layout": (79.2, 0.63), "matmul": (65.4, 0.15), "eltwise": (54.8, 0.89),
        "reduction": (7.3, 0.88), "softmax": (2.9, 0.84), "layernorm": (0.9, 0.68),
        "other": (0.1, 1.0)}


def roofs():
    at_roof = sum(ms * u for ms, u in UTIL.values())
    bf16_x = DEV_MS / at_roof
    bfp8_x = DEV_MS / (at_roof / NOMINAL)
    print("\n================ ROOF, re-derived on the REAL block ================")
    print(f"block backward device time {DEV_MS:.1f} ms; every op at its OWN class roof "
          f"{at_roof:.1f} ms")
    print(f"  bf16 DRAM roof is {bf16_x:.2f}x away on the real block  "
          f"(bcx-plan said 8.73x, on the superseded proxy unit)")
    print(f"  bfp8 DRAM roof is {bfp8_x:.2f}x away on the real block  "
          f"(bcx-plan said 17.5x)")
    need = STEP_S / BAR_S
    print(f"  the bar needs {need:.2f}x on the whole step ({STEP_S:.3f} -> {BAR_S:.3f} s)")

    # what each roof gives as a whole step, with the host floor and then without it
    dev_bwd = 10.76                      # 48 x 0.2108 + 4 x 0.1599, bcx-realcensus
    for name, x in (("bf16 roof", bf16_x), ("bfp8 roof", bfp8_x)):
        bwd_dev = dev_bwd / x
        clamped = STEP_FWD_S + max(bwd_dev + (STEP_BWD_S - dev_bwd), STEP_HOST_S)
        traced = STEP_FWD_S + bwd_dev    # host enqueue removed by trace capture
        print(f"  {name:9s}: bwd device {dev_bwd:.2f} -> {bwd_dev:.2f} s | step today "
              f"{clamped:.2f} s ({STEP_S/clamped:.2f}x, {clamped/BAR_S:.2f}x over bar) | "
              f"step if traced {traced:.2f} s ({STEP_S/traced:.2f}x, {traced/BAR_S:.2f}x over bar)")
    print("NEITHER roof contains the target on the real block, with or without the host floor.")
    print("bcx-plan's bracket -- 'the bar sits between two measured roofs and bfp8 is the only")
    print("one that contains it' -- does not survive being re-derived on the block it has to run on.")
    return {"at_roof_ms": at_roof, "bf16_roof_x": bf16_x, "bfp8_roof_x": bfp8_x,
            "step_x_needed": need}


if __name__ == "__main__":
    main()
