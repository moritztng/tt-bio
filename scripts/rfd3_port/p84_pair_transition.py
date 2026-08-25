#!/usr/bin/env python3
"""p84 -- what do the eight pair Transitions per step ACTUALLY cost, on the shipped default?

The census (p63) names the pair-shaped Transition as the largest (actual - irreducible) site in
the model by a wide margin: `pairformer z_transition x2 [z,128,H=512]` at 53.5 ms/step of roof
delta and `transition_2 x2 [z,128,H=256]` at 32.1, together 85.6 ms/step = 17.1 s/design. Its
PHASE 1 quotes the shipped cost of the same eight calls as 59.504 + 4*18.9 = 135.1 ms/step.

Those per-call figures (14.876 at H=256 from p46, ~18.9 at H=512) predate the landed
`_PAIR_TRANSITION_L1` lever. p83 has just shown what happens when a price is carried against a
route the model no longer takes: p77's atom-attention baseline was 1.69x the shipped chain and
every prize subtracted from it was inflated by up to 11.914 s/design.

So measure the real thing, on the real code path. `Transition.__call__` is exercised directly --
not a reconstruction of it -- at the production pair shape [1, 685, 704, 128], for both hidden
sizes, with `_PAIR_TRANSITION_L1` on (the shipped default) and off.
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p84/pair_transition.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
I, IP, C = 685, 704, 128
STEPS = 200
CALLS_PER_HIDDEN = 4          # 4x H=256 (transition_2) + 4x H=512 (z_transition) per step


def timeit(fn, dev, n=N, warm=2):
    for _ in range(warm):
        o = fn()
        if o is not None:
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3)
        if o is not None:
            ttnn.deallocate(o)
    return statistics.median(out), min(out), max(out)


def make_transition(dev, ckc, hidden):
    """The shipped Transition with random weights. object.__new__ skips the state-dict load
    so the real __call__/_swiglu run without a checkpoint."""
    t = object.__new__(M.Transition)
    t.dtype = ttnn.bfloat16
    t.compute_kernel_config = ckc

    def w(*shape):
        return ttnn.from_torch(torch.randn(*shape) * 0.02, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev)

    t.norm_w = w(C)
    t.fc1_w = w(C, hidden)
    t.fc2_w = w(C, hidden)
    t.fc3_w = w(hidden, C)
    return t


def main():
    dev = get_device()
    torch.manual_seed(42)
    ckc = M._default_compute_kernel_config()
    x = ttnn.from_torch(torch.randn(1, I, IP, C), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    print("[p84] pair tensor %s  padded %s  lever default=%s  H_CHUNK=%s"
          % (tuple(x.shape), tuple(x.padded_shape), M._PAIR_TRANSITION_L1,
             M._PAIR_TRANSITION_H_CHUNK), flush=True)

    rows = []
    totals = {}
    for hidden in (256, 512):
        t = make_transition(dev, ckc, hidden)
        h = M._pair_transition_chunk_h(IP, hidden, I)
        print("\n=== hidden=%d   chunk h=%d  (%d chunks of %d rows) ==="
              % (hidden, h, -(-I // h), I), flush=True)
        per = {}
        for lever in (True, False):
            M._PAIR_TRANSITION_L1 = lever
            med, lo, hi = timeit(lambda: t(x), dev)
            per["on" if lever else "off"] = med
            print("  _PAIR_TRANSITION_L1=%-5s %9.4f ms/call  [%.4f, %.4f]"
                  % (lever, med, lo, hi), flush=True)
            rows.append(dict(hidden=hidden, lever=lever, chunk_h=h, ms=round(med, 4),
                             lo=round(lo, 4), hi=round(hi, 4)))
        M._PAIR_TRANSITION_L1 = True
        gain = per["off"] - per["on"]
        print("  lever is worth %.4f ms/call = %.3f s/design over %d calls/step"
              % (gain, gain * CALLS_PER_HIDDEN * STEPS / 1000.0, CALLS_PER_HIDDEN), flush=True)
        totals[hidden] = per
        rows.append(dict(hidden=hidden, lever_gain_ms=round(gain, 4),
                         lever_gain_s_per_design=round(gain * CALLS_PER_HIDDEN * STEPS / 1000.0, 3)))

        # Where does one whole-tensor call go? Same six ops _swiglu issues.
        M._PAIR_TRANSITION_L1 = False
        xn = ttnn.rms_norm(x, weight=t.norm_w, epsilon=1e-6, compute_kernel_config=ckc)
        a = ttnn.linear(xn, t.fc1_w, activation="silu", compute_kernel_config=ckc,
                        dtype=ttnn.bfloat16, core_grid=M.BATCH_INVARIANT_GRID)
        b = ttnn.linear(xn, t.fc2_w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                        core_grid=M.BATCH_INVARIANT_GRID)
        m = ttnn.multiply(a, b)
        parts = [
            ("rms_norm", lambda: ttnn.rms_norm(x, weight=t.norm_w, epsilon=1e-6,
                                               compute_kernel_config=ckc)),
            ("fc1 silu [128->%d]" % hidden,
             lambda: ttnn.linear(xn, t.fc1_w, activation="silu", compute_kernel_config=ckc,
                                 dtype=ttnn.bfloat16, core_grid=M.BATCH_INVARIANT_GRID)),
            ("fc2 [128->%d]" % hidden,
             lambda: ttnn.linear(xn, t.fc2_w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                 core_grid=M.BATCH_INVARIANT_GRID)),
            ("multiply a*b", lambda: ttnn.multiply(a, b)),
            ("fc3 [%d->128]" % hidden,
             lambda: ttnn.linear(m, t.fc3_w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                 core_grid=M.CORE_GRID_MAIN)),
        ]
        print("  whole-tensor call, op by op:", flush=True)
        s = 0.0
        for name, fn in parts:
            try:
                med, _, _ = timeit(fn, dev, n=3)
                s += med
                print("    %-24s %9.4f ms" % (name, med), flush=True)
                rows.append(dict(hidden=hidden, part=name, ms=round(med, 4)))
            except Exception as e:
                print("    %-24s EXC %s" % (name, str(e)[:60]), flush=True)
        print("    %-24s %9.4f ms (sum)" % ("", s), flush=True)
        for tt_ in (xn, a, b, m):
            ttnn.deallocate(tt_)
        M._PAIR_TRANSITION_L1 = True

    step_on = CALLS_PER_HIDDEN * (totals[256]["on"] + totals[512]["on"])
    step_off = CALLS_PER_HIDDEN * (totals[256]["off"] + totals[512]["off"])
    print("\n" + "=" * 78, flush=True)
    print("eight pair Transitions per step, shipped default : %8.2f ms/step = %6.3f s/design"
          % (step_on, step_on * STEPS / 1000.0), flush=True)
    print("eight pair Transitions per step, lever OFF       : %8.2f ms/step = %6.3f s/design"
          % (step_off, step_off * STEPS / 1000.0), flush=True)
    print("census PHASE 1 quoted shipped cost              :   135.10 ms/step = 27.020 s/design",
          flush=True)
    print("census overstates the shipped site by           : %8.2f ms/step = %6.3f s/design"
          % (135.10 - step_on, (135.10 - step_on) * STEPS / 1000.0), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "I": I, "IP": IP, "C": C, "steps": STEPS,
        "calls_per_hidden_per_step": CALLS_PER_HIDDEN,
        "step_ms_shipped": round(step_on, 3), "step_ms_lever_off": round(step_off, 3),
        "s_per_design_shipped": round(step_on * STEPS / 1000.0, 3),
        "s_per_design_lever_off": round(step_off * STEPS / 1000.0, 3),
        "census_phase1_step_ms": 135.10,
        "census_overstatement_s_per_design": round((135.10 - step_on) * STEPS / 1000.0, 3),
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
