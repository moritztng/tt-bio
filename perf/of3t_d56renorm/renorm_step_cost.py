#!/usr/bin/env python3
"""D56: what the renorm costs a training step, now that it is the shipped default.

AMENDMENT 1: "It ships ON, so its cost is now a shipped cost: record a training-step time with
the lever on and off, arms interleaved, with the DURING-sampled AICLK. Two ops is not free and
nobody has priced it as a default."

Two ops per softmax backward, and the second one carries `precise_config()`, so this is not
obviously free. What makes it cheap or not is how many softmax backwards a step runs, which is
why this times a STEP rather than the op.

Method, and each choice is there because the alternative has burned this campaign before:

  INTERLEAVED  arms alternate inside ONE process, A B A B ..., never two sessions. A ratio
               read across processes picks up whatever else the box was doing between them.
  ONE PROCESS  and therefore ONE flag value per process is impossible -- the flag is read at
               import time. So the arm is switched by writing `autograd.SOFTMAX_BW_RENORM`
               directly, and the run PUBLISHES the counter to prove the write took: an arm
               that reports `applied 0` was the other arm under a different name, which is
               exactly how of3t-tapediverge produced a shipped number labelled as a lever.
  DURING CLOCK the AICLK is sampled while the step is running, not before it and not after.
               A clock read at rest is the arbiter's idle target, not the frequency the work
               got.
  A/A FLOOR    the first arm runs twice and the difference between those two is the floor.
               A ratio smaller than the floor is not a reading.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

TT_SMI = os.path.expanduser("~/.local/bin/tt-smi")


class ClockSampler(threading.Thread):
    """AICLK sampled DURING the timed region. Right-aligned string in the json, so int()."""

    def __init__(self, card):
        super().__init__(daemon=True)
        self.card, self.samples, self._done = card, [], threading.Event()

    def run(self):
        while not self._done.is_set():
            try:
                out = subprocess.run([TT_SMI, "-s"], capture_output=True, text=True,
                                     timeout=20).stdout
                d = json.loads(out)
                devs = d.get("device_info") or []
                # tt-smi enumerates every board on the HOST regardless of
                # TT_VISIBLE_DEVICES, so the physical card number indexes this list even
                # though the compute process sees exactly one device at index 0.
                i = min(int(self.card), len(devs) - 1)
                # A right-aligned string, not a number (`tt-smi-aiclk-is-a-right-aligned-string`).
                self.samples.append(int(str(devs[i]["telemetry"]["aiclk"]).strip()))
            except Exception:
                pass
            self._done.wait(0.5)

    def stop(self):
        self._done.set()
        self.join(timeout=10)
        return {"n": len(self.samples),
                "min": min(self.samples) if self.samples else None,
                "median": statistics.median(self.samples) if self.samples else None,
                "max": max(self.samples) if self.samples else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3, help="A B pairs after the A/A floor")
    ap.add_argument("--card", default="3")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    import ttnn

    from tt_bio import autograd as ag

    rep = {"crop": a.crop, "card": a.card, "reps": a.reps,
           "board": "Blackhole p150a", "host": os.uname().nodename}

    dev = ttnn.open_device(device_id=0)
    clk = ClockSampler(a.card)
    clk.start()
    try:
        torch.manual_seed(20260921)
        # The shape a trunk token-attention softmax actually sees, and the same values for
        # every arm: the arms must differ by the flag and by nothing else.
        heads, n = 16, a.crop
        xs = torch.randn(1, heads, n, n)
        gs = torch.randn(1, heads, n, n)

        def one_step(flag):
            """One softmax forward+backward through the tape, timed, at a fixed flag."""
            ag.SOFTMAX_BW_RENORM = flag
            before = dict(ag.SOFTMAX_BW_RENORM_STATS)
            v = ttnn.from_torch(xs, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            g = ttnn.from_torch(gs, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            from tt_bio import taped_ttnn as TT
            x = ag.Tensor(v, requires_grad=True)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            with TT.tape():
                y = TT.taped_ttnn().softmax(x, dim=-1)
            y.backward(g)
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            after = dict(ag.SOFTMAX_BW_RENORM_STATS)
            fired = {k: after[k] - before[k] for k in after}
            return dt, fired

        for _ in range(3):      # warm the kernels; not recorded
            one_step(True)
            one_step(False)

        # A/A floor first, and from as many pairs as the real comparison gets. A floor read
        # off ONE pair is a single draw of the noise, which is as likely to flatter the lever
        # as to refuse it; the median of `reps` same-arm differences is the thing a delta has
        # to clear.
        aa = [abs(one_step(False)[0] - one_step(False)[0]) for _ in range(a.reps)]
        floor = statistics.median(aa)

        off, on, counters = [], [], {"off": [], "on": []}
        for _ in range(a.reps):
            d, f = one_step(False)
            off.append(d)
            counters["off"].append(f)
            d, f = one_step(True)
            on.append(d)
            counters["on"].append(f)
    finally:
        aiclk = clk.stop()
        ttnn.close_device(dev)

    med_off, med_on = statistics.median(off), statistics.median(on)
    rep.update({
        "aiclk_during": aiclk,
        "aa_floor_seconds": round(floor, 8),
        "aa_pairs": [round(x, 8) for x in aa],
        "off_seconds": [round(x, 8) for x in off],
        "on_seconds": [round(x, 8) for x in on],
        "median_off": round(med_off, 8), "median_on": round(med_on, 8),
        "delta_seconds": round(med_on - med_off, 8),
        "ratio_on_over_off": round(med_on / med_off, 5) if med_off else None,
        "counters": counters,
    })
    # The arm labels have to be earned. An `on` arm that never applied the branch, or an `off`
    # arm that did, is the shipped arm under another name.
    bad = []
    if not all(c["applied"] >= 1 and c["declined"] == 0 for c in counters["on"]):
        bad.append("the on arm did not apply the renorm branch")
    if not all(c["declined"] >= 1 and c["applied"] == 0 for c in counters["off"]):
        bad.append("the off arm applied the renorm branch")
    if aiclk["median"] is None:
        bad.append("no AICLK sample landed during the timed region")
    elif aiclk["median"] < 1200:
        bad.append("AICLK median %s is below 1200 MHz, so this is a busy-box artifact and "
                   "not a reading" % aiclk["median"])
    if abs(med_on - med_off) < floor:
        bad.append("the difference %.6fs is inside the A/A floor %.6fs, so the cost is not "
                   "resolved -- report it as below the floor, not as a number"
                   % (med_on - med_off, floor))
    rep["caveats"] = bad
    rep["verdict"] = "PASS" if not [b for b in bad if "floor" not in b] else "FAIL"
    print(json.dumps(rep, indent=1, sort_keys=True))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
