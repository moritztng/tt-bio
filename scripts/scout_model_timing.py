#!/usr/bin/env python3
"""Scout-only: end-to-end ESMC-300M device timing, same model + weights on both ttnn versions.

The op-level sweep showed 0.75 carries a large fixed HOST cost per eager dispatch. This asks
the only question that matters for tt-bio: what does that do to a real model forward?

Timed region is bracketed by ttnn.synchronize_device on both sides, so queued device work
cannot leak out of the measurement (`to_torch` is a blocking drain and would otherwise absorb
it). Kernel + program cache are warmed by discarded warmup passes before the timer starts.

--trace replays the forward as a captured device graph (esmc.ESMC trace=True). The harness
then also runs the mandatory trace correctness gates in-process: (a) traced output
bit-identical to same-version eager, (b) traced -> eager -> traced interleaving stays
bit-identical (the ttnn-trace-interleaved-eager-corruption trap).

Usage: TT_VISIBLE_DEVICES=1 python3 scout_model_timing.py <out.json> [reps] [--trace]
"""
import json, os, sys, time
import numpy as np
import ttnn

OUT = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
TRACE = "--trace" in sys.argv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esmc6b_embed_parity import ESMC_SEQS  # noqa: E402
from tt_bio import esmc as tt_esmc  # noqa: E402
from tt_bio import tenstorrent as _TTd  # noqa: E402

import importlib.metadata as md
res = {"ttnn_version": md.version("ttnn"), "reps": REPS, "trace": TRACE, "runs": {}}

if TRACE:
    _TTd.get_device(trace_region_size=1 << 30)  # open the singleton with a trace region
model = tt_esmc.load_esmc("esmc-300m", trace=TRACE)
dev = getattr(model, "device", None) or getattr(getattr(model, "cfg", None), "device", None)


def sync():
    if dev is not None:
        ttnn.synchronize_device(dev)


for label, seqs in (("single_ubiquitin_76", {"ubiquitin": ESMC_SEQS["ubiquitin"]}),
                    ("batch4", {k: ESMC_SEQS[k] for k in
                                ("trpcage", "gb1", "ubiquitin", "lysozyme")})):
    for _ in range(3):                       # warm kernel + program cache, discarded
        tt_esmc.embed_sequences(model, seqs)
    sync()
    per = []
    for _ in range(REPS):
        sync()                               # drain before: region starts empty
        t0 = time.perf_counter()
        out = tt_esmc.embed_sequences(model, seqs)
        sync()                               # drain after: queued device time is inside
        per.append((time.perf_counter() - t0) * 1e3)
    a = np.array(per)
    res["runs"][label] = {"ms": per, "min": float(a.min()), "median": float(np.median(a)),
                          "mean": float(a.mean()), "max": float(a.max()),
                          "spread_max_over_min": float(a.max() / a.min())}
    # fingerprint the output so a timing run doubles as a numerics check
    emb = np.asarray(out[0].per_residue, dtype=np.float64)
    res["runs"][label]["out_sum"] = float(emb.sum())
    res["runs"][label]["out_absmax"] = float(np.abs(emb).max())
    print("  %-22s min %8.2f  median %8.2f  max %8.2f ms  (spread %.2fx)" %
          (label, a.min(), np.median(a), a.max(), a.max() / a.min()), flush=True)

    if TRACE:
        # Gates (a)+(b): traced == eager bit-identical, incl. interleaved re-entry.
        traced1 = tt_esmc.embed_sequences(model, seqs)
        model.trace = False
        eager = tt_esmc.embed_sequences(model, seqs)
        model.trace = True
        traced2 = tt_esmc.embed_sequences(model, seqs)
        ok = all(np.array_equal(np.asarray(t.per_residue), np.asarray(e.per_residue))
                 and np.array_equal(np.asarray(t.per_residue), np.asarray(t2.per_residue))
                 for t, e, t2 in zip(traced1, eager, traced2))
        res["runs"][label]["trace_bit_identical_interleaved"] = bool(ok)
        print(f"  {label:<22s} trace gates (bit-identical, interleaved): "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            sys.exit("TRACE CORRECTNESS GATE FAILED")

json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
print("wrote", OUT)
