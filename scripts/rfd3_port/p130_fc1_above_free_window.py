#!/usr/bin/env python3
"""p130 -- the split `fc1` above its free window, which is the size the guard declines.

`state/rfd3-fusion-programme.md` §13.12 item 2. `Transition.__call__` admits `fc1`'s output to L1
only where a third L1 resident does not cost an extra chunk call, and that window closes at 693
tokens for hidden=512 (11 chunks -> 12 at 694, 16 -> 24 at 1024). hidden=256 never leaves it. So
above 693 tokens the hidden=512 sites take the split with a DRAM output, and §13.10 measured that
half alone at a mere 1.05x -- the split's own extra DRAM round trip eats the pinned config's gain.

1.05x isolated is not obviously a gain at the fold, and a lever tuned at 685 tokens that goes
negative at 700 is `one-size-tuning-is-a-standing-defect-class`. This screen prices the declined
path against the shipped call at the two sizes past the window, so the guard is decided by a number
instead of by the 685-token measurement extrapolated.

Three arms per key, all through the SHIPPED `_tuned_linear` rather than a transcription of it:

    shipped     ttnn.linear(x, w, activation="silu")                 -- today
    split_dram  _tuned_linear(x, w)          + ttnn.silu             -- what >693 tokens gets
    split_l1    _tuned_linear(x, w, mem=L1)  + ttnn.silu(mem=L1)     -- what <=693 tokens gets

Every arm is checked bit-exact against `shipped` first. pc cannot fold these sizes (§10.1, host
RAM), but the matmul does not need the fold to exist. PROVISIONAL-ON-PC-CARD0, and the card control
runs before any verdict: three shipped calls at each key, compared to each other.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import ttnn

from tt_bio.rfd3 import model as M
from tt_bio.tenstorrent import get_device

C_Z = 128
# Body-chunk keys at h = 64, which is what `_pair_transition_chunk_h` returns with two residents at
# every one of these sizes. 704 is the first padded width past the free window; 1024 is where the
# third resident would cost 8 extra chunk calls.
KEYS = [
    ("704 tok", (1, 64,  704, C_Z), 512), ("704 tok", (1, 64,  704, C_Z), 256),
    ("1024 tok", (1, 64, 1024, C_Z), 512), ("1024 tok", (1, 64, 1024, C_Z), 256),
]


def one_key(dev, rung, xshape, hidden, ckc):
    tokens = int(rung.split()[0])
    h2 = M._pair_transition_chunk_h(xshape[2], hidden, tokens)
    h3 = M._pair_transition_chunk_h(xshape[2], hidden, tokens, residents=3)
    n2, n3 = -(-tokens // h2), -(-tokens // h3)
    rec = {"rung": rung, "x": list(xshape), "hidden": hidden, "chunk_h_2res": h2,
           "chunk_h_3res": h3, "n_chunks_2res": n2, "n_chunks_3res": n3,
           "in_free_window": n3 == n2, "extra_chunk_calls": max(0, n3 - n2)}

    x = ttnn.from_torch(torch.randn(xshape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.randn((C_Z, hidden), dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    kw = dict(compute_kernel_config=ckc, dtype=ttnn.bfloat16)

    # --- card control first: three shipped calls, compared to each other ------------------
    reps = [ttnn.linear(x, w, activation="silu", core_grid=None, **kw) for _ in range(3)]
    rec["card_control_maxabs"] = [M._mm_maxabs(reps[0], r) for r in reps[1:]]
    ref = reps[0]
    for r in reps[1:]:
        ttnn.deallocate(r)
    rec["shipped_ms"] = round(1e3 * M._mm_time(
        lambda: ttnn.linear(x, w, activation="silu", core_grid=None, **kw)), 4)

    def chain(mem):
        a = M._tuned_linear(x, w, ckc=ckc, dtype=ttnn.bfloat16, core_grid=None, mem=mem)
        return ttnn.silu(a) if mem is None else ttnn.silu(a, memory_config=mem)

    for name, mem in (("split_dram", None), ("split_l1", ttnn.L1_MEMORY_CONFIG)):
        try:
            out = chain(mem)
            rec[name + "_maxabs"] = M._mm_maxabs(out, ref)
            ttnn.deallocate(out)
            rec[name + "_ms"] = round(1e3 * M._mm_time(lambda: chain(mem)), 4)
            rec[name + "_x"] = round(rec["shipped_ms"] / rec[name + "_ms"], 4)
        except Exception as e:                          # an L1 request can simply not fit
            rec[name + "_error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
    rec["pinned"] = M._tuned_pinned(x, w, ttnn.bfloat16, ckc, core_grid=None)
    for t in (ref, x, w):
        ttnn.deallocate(t)
    return rec


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "perf/p130/fc1_above_free_window.json"
    dev = get_device()
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    M._TUNE_MATMUL = True                               # the >2952-atom sizes this screen models
    rows = []
    for rung, xshape, hidden in KEYS:
        r = one_key(dev, rung, xshape, hidden, ckc)
        rows.append(r)
        print("%-9s hidden=%-4d free_window=%-5s h %d->%d chunks %d->%d  pinned=%s"
              % (rung, hidden, r["in_free_window"], r["chunk_h_2res"], r["chunk_h_3res"],
                 r["n_chunks_2res"], r["n_chunks_3res"], r["pinned"]), flush=True)
        print("   control %s | shipped %7.4f | split_dram %7.4f (%.2fx, maxabs %s) | "
              "split_l1 %s"
              % (r["card_control_maxabs"], r["shipped_ms"],
                 r.get("split_dram_ms", float("nan")), r.get("split_dram_x", float("nan")),
                 r.get("split_dram_maxabs", r.get("split_dram_error")),
                 ("%7.4f (%.2fx, maxabs %s)" % (r["split_l1_ms"], r["split_l1_x"],
                                                r["split_l1_maxabs"])
                  if "split_l1_ms" in r else r.get("split_l1_error"))), flush=True)

    # The guard's verdict. Above the free window the lever gets `split_dram` only, so that arm has
    # to beat the shipped call by more than the isolated screen's own inflation to be worth
    # keeping; `tt-bio-isolated-op-timing-oversync-inflates-cost` applies to both arms of it.
    print("\nthe guard's question -- above the free window, is the split still worth taking?")
    for r in rows:
        if r["in_free_window"] or "split_dram_x" not in r:
            continue
        print("   %s hidden=%d: split_dram %.2fx, %s -> %s"
              % (r["rung"], r["hidden"], r["split_dram_x"],
                 "bit-exact" if r.get("split_dram_maxabs") == 0.0 else "NOT BIT-EXACT",
                 "keep" if r["split_dram_x"] >= 1.05 and r.get("split_dram_maxabs") == 0.0
                 else "decline the split entirely at this size"))
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"note": "isolated per-chunk screen, PROVISIONAL-ON-PC-CARD0",
                             "keys": rows}, indent=2) + "\n")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
