#!/usr/bin/env python3
"""Is the mask add's cost per-PASS and linear, or a one-off? The control for the packer model.

`ablate_512.py` measured that REMOVING the mask add saves 1.373 ms of a 6.809 ms op (20.2 %), while
removing the SFPU exponential over the same 128 tiles saves 0.276 ms (4.1 %), and `addgran_512.py`
ruled out the DST barrier (1.0267x). That left one explanation: the cost is the tile PASSES through
L1 that the add makes and the exp does not, priced by the packer at ~64 cycles a tile write.

That story has a hole. A single removal cannot tell a per-pass cost from a one-off that happens to
sit in the same block -- a format reconfigure, a cold mask CB, a first-touch of cb_qk_im. Both look
identical when you delete the stage.

So this arm ADDS a second identical pass instead of removing the first. Adding is also the safe
direction: removing a stage can leave a downstream `cb_wait_front` unsatisfied and wedge the card,
whereas the doubled call is CB-neutral by construction (pop_in1 false, and in0 is popped and
re-pushed with no net effect). Scores become qk + 2*mask, wrong on purpose.

PREDICTED, before the first number:

    If the cost is per-pass and linear, one ADDED pass costs what one REMOVED pass saved:
    +1.373 ms, i.e. ~8.18 ms against the 6.81 ms baseline, and the three arms lie on a straight
    line with slope ~1.37 ms per pass.

    FALSIFIER: an added pass costing under 0.7 ms or over 2.1 ms. Under 0.7 means the first pass was
    paying a one-off and the per-pass attribution -- and with it the packer model's whole basis for
    extrapolating to other kernels by counting tile writes -- is wrong. Over 2.1 means something
    superlinear (CB pressure, L1 thrash) that the model also does not describe.

Interleaved in one process, same operands, same device, medians over --reps.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import triatt_sdpa as TS

S, H, D = 512, 8, 32

# (name, _ABLATE tuple, how many times the mask add runs in this arm)
ARMS = [("passes_0", ("MASKADD",), 0),
        ("passes_1", (), 1),
        ("passes_2", ("MASKADD_X2",), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/b2z_sdpa_floor/linearity_512.json")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    dev = T.get_device()
    res = {"doc": __doc__, "meta": {
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "grid": list(T.COMPUTE_GRID_MAIN),
        "loadavg": os.getloadavg(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shape": [S, H, S, D]}}
    print(json.dumps(res["meta"]), flush=True)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    q, k, v = (dram(torch.randn(S, H, S, D).to(torch.bfloat16)) for _ in range(3))
    bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16))
    scale = D ** -0.5

    def run(abl):
        TS._ABLATE = abl
        o = T._tri_att_sdpa_at(q, k, v, bias, scale)
        assert o is not None
        return o

    # A doubled mask add must change the OUTPUT, or the define never reached the kernel and the
    # arm is silently measuring the baseline twice. Cheap guard against a no-op instrument.
    base = ttnn.to_torch(run(()))
    x2 = ttnn.to_torch(run(("MASKADD_X2",)))
    res["instrument_live"] = {
        "x2_differs_from_baseline": not bool(torch.equal(base, x2)),
        "max_abs_delta": float((base.float() - x2.float()).abs().max())}
    print("instrument_live", json.dumps(res["instrument_live"]), flush=True)
    assert res["instrument_live"]["x2_differs_from_baseline"], \
        "ABLATE_MASKADD_X2 did not change the output -- the define never reached the kernel"

    for _, abl, _n in ARMS:
        for _ in range(3):
            ttnn.deallocate(run(abl))
    ttnn.synchronize_device(dev)

    acc = {name: [] for name, _, _ in ARMS}
    for _ in range(args.reps):
        for name, abl, _n in ARMS:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = run(abl)
            ttnn.synchronize_device(dev)
            acc[name].append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
    TS._ABLATE = ()

    med = {n: st.median(v_) for n, v_ in acc.items()}
    res["arms"] = {n: {"passes": npass, "ms": med[n], "all_ms": acc[n]} for n, _, npass in ARMS}

    removed = med["passes_1"] - med["passes_0"]   # what deleting the pass saved
    added = med["passes_2"] - med["passes_1"]     # what adding an identical pass cost
    res["linearity"] = {
        "removed_one_pass_ms": removed,
        "added_one_pass_ms": added,
        "ratio_added_over_removed": added / removed if removed else None,
        "linear": bool(0.7 <= added <= 2.1),
        "falsifier": "added pass under 0.7 ms (one-off, not per-pass) or over 2.1 ms (superlinear)"}
    print(json.dumps({n: round(med[n], 4) for n in med}, indent=1), flush=True)
    print(json.dumps({k_: (round(v_, 4) if isinstance(v_, float) else v_)
                      for k_, v_ in res["linearity"].items()}, indent=1), flush=True)

    res["meta"]["loadavg_end"] = os.getloadavg()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
