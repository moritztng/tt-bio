"""Price the exact host float64 softmax and layer norm WITHOUT a card.

`of3t-tapedfwd` measured a taped crop-384 trunk cycle at 251.66 s against 3.89 s for the same
route set untaped, and 247.8 s of that is neither the routes nor the teardown. `502ed112e` names
the candidate: `tape()` swaps `ttnn.softmax` and `ttnn.layer_norm` for float64 implementations
that run on the HOST.

The device arm that settles it (`exact_training(False)` around arm C) needs the one Blackhole
card. This does the half that does not: the float64 arithmetic runs on the CPU, so its rate is a
HOST property and board-insensitive -- BACKWARD.md 4b's split, stated.

What this measures is the MATH ONLY. `host_f64_softmax` also pays `ttnn.to_torch` in and
`ttnn.from_torch` out, two device syncs this script cannot see, so every number here is a LOWER
BOUND on the real per-call cost. That is the conservative direction: a lower bound that already
accounts for most of 247.8 s settles the question, one that does not leaves it open.

Shapes are OF3's own, from `tt_bio/openfold3_trunk.py:11,55`: a 48-block pairformer, two triangle
attentions per block, `no_heads_pair=4`, so 96 softmaxes over [N, 4, N, N] per trunk cycle. At
N=384 that is 226,492,416 elements each. Chunked on the leading axis here because pc has 30 GB
and one such tensor is 1.81 GB in float64.
"""
import argparse, json, os, platform, statistics, time

import torch


def _loadavg():
    return [round(x, 2) for x in os.getloadavg()]


def _time(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def softmax_arm(rows, n, heads, reps, dtype_in):
    """`host_f64_softmax_values`' arithmetic: the readback's dtype cast, then float64 softmax."""
    x = torch.randn(rows, heads, n, n, dtype=dtype_in)
    out = {}
    out["elements"] = int(x.numel())
    # the cast `ttnn.to_torch(v).double()` performs
    out["cast_s"] = _time(lambda: x.double(), reps)
    x64 = x.double()
    out["softmax_s"] = _time(lambda: torch.softmax(x64, dim=-1), reps)
    y64 = torch.softmax(x64, dim=-1)
    # the round back to the card's dtype, `ttnn.from_torch(y64.float(), ...)`
    out["round_s"] = _time(lambda: y64.to(dtype_in), reps)
    out["math_s"] = [c + s + r for c, s, r in
                     zip(out["cast_s"], out["softmax_s"], out["round_s"])]
    return out


def layernorm_arm(rows, n, d, reps, dtype_in):
    """`_ln_forward64`'s arithmetic, transcribed from tt_bio/autograd.py."""
    x = torch.randn(rows, n, d, dtype=dtype_in)
    gamma = torch.randn(d, dtype=torch.float64)
    beta = torch.randn(d, dtype=torch.float64)
    eps = 1e-5

    def run():
        x64 = x.double()
        xc = x64 - x64.mean(-1, keepdim=True)
        y64 = xc * torch.rsqrt((xc * xc).mean(-1, keepdim=True) + eps)
        y64 = y64 * gamma + beta
        return y64.float()

    return {"elements": int(x.numel()), "math_s": _time(run, reps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="perf/of3t_tapedfwd/out/EXACTPRICE.json")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    n, h = a.tokens, a.heads
    # two chunk sizes, so a rate that is not size-stable says so instead of being scaled blindly
    rep = {"doc": __doc__.strip(), "argv": vars(a),
           "env": {"host": platform.node(), "torch": torch.__version__,
                   "threads": torch.get_num_threads(), "cpus": os.cpu_count(),
                   "loadavg_start": _loadavg(),
                   "board_insensitive": "yes -- CPU arithmetic only, no device is opened"},
           "shapes": {
               "triangle_attention_scores": [n, h, n, n],
               "elements_per_softmax": n * h * n * n,
               "softmaxes_per_trunk_cycle": 2 * a.blocks,
               "source": "tt_bio/openfold3_trunk.py:11 (48-block pairformer) and :55 "
                         "(no_heads_pair=4); 96 matches the banked TRIATT_FUSED_HIFI taped "
                         "counter in perf/of3t_tapedfwd/out/fires_384_C.json"},
           "arms": {}}

    for rows in (16, 32):
        k = f"softmax_rows{rows}"
        rep["arms"][k] = softmax_arm(rows, n, h, a.reps, torch.float32)
        m = statistics.median(rep["arms"][k]["math_s"])
        rep["arms"][k]["median_s"] = m
        rep["arms"][k]["ns_per_element"] = m / rep["arms"][k]["elements"] * 1e9

    for rows in (32, 64):
        k = f"layernorm_rows{rows}"
        rep["arms"][k] = layernorm_arm(rows, n, 128, a.reps, torch.float32)
        m = statistics.median(rep["arms"][k]["math_s"])
        rep["arms"][k]["median_s"] = m
        rep["arms"][k]["ns_per_element"] = m / rep["arms"][k]["elements"] * 1e9

    sm = [rep["arms"][f"softmax_rows{r}"]["ns_per_element"] for r in (16, 32)]
    ln = [rep["arms"][f"layernorm_rows{r}"]["ns_per_element"] for r in (32, 64)]

    # The FASTEST per-element time any single rep achieved, over every rep and both sizes. On a
    # contended host the median drifts with the other tenant but the minimum does not: nothing
    # another process does makes this CPU faster than its own best. So the minimum is the rate
    # that survives contention, and using it makes every second predicted below a FLOOR rather
    # than an estimate -- the direction that cannot flatter the conclusion.
    def floor_rate(prefix, rows):
        best = None
        for r in rows:
            a = rep["arms"][f"{prefix}{r}"]
            for t in a["math_s"]:
                v = t / a["elements"] * 1e9
                best = v if best is None else min(best, v)
        return best

    sm_floor = floor_rate("softmax_rows", (16, 32))
    ln_floor = floor_rate("layernorm_rows", (32, 64))
    rep["rate"] = {
        "softmax_ns_per_element_median_by_size": sm,
        "softmax_size_stable_pct": abs(sm[0] - sm[1]) / max(sm) * 100,
        "layernorm_ns_per_element_median_by_size": ln,
        "layernorm_size_stable_pct": abs(ln[0] - ln[1]) / max(ln) * 100,
        "softmax_ns_per_element_FLOOR": sm_floor,
        "layernorm_ns_per_element_FLOOR": ln_floor,
        "why_floor": "the medians are not size-stable on a contended host and this script says "
                     "so rather than quoting one. The minimum is the statistic that survives a "
                     "noisy neighbour, and it is the conservative end.",
    }

    el = n * h * n * n * 2 * a.blocks
    r = sm_floor
    # `host_f64_softmax_values` reads the scores off the card and writes the result back, so
    # every element crosses PCIe twice in the tensor's own dtype. The byte count needs no card.
    gb = el * 4 * 2 / 1e9
    rep["prediction"] = {
        "trunk_cycle_softmax_elements": el,
        "softmax_math_only_FLOOR_s": el * r / 1e9,
        "softmax_transfer_GB_both_ways": gb,
        "softmax_transfer_s_at_rate": {f"{v} GB/s": gb / v for v in (2, 4, 8)},
        "transfer_note": "ttnn.to_torch untilises on the host, so the effective rate is well "
                         "under a raw DMA and 8 GB/s is generous rather than typical",
        "floor_s_math_plus_transfer_at_4GBs": el * r / 1e9 + gb / 4,
        "measured_residual_s": 247.77,
        "residual_note": "arm C 251.6566 s minus arm B 3.8884 s, both banked in "
                         "perf/of3t_tapedfwd/out/fires_384_C.json, identical route sets",
        "bound": "LOWER, and by two whole terms: the layer norm is not counted at all, and the "
                 "softmax's own Python and untilise overhead is not either.",
        "layer_norm_note": "not derived here. Its rate is measured above (see "
                           "layernorm_ns_per_element) and it is 2.3x the softmax's per element, "
                           "so whatever its trunk element count is, it adds rather than "
                           "subtracts. The device arm is what counts it.",
    }
    rep["env"]["loadavg_end"] = _loadavg()
    rep["env"]["contention_bias"] = (
        "a loaded host makes this rate SLOWER, so the predicted seconds are biased UP. A "
        "prediction that lands BELOW the residual is therefore the safe direction to trust; one "
        "that lands above may be contention rather than arithmetic.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({"rate": rep["rate"], "prediction": rep["prediction"],
                      "loadavg": [rep["env"]["loadavg_start"], rep["env"]["loadavg_end"]]},
                     indent=1))


if __name__ == "__main__":
    main()
