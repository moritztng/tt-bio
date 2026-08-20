"""The Nesso-1 device floor: what 304 pairformer blocks cost on one p150a.

Nesso-1 has no structure module. The GPU reference measures the pairformer stack at 97-99 % of
device time at every rung, so the whole port is this one stack: 288 trunk blocks (48 blocks x 6
passes, recycling_steps=5) plus 16 affinity blocks (8 x 2 ensemble members). The block class is
tt_bio.reference.PairformerNoSeqModule at token_z=128, which tt-bio already ships as
tenstorrent.PairformerModule(n, 32, 4, None, None, False, affinity=True).

Two measurements, deliberately separate:

  --census   counts ttnn enqueues per block by wrapping the ttnn namespace. Structural, so it is
             immune to the box being busy. At 10.9 us/enqueue this alone can decide GO/NO-GO.
  --time     warm wall-clock per block across the size ladder. Timing only; needs a quiet box
             (run it under benchlock.sh).
"""

import argparse, json, os, statistics, sys, time
from pathlib import Path

import torch

torch.set_grad_enabled(False)
torch.manual_seed(893)

TOKEN_Z = 128          # nesso hparams token_z
TRI_ATT_HEAD_DIM = 32  # 4 heads x 32 width, pair-only
TRI_ATT_N_HEADS = 4
TRUNK_BLOCKS = 48      # hparams pairformer 48 blocks
TRUNK_PASSES = 6       # recycling_steps=5 -> 6 passes
AFFINITY_BLOCKS = 8    # each affinity_module pairformer_stack
AFFINITY_MEMBERS = 2   # 2-member ensemble
CROP_BUDGET = 256      # --refine_protein_inference token budget (CLI default)


def build(n_blocks):
    from tt_bio.tenstorrent import PairformerModule
    from tt_bio.reference import PairformerNoSeqModule as RefPairformer

    tt = PairformerModule(n_blocks, TRI_ATT_HEAD_DIM, TRI_ATT_N_HEADS, None, None, False, affinity=True)
    ref = RefPairformer(TOKEN_Z, n_blocks, v2=True).eval()
    tt.load_state_dict(ref.state_dict(), strict=False)
    return tt


def make_inputs(n):
    z = 26 * torch.randn(1, n, n, TOKEN_Z)
    pair_mask = torch.ones(1, n, n)
    return z, pair_mask


# ---------------------------------------------------------------- enqueue census

SKIP = {"open_device", "close_device", "from_torch", "to_torch", "synchronize_device",
        "deallocate", "manage_device", "GetMemoryConfig", "get_memory_config"}


def wrap_ttnn(counts, callers=None):
    """Wrap every callable in the ttnn namespace with a counter. Returns the unwind list.

    `callers`, when given, additionally records which tt_bio module issued each `generic_op`, so a
    fused kernel that declines at one size and serves at another is visible by name rather than as
    a bare count.
    """
    import ttnn
    import inspect
    import sys as _sys
    undo = []
    for name in dir(ttnn):
        if name.startswith("_") or name in SKIP:
            continue
        fn = getattr(ttnn, name, None)
        if not callable(fn) or inspect.isclass(fn) or inspect.ismodule(fn):
            continue
        def make(fn=fn, name=name):
            def counted(*a, **kw):
                counts[name] = counts.get(name, 0) + 1
                if callers is not None and name == "generic_op":
                    f = _sys._getframe(1)
                    who = f.f_globals.get("__name__", "?")
                    callers[who] = callers.get(who, 0) + 1
                return fn(*a, **kw)
            return counted
        try:
            setattr(ttnn, name, make())
            undo.append((name, fn))
        except Exception:
            pass
    return undo


def unwrap_ttnn(undo):
    import ttnn
    for name, fn in undo:
        setattr(ttnn, name, fn)


def fused_stats():
    """served/declined for every tt-bio fused path a pairformer block can reach."""
    out = {}
    from tt_bio import trimul_tail
    out["trimul_tail_F1"] = {"served": trimul_tail.STATS[0], "declined": trimul_tail.STATS[1],
                             "rejects": {f"{k[0]}|{k[1]}": v for k, v in trimul_tail.REJECTS.items()}}
    try:
        from tt_bio import tenstorrent as TT
        out["fp32_softmax"] = dict(TT.FP32_SOFTMAX_STATS)
    except Exception as e:
        out["fp32_softmax"] = str(e)
    for mod, attr in (("sdpa_generic", "STATS"), ("triatt_qkv", "STATS"),
                      ("mm_generic", "STATS"), ("reblock_permute", "STATS")):
        try:
            m = __import__(f"tt_bio.{mod}", fromlist=[mod])
            v = getattr(m, attr, None)
            out[mod] = v if v is None else (dict(v) if isinstance(v, dict) else list(v))
        except Exception as e:
            out[mod] = f"n/a: {e}"
    return out


def census(n, out):
    tt = build(1)
    z, pair_mask = make_inputs(n)
    tt(None, z, pair_mask=pair_mask)          # compile + mask cache, uncounted
    counts, callers = {}, {}
    undo = wrap_ttnn(counts, callers)
    try:
        tt(None, z, pair_mask=pair_mask)
    finally:
        unwrap_ttnn(undo)
    # transfers the real port would not repeat per block
    host = counts.pop("from_torch", 0) + counts.pop("to_torch", 0)
    total = sum(counts.values())
    blocks = TRUNK_BLOCKS * TRUNK_PASSES + AFFINITY_BLOCKS * AFFINITY_MEMBERS
    res = {
        "seq_len": n,
        "enqueues_per_block": total,
        "host_transfer_calls_excluded": host,
        "blocks_per_prediction": blocks,
        "enqueues_per_prediction": total * blocks,
        "dispatch_s_at_10p9us": round(total * blocks * 10.9e-6, 4),
        "top_ops": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:25]),
        "generic_op_by_module": callers,
        "fused_stats": fused_stats(),
    }
    out.append(res)
    print(json.dumps(res, indent=1), flush=True)


# ---------------------------------------------------------------------- timing

def time_blocks(n, n_blocks, reps, out):
    tt = build(n_blocks)
    z, pair_mask = make_inputs(n)
    ts = []
    for r in range(reps + 1):
        t0 = time.perf_counter()
        tt(None, z, pair_mask=pair_mask)
        dt = time.perf_counter() - t0
        if r:
            ts.append(dt)
        cold = " (cold, dropped)" if r == 0 else ""
        print(f"  n={n} blocks={n_blocks} rep{r}{cold} {dt:.4f}s", flush=True)
    med = statistics.median(ts)
    res = {
        "seq_len": n,
        "n_blocks": n_blocks,
        "reps_warm": len(ts),
        "module_s_median": round(med, 5),
        "spread_max_over_min": round(max(ts) / min(ts), 4),
        "per_block_ms": round(1000 * med / n_blocks, 3),
    }
    out.append(res)
    print(json.dumps(res), flush=True)
    return res


# ------------------------------------------------------------------- region screen

# Leave-one-out inside the 48-block module, not per-op in isolation. An isolated per-op timing
# over-syncs and inflates the cost roughly 2x against batched work, so the price of a region has
# to be read as a difference between two batched runs of the same depth.
SKIP = set()


def patch_layer():
    from tt_bio import tenstorrent as TT
    import ttnn

    def call(self, s, z, mask=None, attn_mask_start=None, attn_mask_end=None, extra_attn_bias=None):
        for name, fn, m in (
            ("tri_mul_start", self.triangle_multiplication_start, mask),
            ("tri_mul_end", self.triangle_multiplication_end, mask),
            ("tri_att_start", self.triangle_attention_start, attn_mask_start),
            ("tri_att_end", self.triangle_attention_end, attn_mask_end),
        ):
            if name in SKIP:
                continue
            z_update = fn(z, m)
            z = ttnn.add_(z, z_update)
            ttnn.deallocate(z_update)
        if "transition_z" not in SKIP:
            z_update = self.transition_z(z)
            z = ttnn.add_(z, z_update)
            ttnn.deallocate(z_update)
        assert not self.transform_s, "region screen is the pair-only pairformer"
        return s, z

    TT.PairformerLayer.__call__ = call


REGIONS = ["tri_mul_start", "tri_mul_end", "tri_att_start", "tri_att_end", "transition_z"]


def region_screen(n, n_blocks, reps, out):
    patch_layer()
    arms = {}
    for skip in [()] + [(r,) for r in REGIONS]:
        SKIP.clear()
        SKIP.update(skip)
        label = "full" if not skip else "minus:" + skip[0]
        tt = build(n_blocks)
        z, pair_mask = make_inputs(n)
        ts = []
        for r in range(reps + 1):
            t0 = time.perf_counter()
            tt(None, z, pair_mask=pair_mask)
            dt = time.perf_counter() - t0
            if r:
                ts.append(dt)
        arms[label] = {"median_s": round(statistics.median(ts), 5),
                       "spread": round(max(ts) / min(ts), 4)}
        print("  %-26s %.5fs  spread %.4f" % (label, arms[label]["median_s"], arms[label]["spread"]), flush=True)
        del tt
    full = arms["full"]["median_s"]
    prices = {}
    for r in REGIONS:
        d = full - arms["minus:" + r]["median_s"]
        prices[r] = {"s_per_module": round(d, 5),
                     "ms_per_block": round(1000 * d / n_blocks, 3),
                     "share_of_block": round(d / full, 4)}
    res = {"seq_len": n, "n_blocks": n_blocks, "full_module_s": full, "arms": arms,
           "region_prices": prices,
           "sum_of_prices_over_full": round(sum(p["s_per_module"] for p in prices.values()) / full, 4)}
    out.append(res)
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--regions", action="store_true")
    ap.add_argument("--sizes", default="128,256,512,768")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]

    result = {"host": os.uname().nodename, "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
              "census": [], "timing": [], "regions": []}
    if a.census:
        for n in sizes:
            census(n, result["census"])
    if a.time:
        for n in sizes:
            time_blocks(n, a.blocks, a.reps, result["timing"])
    if a.regions:
        for n in sizes:
            region_screen(n, a.blocks, a.reps, result["regions"])
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
