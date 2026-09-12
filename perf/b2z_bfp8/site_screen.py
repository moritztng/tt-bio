#!/usr/bin/env python3
"""One Boltz-2 pairformer block at 512 aa, bfloat8_b at ONE storage site at a time.

`b2x-bfp8-pair-track` flipped the whole pair track at once: 1.496 A on the cdk2x2_298 control
against a 0.60 A reject line, and 0.949x on the fold. Read as one number that says "the precision
axis is dead". Read per site it says something else, because the two halves of it have different
causes. The error is dominated by whichever site the 64-block stack amplifies most, and the
slowdown comes from hand-tuned kernels declining a block-format operand, which is a property of
the site, not of the format.

So: one site per arm, the incumbent interleaved between arms as its own A/A floor, and three
numbers per site -- bytes deleted, block time delta, and the census of every tuned kernel that
declined because of it. The realization (time cut / byte cut) is the campaign-level output: it
says whether the phase is bytes-bound or transaction-bound, and it is a result the swarm needs
whatever this bet's verdict is.

Bytes come from `perf/b2x_difflayer/real_traffic.py`, which dedupes on BUFFER ADDRESS. The
tensor-id counter overstates this block ~1.8x and is not used here.
"""
from __future__ import annotations

import argparse, json, os, socket, statistics as st, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prodcfg                                                              # noqa: E402
from prodcfg import REPO, assert_checkout                                   # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x_difflayer"))
from real_traffic import counts as real_counts                              # noqa: E402

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def op_census(nodes):
    ops, depth = Counter(), 0
    for n in nodes:
        t = n.get("node_type")
        if t == "function_start":
            name = str((n.get("params") or {}).get("name", ""))
            if depth == 0 and name.startswith("ttnn."):
                ops[name] += 1
            depth += 1
        elif t == "function_end":
            depth = max(depth - 1, 0)
    return ops


def measure_capture(nodes):
    c = real_counts({"nodes": nodes})
    ops = op_census(nodes)
    return {
        "real_MB": round(c["real_MB"], 3),
        "n_top_level_ops": int(c["n_ops"]),
        "n_typecast": int(ops.get("ttnn.typecast", 0)),
        "ops": dict(ops.most_common(14)),
    }


class GateCensus:
    """served / declined for every tuned kernel whose gate reads the operand dtype.

    This is how a storage-dtype flip fails quietly: the kernel declines, the caller falls back to
    a stock op that moves MORE than the narrower format saves, and the arm reads as a dtype
    result. A decline that is never counted is indistinguishable from an op that got slower.
    """

    TARGETS = [
        ("tt_bio.reblock_permute", "eligible", False),
        ("tt_bio.reblock_permute", "eligible_back", False),
        ("tt_bio.reblock_permute", "eligible_gated", False),
        ("tt_bio.trimul_tail", "eligible", None),        # a reason string means DECLINED
        ("tt_bio.softmax_generic", "eligible", False),
        ("tt_bio.tenstorrent", "_pair_proj_minimal_matmul", None),
        ("tt_bio.triatt_qkv", "qkv_heads", None),
        ("tt_bio.triatt_qkv", "gate_proj", None),
    ]

    def __init__(self):
        import importlib
        self.stats, self._orig = {}, []
        for mod_name, attr, declined_is in self.TARGETS:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr, None)
            if fn is None:
                continue
            key = f"{mod_name.rsplit('.', 1)[-1]}.{attr}"
            self._orig.append((mod, attr, fn))
            self.stats[key] = {"served": 0, "declined": 0}
            self._wrap(mod, attr, fn, key, declined_is)

    def _wrap(self, mod, attr, fn, key, declined_is):
        rec = self.stats[key]

        def wrapper(*a, **k):
            r = fn(*a, **k)
            if declined_is is False:
                declined = r is False
            elif key == "trimul_tail.eligible":
                declined = r is not None
            else:
                declined = r is None
            rec["declined" if declined else "served"] += 1
            return r

        setattr(mod, attr, wrapper)

    def reset(self):
        for v in self.stats.values():
            v["served"] = v["declined"] = 0

    def snapshot(self):
        return {k: dict(v) for k, v in self.stats.items() if v["served"] or v["declined"]}


def find_pairformer(model, n_blocks):
    from tt_bio.tenstorrent import PairformerModule
    for name, m in model.named_modules():
        if isinstance(m, PairformerModule) and m.n_blocks == n_blocks and m.module is not None:
            return name, m
    raise LookupError(f"no PairformerModule with {n_blocks} blocks and a built module")


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sites", default="",
                    help="comma list of sites to screen; default every site in PAIR_B8_SITES")
    args = ap.parse_args()
    OUT_PATH = args.out

    import tempfile
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio import tenstorrent as T
    from tt_bio.worker import _WorkerState
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)

    sites = [s for s in args.sites.split(",") if s] or list(T.PAIR_B8_SITES)
    unknown = [s for s in sites if s not in T.PAIR_B8_SITES]
    assert not unknown, f"unknown site(s) {unknown}; known {list(T.PAIR_B8_SITES)}"

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": [g.x, g.y], "arch": str(dev.arch()), "seq": args.seq, "reps": args.reps,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tt_bio": assert_checkout(), "sites": sites,
        "byte_counter": "perf/b2x_difflayer/real_traffic.py (buffer-address dedupe)",
    }
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z-bfp8-screen-"))
    cfg = prodcfg.build_cfg(work / "msa", work / "out")
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t0, 2)
    dump()

    name, pf_mod = find_pairformer(state.model, 64)
    OUT["pairformer_path"] = name
    blk = pf_mod.module.blocks[0]

    S, C_Z, C_S = args.seq, 128, 384
    torch.manual_seed(0)
    z_t = (torch.randn(1, S, S, C_Z) * 0.35).bfloat16().float()
    z_t = 0.5 * (z_t + z_t.transpose(1, 2))          # z is near-symmetric in a real trunk
    s_t = (torch.randn(1, S, C_S) * 0.6).bfloat16().float()

    def tt(x, dtype=ttnn.bfloat16):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)

    mask = tt(torch.ones(1, S, S))
    attn = tt(torch.zeros(1, 1, 1, S))

    SUB = [("trimul_start", "triangle_multiplication_start"),
           ("trimul_end", "triangle_multiplication_end"),
           ("triatt_start", "triangle_attention_start"),
           ("triatt_end", "triangle_attention_end"),
           ("transition_z", "transition_z")]

    def free(*ts):
        for t in ts:
            if t is not None:
                ttnn.deallocate(t)

    def sub_args(z, sub):
        if sub.startswith("triangle_multiplication"):
            return (z, mask)
        if sub.startswith("triangle_attention"):
            return (z, attn)
        return (z,)

    def time_sub(z, sub):
        op = getattr(blk, sub)
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        out = op(*sub_args(z, sub))
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t
        free(out)
        return dt

    def capture_sub(z, sub):
        op = getattr(blk, sub)
        ttnn.synchronize_device(dev)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = op(*sub_args(z, sub))
        ttnn.synchronize_device(dev)
        m = measure_capture(ttnn.graph.end_graph_capture())
        free(out)
        return {"real_MB": m["real_MB"], "n_typecast": m["n_typecast"]}

    census = GateCensus()
    OUT["gate_targets"] = sorted(census.stats)

    def arm(label: str, site_set: frozenset):
        T._PAIR_B8_SITES = site_set
        # Every fused-input cache the trimul keeps is keyed on shape, not dtype, so an arm flip
        # must not reuse the previous arm's weights uploaded at the other dtype.
        for tm in (blk.triangle_multiplication_start, blk.triangle_multiplication_end):
            tm._gp_cache.clear(); tm._gp_bias_cache.clear()
        # The screen drives `PairformerLayer` directly, so the entry cast that
        # `Pairformer.__call__` does once for the whole stack has to be reproduced here:
        # without it the `z` arm runs the shipped bf16 z and scores as an A/A.
        zdt = ttnn.bfloat8_b if "z" in site_set else ttnn.bfloat16
        res = {"arm": label, "sites": sorted(site_set), "z_store": str(zdt)}
        census.reset()

        z0, s0 = tt(z_t, zdt), tt(s_t)
        s1, z1 = blk(s0, z0, mask, attn, attn)          # warm: compile at this shape and dtype
        free(z1, s1)

        walls = []
        for _ in range(args.reps):
            z0, s0 = tt(z_t, zdt), tt(s_t)
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            s1, z1 = blk(s0, z0, mask, attn, attn)
            ttnn.synchronize_device(dev)
            walls.append(time.perf_counter() - t)
            free(z1, s1)
        res["block_s"] = [round(w, 5) for w in walls]
        res["block_median_s"] = round(st.median(walls), 5)

        zc = tt(z_t, zdt)
        subs, sub_mb = {}, {}
        for lab, sub in SUB:
            time_sub(zc, sub)                            # warm this shape+dtype
            subs[lab] = round(st.median([time_sub(zc, sub) for _ in range(3)]), 5)
            sub_mb[lab] = capture_sub(zc, sub)
        free(zc)
        res["sub_median_s"] = subs
        res["sub_MB"] = sub_mb

        z0, s0 = tt(z_t, zdt), tt(s_t)
        ttnn.synchronize_device(dev)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        s1, z1 = blk(s0, z0, mask, attn, attn)
        ttnn.synchronize_device(dev)
        res["graph"] = measure_capture(ttnn.graph.end_graph_capture())
        free(z1, s1)
        res["achieved_GBps"] = round(res["graph"]["real_MB"] / 1e3 / res["block_median_s"], 1)
        res["gates"] = census.snapshot()
        return res

    # base, site, base, site, ... : every site is paired with a base measured beside it, and the
    # bases are the A/A floor. Contention on this box is 1-10 % and the levers are smaller, so
    # only the ratios are claimed.
    arms, order = [], []
    for s in sites:
        order += [("base", frozenset()), (s, frozenset([s]))]
    order.append(("base", frozenset()))
    if len(sites) > 1:
        order.append(("all", frozenset(sites)))
        order.append(("base", frozenset()))

    # An arm that cannot RUN is a result, not a crash: a storage dtype changes which program
    # config a tuned matmul picks, and on Wormhole's 1.5 MB L1 that is enough to overflow the
    # static CBs. Record it against the site and keep sweeping, or one such site costs the other
    # eight their numbers (which is exactly what happened on card 15 at 10:54Z).
    failed: set = set()
    for i, (label, ss) in enumerate(order):
        if label == "all":
            ss = frozenset(ss) - failed
            if not ss:
                continue
        try:
            a = arm(label, ss)
        except Exception as e:                                        # noqa: BLE001
            T._PAIR_B8_SITES = frozenset()
            failed.add(label)
            a = {"arm": label, "sites": sorted(ss), "order": i,
                 "error": f"{type(e).__name__}: {e}".split("\nbacktrace")[0][:600]}
            arms.append(a)
            OUT["arms"] = arms
            dump()
            print(f"[{i:2d}] {label:13s} FAILED  {a['error'][:160]}", flush=True)
            continue
        a["order"] = i
        arms.append(a)
        OUT["arms"] = arms
        dump()
        print(f"[{i:2d}] {label:13s} block {a['block_median_s']*1e3:8.2f} ms  "
              f"{a['graph']['real_MB']:9.1f} MB  {a['achieved_GBps']:6.1f} GB/s  "
              f"typecasts {a['graph']['n_typecast']}", flush=True)

    T._PAIR_B8_SITES = frozenset()
    base = [a for a in arms if a["arm"] == "base" and "error" not in a]
    if not base:
        OUT["summary"] = {"error": "every base arm failed"}
        dump()
        return 1
    t0_ = st.median([a["block_median_s"] for a in base])
    B0 = st.median([a["graph"]["real_MB"] for a in base])
    aa = ((max(a["block_median_s"] for a in base) - min(a["block_median_s"] for a in base)) / t0_
          if len(base) > 1 else None)

    per_site = {}
    for a in arms:
        if a["arm"] == "base":
            continue
        if "error" in a:
            per_site[a["arm"]] = {"error": a["error"]}
            continue
        t1, B1 = a["block_median_s"], a["graph"]["real_MB"]
        bc, tc = 1 - B1 / B0, 1 - t1 / t0_
        declined = {k: v["declined"] for k, v in a["gates"].items() if v["declined"]}
        base_decl = {k: v["declined"] for k, v in base[0]["gates"].items() if v["declined"]}
        per_site[a["arm"]] = {
            "block_ms": round(t1 * 1e3, 3), "MB": round(B1, 1),
            "byte_cut": round(bc, 4), "time_cut": round(tc, 4),
            "realization": round(tc / bc, 3) if abs(bc) > 1e-9 else None,
            "speedup": round(t0_ / t1, 4),
            "GBps": round(B1 / 1e3 / t1, 1),
            "n_typecast": a["graph"]["n_typecast"],
            "declined": declined, "declined_base": base_decl,
            "sub_ms": a["sub_median_s"],
        }
    OUT["summary"] = {
        "aa_floor_rel": round(aa, 5) if aa is not None else None,
        "base_block_ms": round(t0_ * 1e3, 3), "base_MB": round(B0, 1),
        "base_GBps": round(B0 / 1e3 / t0_, 1),
        "base_typecast": base[0]["graph"]["n_typecast"],
        "n_base_arms": len(base),
    }
    OUT["per_site"] = per_site
    dump()
    print(json.dumps(OUT["summary"], indent=1))
    print(json.dumps(per_site, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
