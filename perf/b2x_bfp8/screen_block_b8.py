#!/usr/bin/env python3
"""One Boltz-2 pairformer block at 512 aa, bf16 pair track against bfloat8_b.

The screen for bet P1, and falsifier (b) of the whole precision axis: does the block's TIME
follow its BYTES? bfloat8_b changes the payload per transaction (2048 -> 1088 B a tile) and not
the transaction count, so a path bound by page-read latency shows the byte cut and none of the
time. The two deltas are reported as two separate measured numbers; the ratio between them is
this pass's most valuable output, ahead of the speedup.

Real layer-0 weights off the shipped checkpoint, production shapes, one device open. Arms
alternate b16/b8/b16/b8 so session drift cannot be read as an effect, and the incumbent runs
twice as its own A/A floor. Per arm: block wall time, per sub-module time, DRAM traffic on the
corrected buffer-address counter, and a census of every `ttnn.typecast` the arm inserted, which
is how this bet fails quietly (falsifier (c)).

Bytes come from `perf/b2x_difflayer/real_traffic.py`, which dedupes on BUFFER ADDRESS. The
campaign's published per-call figures came from a counter that deduped on tensor id and so
charged every free metadata view as a full DRAM read; they are overstated ~1.8x on this block
and are not used here.
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
    """Every top-level ttnn call in the capture, by name."""
    ops = Counter()
    depth = 0
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
    """Corrected DRAM traffic plus the op census for one captured block."""
    c = real_counts({"nodes": nodes})
    ops = op_census(nodes)
    return {
        "real_MB": round(c["real_MB"], 3),
        "real_read_MB": round(c["real_r_MB"], 3),
        "real_write_MB": round(c["real_w_MB"], 3),
        "roundtrip_floor_MB": round(c["floor_MB"], 3),
        "n_top_level_ops": int(c["n_ops"]),
        "n_typecast": int(ops.get("ttnn.typecast", 0)),
        "top_ops_MB": [[round(tot / 1e6, 3), name] for tot, name, _w, _r in c["per_op"][:12]],
        "ops": dict(ops.most_common()),
    }


class GateCensus:
    """served / declined counts for every tuned kernel whose gate reads the operand dtype.

    This is how the bet fails quietly. Four hand-tuned kernels on the pair track refuse a
    non-bf16 operand outright and their callers fall back to the stock op, so a bfloat8_b arm
    can trade a fused kernel for a generic one and read as a dtype result. A decline that is
    never counted is indistinguishable from an op that simply got slower.
    """

    #: module attribute -> the value that means "declined"
    TARGETS = [
        ("tt_bio.reblock_permute", "eligible", False),
        ("tt_bio.reblock_permute", "eligible_back", False),
        ("tt_bio.reblock_permute", "eligible_gated", False),
        ("tt_bio.trimul_tail", "eligible", None),        # returns a reason string on decline
        ("tt_bio.softmax_generic", "eligible", False),
        ("tt_bio.tenstorrent", "_pair_proj_minimal_matmul", None),
    ]

    def __init__(self):
        import importlib
        self.stats = {}
        self._orig = []
        for mod_name, attr, declined_is in self.TARGETS:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr, None)
            if fn is None:
                continue
            key = f"{mod_name.rsplit('.', 1)[-1]}.{attr}"
            self._orig.append((mod, attr, fn))
            self.stats[key] = {"served": 0, "declined": 0, "reasons": {}}
            self._wrap(mod, attr, fn, key, declined_is)

    def _wrap(self, mod, attr, fn, key, declined_is):
        st = self.stats[key]

        def wrapper(*a, **k):
            r = fn(*a, **k)
            # `eligible*` returns True/False; the other two return the result or None, and
            # trimul_tail.eligible returns None when it is ELIGIBLE and a reason when not.
            if declined_is is False:
                declined = r is False
            elif key == "trimul_tail.eligible":
                declined = r is not None
            else:
                declined = r is None
            if declined:
                st["declined"] += 1
                reason = r if isinstance(r, str) else "dtype_or_shape"
                st["reasons"][reason] = st["reasons"].get(reason, 0) + 1
            else:
                st["served"] += 1
            return r

        setattr(mod, attr, wrapper)

    def reset(self):
        for v in self.stats.values():
            v["served"] = v["declined"] = 0
            v["reasons"] = {}

    def snapshot(self):
        return {k: dict(v, reasons=dict(v["reasons"]))
                for k, v in self.stats.items() if v["served"] or v["declined"]}


def find_pairformer(model, n_blocks):
    """The trunk `PairformerModule` with `n_blocks` blocks, wherever boltz2 hangs it."""
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
    ap.add_argument("--order", default="b16,b8,b16,b8")
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

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": [g.x, g.y], "arch": str(dev.arch()), "seq": args.seq, "reps": args.reps,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tt_bio": assert_checkout(),
        "byte_counter": "perf/b2x_difflayer/real_traffic.py (buffer-address dedupe)",
    }
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    dump()

    # Production config and the production load path, so the block under test is the shipped
    # one: same checkpoint, same pairformer_args, same kernels.
    work = Path(tempfile.mkdtemp(prefix="b2x-bfp8-screen-"))
    cfg = prodcfg.build_cfg(work / "msa", work / "out")
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t0, 2)
    dump()

    name, pf_mod = find_pairformer(state.model, 64)
    OUT["pairformer_path"] = name
    blk = pf_mod.module.blocks[0]
    dump()

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

    def run_block(z, s):
        return blk(s, z, mask, attn, attn)

    def free(*ts):
        for t in ts:
            if t is not None:
                ttnn.deallocate(t)

    def time_sub(z, sub):
        """One sub-module on this arm's z, timed on its own."""
        op = getattr(blk, sub)
        arg = (z, mask) if sub.startswith("triangle_multiplication") else (
            (z, attn) if sub.startswith("triangle_attention") else (z,))
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        out = op(*arg)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t
        free(out)
        return dt

    def capture_sub(z, sub):
        """The same sub-module under graph capture: what it really moves, on its own."""
        op = getattr(blk, sub)
        arg = (z, mask) if sub.startswith("triangle_multiplication") else (
            (z, attn) if sub.startswith("triangle_attention") else (z,))
        ttnn.synchronize_device(dev)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = op(*arg)
        ttnn.synchronize_device(dev)
        m = measure_capture(ttnn.graph.end_graph_capture())
        free(out)
        return {"real_MB": m["real_MB"], "n_typecast": m["n_typecast"]}

    def arm(label: str):
        pair_b8 = label == "b8"
        T._PAIR_B8 = pair_b8
        # Every fused-input cache the trimul keeps is keyed on shape, not dtype, so an arm flip
        # must not reuse the other arm's weights uploaded at the other dtype.
        for tm in (blk.triangle_multiplication_start, blk.triangle_multiplication_end):
            tm._gp_cache.clear(); tm._gp_bias_cache.clear()
        # The screen drives `PairformerLayer` directly, so the entry cast that
        # `Pairformer.__call__` does once for the whole stack has to be reproduced here:
        # without it the b8 arm runs the shipped bf16 z and scores as an A/A.
        zdt = ttnn.bfloat8_b if pair_b8 else ttnn.bfloat16
        res = {"arm": label, "pair_b8": pair_b8, "z_store": str(zdt)}
        census.reset()

        # warm: compiles every program at this shape and dtype
        z0, s0 = tt(z_t, zdt), tt(s_t)
        s1, z1 = run_block(z0, s0)
        free(z1, s1)

        walls = []
        for _ in range(args.reps):
            z0, s0 = tt(z_t, zdt), tt(s_t)
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            s1, z1 = run_block(z0, s0)
            ttnn.synchronize_device(dev)
            walls.append(time.perf_counter() - t)
            free(z1, s1)
        res["block_s"] = [round(w, 5) for w in walls]
        res["block_median_s"] = round(st.median(walls), 5)

        zc = tt(z_t, zdt)
        subs, sub_mb = {}, {}
        for lab, sub in SUB:
            time_sub(zc, sub)                                # warm this shape+dtype
            subs[lab] = round(st.median([time_sub(zc, sub) for _ in range(3)]), 5)
            sub_mb[lab] = capture_sub(zc, sub)
        free(zc)
        res["sub_median_s"] = subs
        res["sub_sum_s"] = round(sum(subs.values()), 5)
        res["sub_MB"] = sub_mb

        z0, s0 = tt(z_t, zdt), tt(s_t)
        ttnn.synchronize_device(dev)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        s1, z1 = run_block(z0, s0)
        ttnn.synchronize_device(dev)
        res["graph"] = measure_capture(ttnn.graph.end_graph_capture())
        free(z1, s1)
        res["achieved_GBps"] = round(res["graph"]["real_MB"] / 1e3 / res["block_median_s"], 1)
        res["gates"] = census.snapshot()
        return res

    census = GateCensus()
    OUT["gate_targets"] = sorted(census.stats)
    arms = []
    for i, label in enumerate([a for a in args.order.split(",") if a]):
        a = arm(label)
        a["order"] = i
        arms.append(a)
        OUT["arms"] = arms
        dump()
        print(f"[{i}] {label:4s} block {a['block_median_s']*1e3:8.2f} ms  "
              f"{a['graph']['real_MB']:9.1f} MB  {a['achieved_GBps']:6.1f} GB/s  "
              f"typecasts {a['graph']['n_typecast']}", flush=True)

    b16 = [a for a in arms if a["arm"] == "b16"]
    b8 = [a for a in arms if a["arm"] == "b8"]
    t16, t8 = st.median([a["block_median_s"] for a in b16]), st.median([a["block_median_s"] for a in b8])
    B16 = st.median([a["graph"]["real_MB"] for a in b16])
    B8 = st.median([a["graph"]["real_MB"] for a in b8])
    aa = ((max(a["block_median_s"] for a in b16) - min(a["block_median_s"] for a in b16)) / t16
          if len(b16) > 1 else None)
    byte_cut, time_cut = 1 - B8 / B16, 1 - t8 / t16
    OUT["summary"] = {
        "aa_floor_rel": round(aa, 5) if aa is not None else None,
        "block_ms_b16": round(t16 * 1e3, 3), "block_ms_b8": round(t8 * 1e3, 3),
        "MB_b16": round(B16, 1), "MB_b8": round(B8, 1),
        "byte_cut": round(byte_cut, 4), "time_cut": round(time_cut, 4),
        "time_per_byte_realization": round(time_cut / byte_cut, 4) if byte_cut else None,
        "GBps_b16": round(B16 / 1e3 / t16, 1), "GBps_b8": round(B8 / 1e3 / t8, 1),
        "typecast_b16": b16[0]["graph"]["n_typecast"], "typecast_b8": b8[0]["graph"]["n_typecast"],
        "speedup": round(t16 / t8, 4),
        "falsifier_b_fired": bool(time_cut < 0.08 and byte_cut >= 0.45),
    }
    per_sub = {}
    for lab, _sub in SUB:
        t1 = st.median([a["sub_median_s"][lab] for a in b16])
        t2 = st.median([a["sub_median_s"][lab] for a in b8])
        m1 = st.median([a["sub_MB"][lab]["real_MB"] for a in b16])
        m2 = st.median([a["sub_MB"][lab]["real_MB"] for a in b8])
        bc = 1 - m2 / m1 if m1 else None
        tc = 1 - t2 / t1 if t1 else None
        per_sub[lab] = {
            "ms_b16": round(t1 * 1e3, 3), "ms_b8": round(t2 * 1e3, 3),
            "MB_b16": round(m1, 1), "MB_b8": round(m2, 1),
            "byte_cut": round(bc, 4), "time_cut": round(tc, 4),
            "realization": round(tc / bc, 3) if bc else None,
            "GBps_b16": round(m1 / 1e3 / t1, 1), "GBps_b8": round(m2 / 1e3 / t2, 1),
        }
    OUT["per_sub"] = per_sub
    OUT["gates"] = {"b16": b16[0]["gates"], "b8": b8[0]["gates"]}
    dump()
    print(json.dumps(OUT["summary"], indent=1))
    print(json.dumps(per_sub, indent=1))
    print(json.dumps(OUT["gates"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
