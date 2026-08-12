#!/usr/bin/env python3
"""Phase-0 lever census: prove ON/OFF from inside the run, not by reading source.

Every landed ESMFold2 lever carries a sequence-length window, and the windows were all measured
at 512 aa or below. This module counts what the running fold actually did, so a lever that was
gated off is distinguishable from one that fired and did nothing:

  * `esmc.SwiGLUFFN.__call__`  -> which branch the pair transition took (row-blocked / plain)
  * `esmc.SwiGLUFFN._ffn`      -> the `split` and `l1_gated` arguments as passed, per call
  * `reblock_permute.STATS*`   -> served/declined for the plain, back and E6-gated moves
  * `reblock_permute.REJECTS`  -> the reason string for every decline, which is the gate that fired
  * `_triangle_mul_memory_config` -> L1 or DRAM per trimul, the input to the E6 window test
  * `_trimul_chunk_size`, `pair_row_tile` -> the two other size-dependent widths
  * shared-engine classes esmfold2 does not construct -> a call count that must stay 0

Counters are host-side dict increments and are installed on the instrumented fold only, so the
plain folds that carry the wall are untouched.
"""
from collections import Counter

_ON = [False]
CALLS = Counter()          # "<what>|<key>" -> n
_ORIG = []


def on(flag: bool):
    _ON[0] = bool(flag)


def reset():
    CALLS.clear()


def _wrap_meth(cls, meth, fn):
    g = getattr(cls, meth)
    _ORIG.append((cls, meth, g))
    setattr(cls, meth, fn(g))


def _wrap_fn(mod, name, fn):
    g = getattr(mod, name)
    _ORIG.append((mod, name, g))
    setattr(mod, name, fn(g))


def install():
    """Idempotent: installs once per process."""
    if _ORIG:
        return
    import tt_bio.esmc as EC
    import tt_bio.tenstorrent as T
    import tt_bio.reblock_permute as RB
    import tt_bio.esmfold2 as E2

    # --- lever A (split fc1) and lever C (row-blocked L1 SwiGLU product) --------------------
    def ffn(g):
        def w(self, x, split=False, l1_gated=False):
            if _ON[0]:
                CALLS[f"swiglu_ffn|ndim={len(x.shape)},rows={int(x.shape[1])},"
                      f"split={bool(split)},l1_gated={bool(l1_gated)}"] += 1
            return g(self, x, split=split, l1_gated=l1_gated)
        return w
    _wrap_meth(EC.SwiGLUFFN, "_ffn", ffn)

    def call(g):
        def w(self, x):
            if _ON[0]:
                L = int(x.shape[1])
                lo, hi = EC.PAIR_FFN_ROW_BLOCK_SEQ
                # Record the parent shape and the two window predicates SEPARATELY from the
                # branch actually taken; _ffn above reports the branch. A disagreement between
                # them is the bug this census exists to catch.
                CALLS[f"swiglu_call|ndim={len(x.shape)},L={L},minseq_ok={L >= EC.SPLIT_SWIGLU_MIN_SEQ},"
                      f"rowwin_ok={lo <= L <= hi}"] += 1
            return g(self, x)
        return w
    _wrap_meth(EC.SwiGLUFFN, "__call__", call)

    # --- the trimul memory config, which is what decides the E6 window branch ---------------
    def tmc(g):
        def w(seq_len):
            out = g(seq_len)
            if _ON[0]:
                bt = "L1" if out.buffer_type.name.upper().startswith("L1") else "DRAM"
                CALLS[f"trimul_memcfg|seq={int(seq_len)},{bt}"] += 1
            return out
        return w
    _wrap_fn(T, "_triangle_mul_memory_config", tmc)

    def tcs(g):
        def w(seq_len, hidden, batch=1):
            out = g(seq_len, hidden, batch)
            if _ON[0]:
                CALLS[f"trimul_chunk|seq={int(seq_len)},chunk={int(out)}"] += 1
            return out
        return w
    _wrap_fn(T, "_trimul_chunk_size", tcs)

    def prt(g):
        def w(L):
            out = g(L)
            if _ON[0]:
                CALLS[f"pair_row_tile|L={int(L)},tile={int(out)}"] += 1
            return out
        return w
    _wrap_fn(T, "pair_row_tile", prt)

    # --- levers that provably cannot reach this model: the count must stay 0 ----------------
    for name in ("TriangleAttention", "Transition", "PairformerBlock", "TriangleAttentionSDPA"):
        cls = getattr(T, name, None)
        if cls is not None and hasattr(cls, "__call__"):
            def unreach(g, n=name):
                def w(self, *a, **k):
                    if _ON[0]:
                        CALLS[f"unreachable|{n}"] += 1
                    return g(self, *a, **k)
                return w
            _wrap_meth(cls, "__call__", unreach)

    # --- head_dim 48 padding, priced from the live module ------------------------------------
    def apb(g):
        def w(self, *a, **k):
            if _ON[0]:
                CALLS[f"attn_pair_bias|head_dim={self.head_dim},pad={self.head_dim_pad},"
                      f"heads={self.num_heads}"] += 1
            return g(self, *a, **k)
        return w
    _wrap_meth(E2.AttentionPairBias, "__call__", apb)


def snapshot(reset_rb=False):
    """Everything the census knows, as plain JSON-able data."""
    import tt_bio.reblock_permute as RB
    import tt_bio.esmc as EC
    import tt_bio.tenstorrent as T
    out = {
        "calls": dict(sorted(CALLS.items())),
        "reblock": {"plain_served": RB.STATS[0], "declined_total": RB.STATS[1],
                    "back_served": RB.STATS_BACK[0], "gated_served": RB.STATS_GATED[0],
                    "gated_declined": RB.STATS_GATED[1]},
        "rejects": {f"{r}|{list(s)}": n for (r, s), n in sorted(RB.REJECTS.items(),
                                                               key=lambda kv: str(kv[0]))},
        "gate_constants": {
            "SPLIT_SWIGLU": EC._SPLIT_SWIGLU, "SPLIT_SWIGLU_MIN_SEQ": EC.SPLIT_SWIGLU_MIN_SEQ,
            "PAIR_FFN_ROW_BLOCK": EC._PAIR_FFN_ROW_BLOCK,
            "PAIR_FFN_ROW_BLOCK_SEQ": list(EC.PAIR_FFN_ROW_BLOCK_SEQ),
            "REBLOCK_PERMUTE": RB._ENABLED, "REBLOCK_PERMUTE_GATED": RB._ENABLED_GATED,
            "L1_N_MIN": RB.L1_N_MIN, "L1_N_MAX": RB.L1_N_MAX,
            "TRIANGLE_MULT_L1_MAX_SEQ": T.TRIANGLE_MULT_L1_MAX_SEQ,
            "TRIANGLE_MULT_L1_MAX_SEQ_FAST": T.TRIANGLE_MULT_L1_MAX_SEQ_FAST,
            "SEQ_LEN_MORE_CHUNKING": T.SEQ_LEN_MORE_CHUNKING,
            "TRANSITION_H_CHUNK_SIZE_BIG": T.TRANSITION_H_CHUNK_SIZE_BIG,
            "IS_SMALL_GRID": T._IS_SMALL_GRID, "FAST_MODE": T._FAST_MODE,
        },
    }
    if reset_rb:
        RB.STATS[:] = [0, 0]; RB.STATS_BACK[:] = [0, 0]; RB.STATS_GATED[:] = [0, 0]
        RB.REJECTS.clear()
    return out
