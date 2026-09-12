"""Where do an MSALayer's milliseconds go, and which of them see the MSA row axis at all?

`b2z-work-removal` fitted the layer at `139.47 ms + 0.0995 ms per padded row` and attributed the
depth-free intercept to the layer's inner PairformerNoSeq. This checks that attribution, because
the intercept is what a depth lever can never reach: at the 64 rung it is 96 % of the call.

Two levels, both on real weights at the cell's shape:

  * sub-unit -- PairWeightedAveraging / Transition / OuterProductMean / PairformerLayer, timed by
    swapping the four bound attributes of every block for a synchronising proxy. Four extra
    device syncs per block, so the sum runs a little above the unsynced module time; the row
    prints both and the gap is the distortion.
  * inside OuterProductMean -- every ttnn call it makes, recorded with its shapes while a flag
    says we are inside OPM. One sync per op serialises dispatch, so these are reported as SHARES
    of the synced OPM total and never as absolutes.

    python msa_subunit_census.py --out out.json --tokens 512 --depths 64,1024
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

import torch                                                          # noqa: E402
torch.set_grad_enabled(False)

OPS = ("matmul", "linear", "permute", "to_layout", "reshape", "multiply_", "multiply",
       "layer_norm", "concat", "repeat", "add_", "add", "transpose")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--n-msa", type=int, default=35)
    ap.add_argument("--depths", default="64,1024")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]

    import tt_bio.tenstorrent as T
    import ttnn
    import tt_baseline as B
    from fold_ab_multi import patch_boltz2_cfg

    patch_boltz2_cfg()
    target = ROOT / "perf/size512/fixtures" / f"cdk2x2_{a.tokens}.yaml"
    a3m = target.with_suffix(".a3m")
    _one, meta, state = B.build_fold("boltz2", ROOT / f".msa_b2z2_{a.tokens}", target, a3m)
    model = state.model
    msa_wrapper = model.msa_module
    msa = msa_wrapper.module
    n_blocks = len(msa.blocks)

    tok = a.tokens
    seq_pad = T.pad_amount(tok, T.PAIRFORMER_PAD_MULTIPLE) if T.bucket_enabled() else 0
    padded_seq = tok + seq_pad
    c_z = int(model.z_norm.weight.shape[-1])
    c_s = int(model.s_init.in_features)

    z_host = torch.randn(1, padded_seq, padded_seq, c_z, dtype=torch.float32) * 0.02
    emb_host = torch.randn(1, padded_seq, c_s, dtype=torch.float32) * 0.02
    mask_1d = z_host.new_ones(1, padded_seq)
    if seq_pad:
        mask_1d[:, tok:] = 0.0
    mask_tt = msa_wrapper._from_torch(mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(1))
    attn_tt = msa_wrapper._from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9)
    emb_tt = msa_wrapper._from_torch(emb_host)

    legs = {}
    for d in depths:
        assert d >= a.n_msa and d % 32 == 0, f"bad depth {d}"
        m = torch.zeros(1, d, padded_seq, 36, dtype=torch.float32)
        m[:, : a.n_msa, :tok, :] = torch.randn(1, a.n_msa, tok, 36) * 0.1
        rowmask = torch.zeros(d, 1, 1, dtype=torch.float32)
        rowmask[: a.n_msa] = 1.0
        legs[d] = dict(m=msa_wrapper._from_torch(m),
                       rowmask=msa_wrapper._from_torch(rowmask))

    dev = T.get_device()

    # ---- sub-unit proxies -------------------------------------------------------------------
    sub_ms = defaultdict(list)
    collect = {"on": False}

    class Timed:
        def __init__(self, inner, name):
            self.inner, self.name = inner, name

        def __call__(self, *args, **kw):
            if not collect["on"]:
                return self.inner(*args, **kw)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = self.inner(*args, **kw)
            ttnn.synchronize_device(dev)
            sub_ms[self.name].append((time.perf_counter() - t0) * 1e3)
            return r

    for blk in msa.blocks:
        for attr, nm in (("pair_weighted_averaging", "pair_weighted_averaging"),
                         ("msa_transition", "msa_transition"),
                         ("outer_product_mean", "outer_product_mean"),
                         ("pairformer_layer", "pairformer_layer")):
            setattr(blk, attr, Timed(getattr(blk, attr), nm))

    def one(d, timed):
        L = legs[d]
        z_tt = msa_wrapper._from_torch(z_host)
        collect["on"] = timed
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        msa(z_tt, L["m"], emb_tt, mask_tt, attn_tt, L["rowmask"], a.n_msa)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        collect["on"] = False
        ttnn.deallocate(z_tt)
        return dt

    for d in depths:
        for _ in range(a.warm):
            one(d, False)

    rows = []
    for d in depths:
        plain = sorted(one(d, False) for _ in range(a.reps))
        sub_ms.clear()
        synced = sorted(one(d, True) for _ in range(a.reps))
        # each sub-unit fired reps*n_blocks times; per-CALL median
        per_call = {k: round(st.median(v) , 4) for k, v in sub_ms.items()}
        rows.append({
            "padded_depth": d, "true_rows": a.n_msa, "tokens": tok, "padded_tokens": padded_seq,
            "n_blocks": n_blocks, "reps": a.reps,
            "module_ms_unsynced": round(st.median(plain) * 1e3, 4),
            "per_layer_ms_unsynced": round(st.median(plain) * 1e3 / n_blocks, 4),
            "module_ms_synced": round(st.median(synced) * 1e3, 4),
            "subunit_per_call_ms": per_call,
            "subunit_sum_ms": round(sum(per_call.values()), 4),
        })
        print(json.dumps(rows[-1]), flush=True)

    # ---- inside OuterProductMean ------------------------------------------------------------
    op_ms = defaultdict(float)
    op_n = defaultdict(int)
    inside = {"on": False}
    orig = {n: getattr(ttnn, n) for n in OPS if hasattr(ttnn, n)}

    def wrap(name, fn):
        def g(*args, **kw):
            if not inside["on"]:
                return fn(*args, **kw)
            inside["on"] = False                      # do not double-count nested calls
            try:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = fn(*args, **kw)
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) * 1e3
            finally:
                inside["on"] = True
            shp = "x".join(str(s) for s in tuple(args[0].shape)) if args and hasattr(args[0], "shape") else "?"
            op_ms[f"{name} {shp}"] += dt
            op_n[f"{name} {shp}"] += 1
            return r
        return g

    for n, f in orig.items():
        setattr(ttnn, n, wrap(n, f))

    class OpmProbe:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, *args, **kw):
            inside["on"] = True
            try:
                return self.inner(*args, **kw)
            finally:
                inside["on"] = False

    for blk in msa.blocks:
        blk.outer_product_mean = OpmProbe(blk.outer_product_mean.inner)

    opm_detail = {}
    for d in depths:
        op_ms.clear()
        op_n.clear()
        one(d, False)
        tot = sum(op_ms.values())
        top = sorted(op_ms.items(), key=lambda kv: -kv[1])[:14]
        opm_detail[str(d)] = {
            "opm_synced_total_ms_all_blocks": round(tot, 3),
            "top_ops": [{"op": k, "calls": op_n[k], "ms": round(v, 3),
                         "share": round(v / tot, 4)} for k, v in top],
        }
        print(json.dumps({"depth": d, **opm_detail[str(d)]}), flush=True)

    for n, f in orig.items():
        setattr(ttnn, n, f)

    import importlib.metadata as im
    import socket
    out = {"host": socket.gethostname(), "arch": T.arch_name(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"), "ttnn": im.version("ttnn"),
           "note": "sub-unit numbers carry 4 extra device syncs per block; the OPM op table is "
                   "one sync per op and is only readable as shares",
           "rows": rows, "opm_ops": opm_detail}
    Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
