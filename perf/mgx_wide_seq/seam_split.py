"""Where OpenDDE's structural-token seam spends its time, part by part.

The seam (`OpenDDE.expand_and_refine`: expander, then the 4-block refiner at the structural-token
width) is 1900 s of the 4221 s opendde 1536 fold on whglx (state/mgx-speed.md). This runs the seam
alone on the real model and the real structural-token features of a ladder fixture, with the
residue-axis trunk outputs replaced by seeded random tensors of the right shape: the seam's op
sequence, its shapes and its chunking decisions depend on the widths, not on the values, so the
times are the fold's. Wall time is not the only output: every host<->device transfer is counted and
timed, and the fused-SDPA route counters are read after each part, because the question is WHICH
mechanism carries the time.

Each part is bracketed by `ttnn.synchronize_device`, so a part's time is its own.

    TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> TT_BIO_LEASE_HOLDER=worker:mgx-wide-seq \
        python perf/mgx_wide_seq/seam_split.py --rung 1536 --out seam_1536.json [--blocks 4]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

import ttnn

ROOT = Path(__file__).resolve().parents[2]

T = defaultdict(float)          # part -> seconds
N_CALLS = defaultdict(int)
XFER = defaultdict(lambda: [0, 0, 0.0])   # direction -> [calls, bytes, seconds]
_STACK = []


def _bytes(t):
    try:
        return int(t.numel()) * t.element_size()
    except Exception:  # noqa: BLE001
        try:
            return int(t.logical_volume()) * (4 if t.dtype == ttnn.float32 else 2)
        except Exception:  # noqa: BLE001
            return 0


def _wrap_xfer(mod, name, direction, size_of):
    f = getattr(mod, name)

    def g(*a, **kw):
        t0 = time.perf_counter()
        out = f(*a, **kw)
        c = XFER[direction]
        c[0] += 1
        c[1] += size_of(a, out)
        c[2] += time.perf_counter() - t0
        return out
    setattr(mod, name, g)


def timed(dev, label, fn):
    def g(*a, **kw):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        _STACK.append(label)
        try:
            out = fn(*a, **kw)
            ttnn.synchronize_device(dev)
        finally:
            _STACK.pop()
        dt = time.perf_counter() - t0
        T[label] += dt
        N_CALLS[label] += 1
        print(f"  {label:28s} {dt:9.2f} s", flush=True)
        return out
    return g


def build_feats(rung):
    from tt_bio import main as M
    from tt_bio import worker as W
    from tt_bio.protenix_data import build_complex_features
    path = ROOT / "perf" / "size512" / "fixtures" / f"cdk2x2_{rung}.yaml"
    chains = M._read_bio_chains(path)
    cfg = {"msa_dir": "/tmp/mgx-wide-seq-msa", "model": "opendde"}
    specs = W._build_chain_specs(chains, Path(cfg["msa_dir"]), cfg, protein_only=False)
    return build_complex_features(specs, chain_ids=[c[0] for c in chains],
                                  bonds=M._read_bio_bonds(path, chains))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=1536)
    ap.add_argument("--blocks", type=int, default=4, help="refiner blocks to run (model has 4)")
    ap.add_argument("--abag", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tt_bio.tenstorrent as TT
    from tt_bio import token_axis
    from tt_bio import triatt_sdpa as TS
    from tt_bio.opendde import OpenDDE
    from tt_bio.opendde_data import build_structural_token_features
    from tt_bio.tenstorrent import get_device

    feats = build_feats(args.rung)
    ifd = build_structural_token_features(feats)
    dev = get_device()
    model = OpenDDE.load_from_checkpoint(abag=args.abag)
    NT = int(feats["restype"].shape[0])
    Ns = int(ifd["parent_residue_idx"].shape[0])
    from tt_bio.protenix import bucketed_width
    print(f"rung {args.rung}: NT={NT} Ns={Ns} refiner width {bucketed_width(Ns)} "
          f"grid {TT.COMPUTE_GRID_MAIN}", flush=True)

    g = torch.Generator().manual_seed(0)
    C = model.expander
    s_inputs = torch.randn(NT, C.c_s_inputs, generator=g)
    s_trunk = torch.randn(NT, C.c_s, generator=g)
    z_trunk = (torch.randn(NT, NT, C.c_z, generator=g) * 0.5).to(torch.bfloat16)

    model.refiner.blocks = model.refiner.blocks[:args.blocks]
    for i, blk in enumerate(model.refiner.blocks):
        for part in ("triangle_multiplication_start", "triangle_multiplication_end",
                     "triangle_attention_start", "triangle_attention_end", "transition_z",
                     "attention_pair_bias", "transition_s"):
            m = getattr(blk, part)
            setattr(blk, part, _Timed(dev, f"b{i}.{part}", m))
    model.expander = _Timed(dev, "expander", model.expander)
    rp = TT.replace_after_refusal

    def _rp(x, dfn, hfn):
        return timed(dev, "pad/slice (replace_after_refusal)", rp)(x, dfn, hfn)
    TT.replace_after_refusal = _rp

    _wrap_xfer(ttnn, "to_torch", "d2h", lambda a, out: _bytes(out))
    _wrap_xfer(ttnn, "from_torch", "h2d", lambda a, out: _bytes(a[0]) if a else 0)
    _wrap_xfer(ttnn, "from_device", "d2h_raw", lambda a, out: _bytes(a[0]) if a else 0)
    _wrap_xfer(ttnn, "to_device", "h2d_raw", lambda a, out: _bytes(a[0]) if a else 0)

    t0 = time.perf_counter()
    out = model.expand_and_refine(ifd, s_inputs, s_trunk, z_trunk)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t0
    for t in out:
        if t is not None and hasattr(t, "is_allocated"):
            ttnn.deallocate(t)

    agg = defaultdict(float)
    for k, v in T.items():
        agg[k.split(".", 1)[-1] if k.startswith("b") and "." in k else k] += v
    rec = {
        "rung": args.rung, "NT": NT, "Ns": Ns, "width": bucketed_width(Ns),
        "blocks": args.blocks, "grid": list(TT.COMPUTE_GRID_MAIN), "seam_wall_s": round(wall, 2),
        "parts_s": {k: round(v, 2) for k, v in sorted(T.items())},
        "by_kind_s": {k: round(v, 2) for k, v in sorted(agg.items(), key=lambda x: -x[1])},
        "xfer": {k: {"calls": v[0], "GB": round(v[1] / 1e9, 3), "s": round(v[2], 2)}
                 for k, v in XFER.items()},
        "sdpa_fused_large_s": list(TT.SDPA_FUSED_LARGE_S_STATS),
        "sdpa_routes": dict(TT.SDPA_ROUTE_COUNTS),
        "sdpa_picks": {f"{a}x{b}": v for (a, b), v in TT.SDPA_CHUNK_PICKS.items()},
        "triatt_pm_stats": list(TS.STATS),
        "triatt_pm_rejects": {f"{r}:{s}": n for (r, s), n in TS.REJECTS.items()},
        "host_acc_keys": sorted(str(k) for k in TT._HOST_ACC_KEYS),
        "acc_concat_host_fallbacks": TT.ACC_CONCAT_HOST_FALLBACKS[0],
        "aiclk_note": "sample AICLK during the run with tt-smi; see the runner script",
        "env": {k: v for k, v in os.environ.items() if k.startswith("TT_BIO_")},
    }
    print(json.dumps(rec, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")


class _Timed:
    """A module stand-in: calls go through `timed`, attribute reads go to the module."""

    def __init__(self, dev, label, mod):
        object.__setattr__(self, "_m", mod)
        object.__setattr__(self, "_f", timed(dev, label, mod))

    def __call__(self, *a, **kw):
        return self._f(*a, **kw)

    def __getattr__(self, k):
        return getattr(self._m, k)


if __name__ == "__main__":
    sys.exit(main())
