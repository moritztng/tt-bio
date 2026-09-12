#!/usr/bin/env python3
"""Name the program that throws when a pair-track site stores bfloat8_b.

Wave 1 recorded `z`, `trimul_mm` and `triatt_qkv` as CANNOT RUN, each with the SAME circular-buffer
total (1,644,960 B against a 1,499,136 B limit) on the same core range. One number for three
different sites means one program fails in all three, not three configs, so the useful output is
its identity: the tt-bio call site, the ttnn op, and the operand shapes and dtypes it was handed.

Prints the full traceback and the last ttnn op entered before the throw. No timing, no arms.
"""
from __future__ import annotations

import argparse, json, os, socket, sys, tempfile, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b2z_bfp8"))
import prodcfg                                                              # noqa: E402
from prodcfg import REPO, assert_checkout                                   # noqa: E402

sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio import tenstorrent as T
    from tt_bio.worker import _WorkerState
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)

    assert args.site in T.PAIR_B8_SITES, f"unknown site {args.site}"
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    out = {"host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "arch": str(dev.arch()), "site": args.site, "seq": args.seq,
           "tt_bio": assert_checkout()}

    work = Path(tempfile.mkdtemp(prefix="b2z2-cbrepro-"))
    state = _WorkerState("tenstorrent")
    state.load_model(prodcfg.build_cfg(work / "msa", work / "out"))

    from tt_bio.tenstorrent import PairformerModule
    pf = next(m for _n, m in state.model.named_modules()
              if isinstance(m, PairformerModule) and m.n_blocks == 64 and m.module is not None)
    blk = pf.module.blocks[0]

    # Every ttnn op that takes tensors, wrapped to remember the last one entered. The throw comes
    # out of program creation inside the op, so the last entry IS the failing op.
    last = {}
    for name in ("matmul", "linear", "add", "multiply", "typecast", "layer_norm", "transpose",
                 "permute", "concat", "slice", "sigmoid", "generic_op", "reshape"):
        fn = getattr(ttnn, name, None)
        if fn is None:
            continue

        def wrap(fn=fn, name=name):
            def w(*a, **k):
                last["op"] = f"ttnn.{name}"
                last["args"] = [f"{'x'.join(str(d) for d in t.shape)}:{t.dtype}:"
                                f"{t.memory_config().buffer_type}"
                                for t in a if hasattr(t, "shape") and hasattr(t, "dtype")]
                last["kwargs"] = {kk: str(vv) for kk, vv in k.items()
                                  if kk in ("dtype", "program_config", "config", "memory_config")}
                return fn(*a, **k)
            return w
        setattr(ttnn, name, wrap())

    S, C_Z, C_S = args.seq, 128, 384
    torch.manual_seed(0)
    z_t = (torch.randn(1, S, S, C_Z) * 0.35).bfloat16().float()
    z_t = 0.5 * (z_t + z_t.transpose(1, 2))
    s_t = (torch.randn(1, S, C_S) * 0.6).bfloat16().float()
    tt = lambda x, d=ttnn.bfloat16: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=d)
    mask, attn = tt(torch.ones(1, S, S)), tt(torch.zeros(1, 1, 1, S))

    T._PAIR_B8_SITES = frozenset([args.site])
    for tm in (blk.triangle_multiplication_start, blk.triangle_multiplication_end):
        tm._gp_cache.clear(); tm._gp_bias_cache.clear()
    zdt = ttnn.bfloat8_b if args.site == "z" else ttnn.bfloat16
    t0 = time.perf_counter()
    try:
        s1, z1 = blk(tt(z_t, zdt), mask, attn, attn) if False else blk(
            tt(s_t), tt(z_t, zdt), mask, attn, attn)
        out["result"] = "RAN, no throw"
        for t in (s1, z1):
            if t is not None:
                ttnn.deallocate(t)
    except Exception as e:                                                  # noqa: BLE001
        out["result"] = "THREW"
        out["error"] = f"{type(e).__name__}: {e}".split("\nbacktrace")[0][:900]
        out["traceback"] = traceback.format_exc()[-4000:]
        out["last_op"] = dict(last)
    out["elapsed_s"] = round(time.perf_counter() - t0, 2)
    T._PAIR_B8_SITES = frozenset()

    txt = json.dumps(out, indent=1)
    print(txt)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
