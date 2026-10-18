"""ESMC's LM FFNs in the loaded model, every block's in turn over one page-cell activation ([7, 544, d]), with one
tt_bio.esmc switch off and on (arms alternate per repeat, so host drift lands on both). Device ms of the whole
stack and per layer, and the output's error against the off arm. A grade for an FFN lever where the page cell's
eager wall is host-bound.

    python3 perf/spd_gap_whlm/ffn_stack_bench.py --model esmc-300m [--fast] --out ffn.jsonl
"""
import argparse
import json

import torch
import ttnn

from qkfuse_bench import timed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--alt", default="_ESMC_MM_GATE")
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import esmc as E
    sys_path = __import__("sys").path
    sys_path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "spd_gap_esmf"))
    from esmc_ab import _blocks
    model = E.load_esmc(args.model, fast=args.fast)
    dev = model.tt_device
    ffns = [b.ffn for b in _blocks(model)]
    d = int(ffns[0].norm_weight.shape[-1])
    x = ttnn.from_torch(torch.randn(7, 544, d, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    def stack():
        for f in ffns:
            ttnn.deallocate(f(x))

    outs, ms = {}, {False: [], True: []}
    for on in (False, True):
        setattr(E, args.alt, on)
        stack()                                          # builds what the arm builds lazily
        outs[on] = ttnn.to_torch(ffns[-1](x)).float()
    for _ in range(args.repeat):
        for on in (False, True):
            setattr(E, args.alt, on)
            ms[on].append(timed(stack, dev, 1, args.inner))
    med = {on: sorted(t)[len(t) // 2] for on, t in ms.items()}
    e = outs[True] - outs[False]
    rec = {"model": args.model, "fast": args.fast, "alt": args.alt, "layers": len(ffns),
           "off_ms": med[False], "on_ms": med[True], "per_layer_saved_ms": round((med[False] - med[True]) / len(ffns), 4),
           "off_all": ms[False], "on_all": ms[True],
           "last_layer_vs_off": {"rel_l2": float(e.norm() / outs[False].norm()),
                                 "pcc": float(torch.corrcoef(torch.stack([outs[True].flatten(), outs[False].flatten()]))[0, 1])},
           "stats": {n: list(v) for n, v in vars(E).items() if n.endswith("_STATS") and isinstance(v, list) and "GATE" in n}}
    print(json.dumps(rec), flush=True)
    open(args.out, "w").write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
