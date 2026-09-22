#!/usr/bin/env python3
"""Does this row's tape change move the gradient it computes?

PROTOCOL SS3d fixes the bars in advance: per-tensor relative L2 5.0e-02, median 2.0e-02. This
measures against them, per weight, and names the worst tensor by its parameter path rather than
reporting a mean or a norm -- a mean hides the one tensor that disagrees and K52 is the standing
rule that a count beats a norm.

The object is a 4-block OF3 pairformer stack at the checkpoint's own dims, taped, seeded
deterministically, with every weight declared as a leaf. It exercises all three of this row's
tape changes at once: the softmax box (`softmax_in_place` runs in every triangle attention), the
matmul backward's broadcast reduction, and the unsharding at backward dispatch.

Run the SAME file against two checkouts on the SAME card and diff the dumps:

    gradient_ab.py --repo <this worktree>        --out a.pt
    gradient_ab.py --repo <detached main>        --out b.pt
    gradient_ab.py --compare a.pt b.pt --json out.json

num_recycles is not a parameter here and that is deliberate: this is one stack, not the
recycling trunk, so the pass-29 amendment's warning does not apply -- there is no total-versus-
partial derivative to confuse. Stated rather than left implicit, because a gradient artifact
without its recycle count is what that amendment is about.

THE NEGATIVE CONTROL: `--perturb` moves one weight by one bf16 ulp and the comparison must then
FAIL its own bars. A gradient comparison that cannot fail proves nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import traceback
from pathlib import Path


def compare(pa, pb, bar=5.0e-2, median_bar=2.0e-2):
    import torch
    a = torch.load(pa, map_location="cpu")
    b = torch.load(pb, map_location="cpu")
    ga, gb = a["grads"], b["grads"]
    only_a = sorted(set(ga) - set(gb))
    only_b = sorted(set(gb) - set(ga))
    rows = []
    for k in sorted(set(ga) & set(gb)):
        x, y = ga[k].to(torch.float64), gb[k].to(torch.float64)
        if x.shape != y.shape:
            rows.append((k, float("inf"), "shape"))
            continue
        den = torch.linalg.vector_norm(x)
        num = torch.linalg.vector_norm(x - y)
        rows.append((k, float(num / den) if float(den) > 0 else float(num), "ok"))
    rows.sort(key=lambda r: -r[1])
    vals = sorted(r[1] for r in rows)
    med = vals[len(vals) // 2] if vals else float("nan")
    worst = rows[0] if rows else (None, float("nan"), "empty")
    bitex = sum(1 for r in rows if r[1] == 0.0)
    return {
        "a": {"path": str(pa), "head": a["head"], "n_grads": len(ga)},
        "b": {"path": str(pb), "head": b["head"], "n_grads": len(gb)},
        "compared": len(rows),
        "only_in_a": only_a[:10], "only_in_b": only_b[:10],
        "n_only_in_a": len(only_a), "n_only_in_b": len(only_b),
        "bit_identical": bitex,
        "worst_tensor": worst[0], "worst_rel_l2": worst[1],
        "median_rel_l2": med,
        "bar_per_tensor": bar, "bar_median": median_bar,
        "top5": [{"tensor": k, "rel_l2": v} for k, v, _ in rows[:5]],
        "verdict": ("PASS" if (worst[1] <= bar and med <= median_bar
                               and not only_a and not only_b) else "FAIL"),
    }


def run(a):
    import torch
    import ttnn
    repo = a.repo.resolve()
    sys.path.insert(0, str(repo))
    import tt_bio
    if not tt_bio.__file__.startswith(str(repo)):
        raise SystemExit(f"tt_bio came from {tt_bio.__file__}, not {repo}")
    from tt_bio import autograd as ag
    from tt_bio import openfold3_weights as OW
    from tt_bio.tenstorrent import get_device, Pairformer, accurate_softmax_site

    dev = get_device()
    torch.set_grad_enabled(False)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    sd = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    c_z = int(sd["layer_norm_z.weight"].shape[0])
    c_s = int(sd["layer_norm_s.weight"].shape[0])
    comb = {}
    for i in range(a.blocks):
        blk = OW._sub(sd, f"pairformer_stack.blocks.{i}")
        for k, v in OW.remap_pairformer_block(blk).items():
            comb[f"layers.{i}.{k}"] = v
    if a.perturb:
        key = sorted(k for k in comb if k.startswith("layers.0") and comb[k].ndim >= 1)[0]
        comb[key] = comb[key] * (1.0 + 2.0 ** -8)
        print("perturbed", key, flush=True)
    del ck, sd

    pf = Pairformer(a.blocks, 32, 4, 24, 16, True, comb, ckc,
                    scale_pair_bias=False, fp32_softmax=True, transpose_bias=True,
                    accurate_softmax=accurate_softmax_site("openfold3.trunk"))

    # Every device weight the stack holds, declared by IDENTITY of its raw handle. A weight the
    # tape does not know about still runs and still returns; its gradient is simply absent, and
    # the comparison would then be over a smaller set on one side than the other -- which is why
    # the dump carries the names and `compare` reports `only_in_a`.
    params, seen = {}, set()

    def walk(obj, prefix="", depth=0):
        if depth > 8 or id(obj) in seen:
            return
        seen.add(id(obj))
        items = obj.items() if isinstance(obj, dict) else (
            list(enumerate(obj)) if isinstance(obj, (list, tuple)) else
            vars(obj).items() if hasattr(obj, "__dict__") else [])
        for k, v in items:
            if isinstance(k, str) and k.endswith("_cache"):
                continue
            name = f"{prefix}{k}"
            if isinstance(v, ttnn.Tensor):
                params[name] = ag.parameter(v)
            elif isinstance(v, (list, tuple, dict)) or hasattr(v, "__dict__"):
                walk(v, name + ".", depth + 1)
    walk(pf, "pairformer_stack.")

    g = torch.Generator().manual_seed(a.seed)
    s_t = torch.randn(1, a.tokens, c_s, generator=g) * 0.5
    z_t = torch.randn(1, a.tokens, a.tokens, c_z, generator=g) * 0.5
    mk = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    st = ag.Tensor(mk(s_t), requires_grad=True)
    zt = ag.Tensor(mk(z_t), requires_grad=True)
    with ag.tape():
        so, zo = pf(st, zt, None, None, None)
    roots = [t for t in (so, zo) if isinstance(t, ag.Tensor)]
    ag.backward(roots)
    ttnn.synchronize_device(dev)

    grads = {n: ttnn.to_torch(t.grad).to(torch.float32)
             for n, t in params.items() if getattr(t, "grad", None) is not None}
    ag.release_pins()
    head = os.popen(f"git -C {repo} rev-parse HEAD").read().strip()
    torch.save({"grads": grads, "head": head, "repo": str(repo),
                "declared": len(params), "with_grad": len(grads),
                "tokens": a.tokens, "blocks": a.blocks, "seed": a.seed,
                "perturbed": bool(a.perturb),
                "recycles": "N/A -- one pairformer stack, not the recycling trunk"}, a.out)
    print(f"{len(grads)} of {len(params)} declared weights have a gradient -> {a.out}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--ckpt", type=Path, default=Path.home() / ".boltz" / "of3-p2-155k.pt")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if a.compare:
        r = compare(*a.compare)
        r["host"] = socket.gethostname()
        r["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(json.dumps(r, indent=1), flush=True)
        if a.json:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps(r, indent=1))
        return 0
    if not (a.repo and a.out):
        ap.error("--repo and --out are required unless --compare is given")
    try:
        run(a)
    except Exception:                                                    # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
