"""Does row-blocking AF2's transition change a single bit? (`af2.TRANSITION_ROWBLOCK_BYTES`)

Runs one `ReluTransition` (c 128, hidden 512) on a [1, N, N, 128] pair tensor twice on device,
once unblocked (threshold above the hidden's size) and once row-blocked (threshold 1 byte), and
compares the two outputs bit for bit. Both are also graded against a float64 host reference.
Prints and writes one JSON line per token count.

    TT_VISIBLE_DEVICES=<chip> python perf/spd/bc2_rowblock_exact.py --tokens 480 --out x.json
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import af2  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def reference(x, w):
    x = x.double()
    xn = torch.nn.functional.layer_norm(x, (x.shape[-1],), w["norm.weight"].double(),
                                        w["norm.bias"].double(), eps=1e-5)
    h = torch.relu(xn @ w["fc1.weight"].double().t() + w["fc1.bias"].double())
    return h @ w["fc2.weight"].double().t() + w["fc2.bias"].double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[480])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    dev = get_device()
    g = torch.Generator().manual_seed(a.seed)
    c, hid = 128, 512
    w = {"norm.weight": 1 + 0.1 * torch.randn(c, generator=g),
         "norm.bias": 0.1 * torch.randn(c, generator=g),
         "fc1.weight": torch.randn(hid, c, generator=g) / c ** 0.5,
         "fc1.bias": 0.1 * torch.randn(hid, generator=g),
         "fc2.weight": torch.randn(c, hid, generator=g) / hid ** 0.5,
         "fc2.bias": 0.1 * torch.randn(c, generator=g)}
    mod = af2.ReluTransition(w, af2.compute_kernel_config())
    rows = []
    for n in a.tokens:
        x = torch.randn(1, n, n, c, generator=g).bfloat16()
        xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        outs = {}
        for name, thr in (("unblocked", 1 << 62), ("blocked", 1)):
            af2.TRANSITION_ROWBLOCK_BYTES = thr
            y = mod(xt)
            outs[name] = ttnn.to_torch(y).float()
            ttnn.deallocate(y)
        ref = reference(x, w)

        def rel(y):
            return float((y.double() - ref).norm() / ref.norm())
        diff = (outs["blocked"] != outs["unblocked"])
        row = {"tokens": n, "hidden_mib": n * n * hid * 2 / 2 ** 20,
               "bit_identical": not bool(diff.any()), "elements_differ": int(diff.sum()),
               "max_abs_diff": float((outs["blocked"] - outs["unblocked"]).abs().max()),
               "rel_l2_vs_f64_unblocked": rel(outs["unblocked"]),
               "rel_l2_vs_f64_blocked": rel(outs["blocked"]),
               "arch": str(dev.arch())}
        print(json.dumps(row), flush=True)
        rows.append(row)
        ttnn.deallocate(xt)
    if a.out:
        a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
