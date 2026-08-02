"""Is either Transition path WRONG -- the whole one or the chunked one? ANSWER: neither.

This is the probe that ended a long hunt. Every pass before it asked *which path broke*; none asked
whether either was wrong at all. Measured at the real fold's shape (D=8722, W=285), on the rows that
actually differ:

    |whole   - torch fp32 ref|   max 2.5882e-02
    |chunked - torch fp32 ref|   max 2.5882e-02      <- identical to 5 significant figures

Neither path is more correct. The whole-vs-chunked difference is **exactly one bf16 mantissa step**:
relative 3.906e-03, and 2^-8 = 0.00390625. It is confined to the final partial chunk, and it scales
with the data (128 at magnitude 4e4, 0.0156 at magnitude 4.0), which is the signature of rounding
rather than corruption. Changing how an op is decomposed moves the last bit; that is all this is.

Method note, because it is what made the answer trustworthy: swiglu is row-wise over the MSA depth
axis, so a single row's output depends only on that row -- the fp32 reference is computed for JUST
the differing rows rather than the whole 0.6 GiB tensor.

Two reporting rules learned here, both the hard way:
  * An absolute maxabs is meaningless without the scale next to it. `PROBE_REALISTIC=1` uses fan-in
    scaled weights so the output sits at a trained-model magnitude instead of ~4e4.
  * Quote the MEDIAN relative difference at the differing elements, not the max -- the max is taken
    where |ref| is near zero, so it reads 2.3e+02 while the median is 6.4e-03.

Run on the Galaxy with one chip free:
    TT_VISIBLE_DEVICES=<n> [PROBE_REALISTIC=1] python3 -u scripts/abag_xm/probe_transition_vs_torch.py
"""
import os
import torch
import ttnn

from tt_bio.tenstorrent import get_device, MSA_CHUNK_SIZE, Transition, _dtype

D = int(os.environ.get("PROBE_DEPTH", 8722))
W = int(os.environ.get("PROBE_TOKENS", 285))
C = 128
H = 4 * C
CHUNK = int(os.environ.get("PROBE_CHUNK", MSA_CHUNK_SIZE))
NROWS = int(os.environ.get("PROBE_NROWS", 8))     # differing rows to reference-check
REALISTIC = os.environ.get("PROBE_REALISTIC", "0") == "1"


def swiglu_ref(x, w):
    """The reference swiglu, in fp32 on the host. Mirrors Transition.__call__'s swiglu exactly.

    torch_to_tt transposes on upload, so a stored (out, in) weight is used as (in, out) on device;
    here that is `stored.t()`.
    """
    xn = torch.nn.functional.layer_norm(x, (C,), w["norm.weight"], w["norm.bias"], eps=1e-5)
    x1 = torch.nn.functional.silu(xn @ w["fc1.weight"].t())
    x2 = xn @ w["fc2.weight"].t()
    return (x1 * x2) @ w["fc3.weight"].t()


def main():
    dev = get_device()
    torch.manual_seed(0)
    if REALISTIC:
        # randn weights drive the output to ~4e4, nothing like a trained model, and the whole
        # "is this just rounding" question is scale-dependent. Fan-in scaling plus a unit-gain
        # norm gives activations in the range a real checkpoint produces.
        w = {"norm.weight": torch.ones(C), "norm.bias": torch.zeros(C),
             "fc1.weight": torch.randn(H, C) / C ** 0.5,
             "fc2.weight": torch.randn(H, C) / C ** 0.5,
             "fc3.weight": torch.randn(C, H) / H ** 0.5}
    else:
        w = {"norm.weight": torch.randn(C), "norm.bias": torch.randn(C),
             "fc1.weight": torch.randn(H, C), "fc2.weight": torch.randn(H, C),
             "fc3.weight": torch.randn(C, H)}
    kc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, packer_l1_acc=True)
    tr = Transition(w, kc)
    print(f"D={D} W={W} C={C} H={H} chunk={CHUNK}")

    x_host = torch.randn(1, D, W, C)
    x = ttnn.from_torch(x_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=_dtype())

    whole = ttnn.to_torch(tr(x)).to(torch.float32)
    parts = [tr(x[:, s:min(s + CHUNK, D), :, :]) for s in range(0, D, CHUNK)]
    chunked = ttnn.to_torch(ttnn.concat(parts, dim=1)).to(torch.float32)

    diff = (whole - chunked).abs()
    # Absolute maxabs is meaningless without scale. With synthetic weights these outputs run to
    # ~1e3, where one bf16 step is ~8-16, so a "maxabs 128" can be a handful of ULP from a
    # different summation order rather than a defect. Report the scale and the ULP so the number
    # can be judged.
    mag = whole.abs().max().item()
    ulp = 2.0 ** (torch.floor(torch.log2(torch.tensor(max(mag, 1e-30)))).item() - 7)  # bf16: 8-bit mantissa
    print(f"whole vs chunked: maxabs {diff.max():.3e}  frac {(diff > 0).float().mean():.5f}")
    print(f"  output magnitude max {mag:.3e}   bf16 ULP at that magnitude ~{ulp:.3e}"
          f"   -> diff is ~{diff.max().item() / ulp:.1f} ULP, "
          f"rel {diff.max().item() / max(mag, 1e-30):.3e}")
    if diff.max() == 0:
        print("paths agree at this shape -- nothing to attribute")
        print("done")
        return

    # rows (depth indices) that actually disagree, so the reference is computed where it matters
    rows = torch.nonzero(diff.amax(dim=(0, 2, 3)) > 0).flatten()[:NROWS].tolist()
    print(f"differing depth rows: {len(rows)} sampled of "
          f"{int((diff.amax(dim=(0,2,3)) > 0).sum())} -> {rows}")

    ref = swiglu_ref(x_host[:, rows, :, :].to(torch.float32), w)
    ew = (whole[:, rows, :, :] - ref).abs()
    ec = (chunked[:, rows, :, :] - ref).abs()
    rmag = ref.abs().max().item()
    print(f"  reference magnitude max {rmag:.4e}")
    print(f"  |whole   - torch_ref|  max {ew.max():.4e}  mean {ew.mean():.4e}"
          f"  rel {ew.max().item() / max(rmag, 1e-30):.3e}")
    print(f"  |chunked - torch_ref|  max {ec.max():.4e}  mean {ec.mean():.4e}"
          f"  rel {ec.max().item() / max(rmag, 1e-30):.3e}")
    # Relative error evaluated ON THE DIFFERING ELEMENTS. The earlier version divided by the
    # tensor's global max magnitude, which flatters the number whenever the differing elements are
    # small -- the specific weakness I flagged in my own interpretation.
    dsel = (whole[:, rows, :, :] - chunked[:, rows, :, :]).abs()
    msk = dsel > 0
    if msk.any():
        local = ref.abs()[msk]
        rel_local = (dsel[msk] / local.clamp_min(1e-30))
        print(f"  at the {int(msk.sum())} differing elements: |ref| median {local.median():.3e}"
              f"   rel diff median {rel_local.median():.3e}  max {rel_local.max():.3e}")
    if ew.max() > 10 * ec.max():
        print("  VERDICT: the UNCHUNKED path is wrong (chunked matches torch)")
    elif ec.max() > 10 * ew.max():
        print("  VERDICT: the CHUNKED path is wrong (whole matches torch)")
    else:
        print("  VERDICT: inconclusive -- both deviate similarly; look at more rows")
    print("done")


if __name__ == "__main__":
    main()
