"""Is OuterProductMean's depth-axis contraction contaminated by tile padding?

Every component of the chunked MSA path measured bit-exact, yet the fold diverges reproducibly.
This tests the one mechanism that would explain both.

OPM reduces over the MSA depth axis. After `a = ttnn.permute(a, (1, 2, 0))` the DEPTH axis becomes
the last dim, so TILE layout pads it up to a multiple of 32 (e.g. 8998 -> 9024). The matmul then
contracts over the padded extent. If those pad rows are not guaranteed zero, the result depends on
whatever was previously in that memory -- i.e. on allocation history. The whole and chunked branches
have very different allocation histories (the chunked one allocates and frees N slices plus a
concat), which would produce a reproducible per-branch difference while every op tested in isolation
stays bit-exact.

Two independent checks, no chunking involved in either:
  1. SAME input, SAME code, but a different allocation history in between -> do results differ?
  2. depth NOT a multiple of 32 (8998) vs depth that IS (8992) -> is only the padded case unstable?

If check 1 differs, this is a latent correctness bug in a module shared by Boltz-2, Protenix-v2 and
OpenDDE, reachable at any MSA depth that is not a multiple of 32 -- independent of this campaign.

Run on the Galaxy with one chip free:
    TT_VISIBLE_DEVICES=<n> python3 -u scripts/abag_xm/probe_opm_pad_contamination.py
"""
import os
import torch
import ttnn

from tt_bio.tenstorrent import get_device, OuterProductMean, _dtype

S_TOK = int(os.environ.get("PROBE_TOKENS", 288))
C_M = 128
C_OPM = 32          # proj_a / proj_b width
C_Z = 384


def build_opm(kc):
    torch.manual_seed(0)
    w = {
        "norm.weight": torch.randn(C_M), "norm.bias": torch.randn(C_M),
        "proj_a.weight": torch.randn(C_OPM, C_M),
        "proj_b.weight": torch.randn(C_OPM, C_M),
        "proj_o.weight": torch.randn(C_Z, C_OPM * C_OPM),
        "proj_o.bias": torch.randn(C_Z),
    }
    return OuterProductMean(w, kc)


def run(opm, x_host, dev):
    x = ttnn.from_torch(x_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=_dtype())
    out = ttnn.to_torch(opm(x, None, None))
    return out


def churn(dev, mb=512):
    """Dirty the allocator: allocate a large tensor, fill it with a recognisable value, free it.
    Anything OPM's padding picks up afterwards comes from here rather than from zeros."""
    t = ttnn.from_torch(torch.full((mb * 1024 * 1024 // 2 // 128, 128), 7.0),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=_dtype())
    ttnn.deallocate(t)


def main():
    dev = get_device()
    kc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, packer_l1_acc=True)
    opm = build_opm(kc)

    for depth in (int(os.environ.get("PROBE_DEPTH", 8998)), 8992):
        pad = (-depth) % 32
        torch.manual_seed(1)
        x_host = torch.randn(1, depth, S_TOK, C_M)
        print(f"\ndepth={depth}  (tile pad on the contracted axis: {pad} rows)")

        r1 = run(opm, x_host, dev)
        churn(dev)
        r2 = run(opm, x_host, dev)

        if torch.equal(r1, r2):
            print("  same input, dirtied allocator in between   STABLE")
        else:
            d = (r1.float() - r2.float()).abs()
            print(f"  same input, dirtied allocator in between   UNSTABLE  "
                  f"maxabs {d.max():.3e}  frac {(d > 0).float().mean():.4f}"
                  f"   <-- padding contamination")
    print("done")


if __name__ == "__main__":
    main()
