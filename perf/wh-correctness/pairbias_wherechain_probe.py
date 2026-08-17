"""Does opendde's device pair-init-bias where-chain survive a TILE reshape? In isolation, yes.

`6c3f5ecaf` moved `StructuralTokenExpander._pair_init_bias` on device: five where-chain row
selects producing a (1, clen*Ns, c_z) tensor, reshaped at the end to (clen, Ns, c_z). Ns is 249
for a 128 aa fold, so that reshape splits a TILE tensor's row axis on a non-multiple of 32. It
corrupted every OpenDDE fold from 2026-08-08 (25% of heavy atoms clashing at 128 aa) until the
index grid was uploaded already 3D.

This script is here so nobody re-runs it expecting a repro. Every isolated check below passes:
the where-chain is exact against the host gather with realistic fp32 tables, `ttnn.to_torch` of
the reshaped tensor is exact, and `ttnn.add` against a from_torch operand is exact. Reading a
device tensor back to host is what hides this class of defect, which is also why the expander's
host-side grading (state doc §25) declared the function clean while the fold was broken. The
A/B that settles it is a fold: swap the function's return for `self._up(ttnn.to_torch(out))`,
which changes no value, and the structure comes back clean.

    TT_VISIBLE_DEVICES=0 python3 perf/wh-correctness/pairbias_wherechain_probe.py
"""
import torch
import ttnn

C, Ns = 384, 249


def main():
    dev = ttnn.open_device(device_id=0)
    torch.manual_seed(0)
    for clen in (128, 121):          # the two chunks of a 128 aa fold: 128*249 % 32 == 0, 121*249 % 32 == 17
        for n in (2, 8):             # the 2-row masks and the 8-row role-pair-type table
            M = clen * Ns
            tab = torch.randn(n, C) * 0.05
            idx_h = torch.randint(0, n, (M,)).to(torch.int32)
            tab_d = ttnn.from_torch(tab, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
            idx_d = ttnn.from_torch(idx_h.reshape(1, -1, 1), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=ttnn.uint32)
            g = ttnn.reshape(ttnn.slice(tab_d, [n - 1, 0], [n, C]), (1, 1, C))
            for k in range(n - 2, -1, -1):
                rowk = ttnn.reshape(ttnn.slice(tab_d, [k, 0], [k + 1, C]), (1, 1, C))
                g = ttnn.where(ttnn.eq(idx_d, k), rowk, g)
            b = ttnn.reshape(ttnn.typecast(g, ttnn.bfloat16), (clen, Ns, C))
            want = tab[idx_h.long()].reshape(clen, Ns, C).to(torch.bfloat16).float()
            read = ttnn.to_torch(b).float()
            zeros = ttnn.from_torch(torch.zeros(clen, Ns, C), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=ttnn.bfloat16)
            added = ttnn.to_torch(ttnn.add(zeros, b)).float()
            print(f"clen={clen} n={n} M%32={M % 32} "
                  f"readback_exact={(read == want).float().mean():.4f} "
                  f"after_add_exact={(added == want).float().mean():.4f}")
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
