#!/usr/bin/env python3
"""Independently re-derive TRIX's compulsory floor from the mathematics. CPU only, no device.

`trix-floor` (state/trix-floor.md, VERDICT: GO) replaced the campaign's 6.61 ms "arithmetic floor"
with 2.648 ms/call compulsory, built on 274.877 GFLOP and 1073.7 MB at 512 aa / c_z=256. Every score
in the campaign is now divided by that, so the orchestrator re-derives it here from the definition
of the operation rather than reproducing the row's own script. Same answer from a different route is
the check; a different answer is a finding.

Triangle multiplication, one call, batch 1, bf16, N tokens, c_z channels, hidden == c_z (forced by
the reference: p_in and g_in are dim -> 2*dim and the chunk halves that back to c_z).
"""

N, CZ, BYTES = 512, 256, 2          # 512 aa, c_z = 256, bf16
C = CZ                              # hidden == c_z


def gflop():
    """Compulsory multiply-accumulates x2, by leg."""
    legs = {
        # p_in and g_in: each [N,N,c_z] @ [c_z, 2*c_z]; the chunk halves 2*c_z back to c_z
        "in_proj (p and g)": 2 * (2 * N * N * CZ * (2 * CZ)),
        # out[i,j,c] = sum_k a[i,k,c] * b[j,k,c]  ->  N*N*N*c MACs
        "contraction":       2 * N * N * N * C,
        # [N,N,c] @ [c, c_z]
        "out_proj":          2 * N * N * C * CZ,
        # output gate, [N,N,c_z] @ [c_z, c_z]
        "out_gate":          2 * N * N * CZ * CZ,
    }
    return legs, sum(legs.values()) / 1e9


def z_bytes():
    """One pair tensor in bf16."""
    return N * N * CZ * BYTES


def main():
    legs, total = gflop()
    print(f"N={N}  c_z={CZ}  hidden={C}  dtype=bf16\n")
    print("COMPULSORY ARITHMETIC")
    for k, v in legs.items():
        print(f"  {k:22s} {v/1e9:8.2f} GFLOP")
    print(f"  {'TOTAL':22s} {total:8.3f} GFLOP     trix-floor: 274.877  ->  "
          f"{'MATCH' if abs(total - 274.877) < 0.01 else 'MISMATCH'}")

    Z = z_bytes()
    passes = 8                       # trix-floor's compulsory dataflow: 8 Z-sized passes
    traffic = passes * Z
    print(f"\nCOMPULSORY TRAFFIC")
    print(f"  Z (one pair tensor)    {Z/1e6:8.2f} MB")
    print(f"  {passes} Z-sized passes      {traffic/1e6:8.1f} MB     trix-floor: 1073.7  ->  "
          f"{'MATCH' if abs(traffic/1e6 - 1073.7) < 0.1 else 'MISMATCH'}")

    # Which side binds, at trix-floor's own measured roofs (pinned 1350 MHz, shipped kernel config)
    COMPUTE_ROOF, DRAM_COMBINED = 123.65e12, 410.3e9
    t_flops = total * 1e9 / COMPUTE_ROOF
    t_bytes = traffic / DRAM_COMBINED
    print(f"\nWHICH SIDE BINDS (roofs: {COMPUTE_ROOF/1e12} TFLOP/s, {DRAM_COMBINED/1e9} GB/s)")
    print(f"  arithmetic             {t_flops*1e3:8.3f} ms")
    print(f"  traffic                {t_bytes*1e3:8.3f} ms   <-- binds"
          if t_bytes > t_flops else "  traffic binds: no")
    print(f"  => floor is {'BYTE' if t_bytes > t_flops else 'COMPUTE'}-bound, "
          f"{max(t_flops, t_bytes)*1e3:.3f} ms at roof")
    print(f"  trix-floor quotes 2.648 ms, i.e. {traffic/2.648e-3/1e9:.1f} GB/s achieved = "
          f"{100*traffic/2.648e-3/DRAM_COMBINED:.1f} % of the combined roof")

    print(f"\nWHAT THE RETIRED 6.61 ms WAS")
    print(f"  6.61 / {max(t_flops, t_bytes)*1e3:.3f} = {6.61/(max(t_flops,t_bytes)*1e3):.2f}x too high")
    print(f"  trix-floor attributes that to pricing the arithmetic at our own kernels' rates "
          f"(2.16x)\n  and reading those rates at a governor-set clock (a further 1.16x): "
          f"2.16 * 1.16 = {2.16*1.16:.2f}x")


if __name__ == "__main__":
    main()
