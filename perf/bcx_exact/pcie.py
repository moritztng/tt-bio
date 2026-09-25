#!/usr/bin/env python3
"""The term the arithmetic floor excluded: the host round trip, priced on BC2's own census.

ROUNDFLOOR.json measured the instrument's float64 ARITHMETIC and said so. The docstring of
`exact_training` names the other half -- "a host round trip per softmax and per layer norm, in
the forward and in the backward's recompute" -- and that half needs a card to measure directly.
It does not need a card to PRICE, because of3t-bwattrib already measured the only two numbers it
takes, on real hardware:

    226,492,416 elements per exact softmax, 0.906 GB each way, 1.81 GB round trip
      -> 4 bytes per element each way, 8 bytes per element round trip
    391.4 GB over 216 calls, 178 s  ->  2.199 GB/s measured PCIe

Both are of3t-bwattrib's, measured on pc against a Blackhole card. BindCraft 2 runs on qb2 and
qb2's PCIe was not measured, so this is the campaign's only measured rate and it is borrowed,
which is stated rather than hidden. The ELEMENT counts are BindCraft 2's own, from the runtime
census in BLOCKCOUNT.json reduced by ladder.py and validated against it to 0.7 %.

Byte model, stated before the number:
  forward, per exact call   x off the card (4 B/elem) + y back (4 B/elem)          =  8 B/elem
  backward, per exact call  g off + x re-read off + dx back, 4 B/elem each         = 12 B/elem
`_v_exact_layer_norm.bw` does `ttnn.to_torch(g)` and `ttnn.to_torch(x.value)` and puts one dx
back, which is where 12 comes from; the softmax backward reads g and holds y64 on the host from
the forward, so 12 is the conservative (larger) end for it and 8 the floor.
"""
import json, sys

BYTES_PER_ELEM_FWD = 8
BYTES_PER_ELEM_BWD_LOW, BYTES_PER_ELEM_BWD_HIGH = 8, 12
PCIE_GBPS = 391.4 / 178.0          # 2.199, of3t-bwattrib measured
BLOCKS, FWD_CENSUSES, BWD_CENSUSES = 48, 3, 1
ROUND_S = 39.08                    # bcx-round, 25 rounds, n=224, AICLK 1350
# ROUNDFLOOR.json on pc, quiet, three runs: the arithmetic-only floor per round.
ARITH_PC = {224: (80.1, 82.3), 288: (158.6, 160.8)}


def census(n):
    """ladder.py's validated reduction: 2 x (n,4,n,n) softmax + 8 x (n,n,128) layer_norm."""
    return 2 * n * 4 * n * n + 8 * n * n * 128


def main():
    rows = []
    for n, (alo, ahi) in ARITH_PC.items():
        per_block = census(n)
        fwd_el = BLOCKS * FWD_CENSUSES * per_block
        bwd_el = BLOCKS * BWD_CENSUSES * per_block
        gb_lo = (fwd_el * BYTES_PER_ELEM_FWD + bwd_el * BYTES_PER_ELEM_BWD_LOW) / 1e9
        gb_hi = (fwd_el * BYTES_PER_ELEM_FWD + bwd_el * BYTES_PER_ELEM_BWD_HIGH) / 1e9
        pcie_lo, pcie_hi = gb_lo / PCIE_GBPS, gb_hi / PCIE_GBPS
        tot_lo, tot_hi = alo + pcie_lo, ahi + pcie_hi
        rows.append({
            "n": n, "elements_per_block_census": per_block,
            "forward_elements_per_round": fwd_el, "backward_elements_per_round": bwd_el,
            "pcie_GB_per_round": [round(gb_lo, 1), round(gb_hi, 1)],
            "pcie_s_per_round": [round(pcie_lo, 1), round(pcie_hi, 1)],
            "arithmetic_s_per_round_pc_quiet": [alo, ahi],
            "added_s_per_round": [round(tot_lo, 1), round(tot_hi, 1)],
            "round_s_with_instrument": [round(ROUND_S + tot_lo, 1), round(ROUND_S + tot_hi, 1)],
            "slowdown_x": [round((ROUND_S + tot_lo) / ROUND_S, 2),
                           round((ROUND_S + tot_hi) / ROUND_S, 2)]})
    out = {"pcie_GB_per_s_measured": round(PCIE_GBPS, 3),
           "pcie_source": "of3t-bwattrib, 391.4 GB / 178 s, pc, Blackhole",
           "bytes_per_element": {"forward": BYTES_PER_ELEM_FWD,
                                 "backward": [BYTES_PER_ELEM_BWD_LOW, BYTES_PER_ELEM_BWD_HIGH]},
           "round_s_measured_without_instrument": ROUND_S,
           "round_source": "bcx-round, 25 rounds, n=224, AICLK 1350, loadavg 37-52",
           "rows": rows}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
