#!/usr/bin/env python3
"""The term the arithmetic floor excluded: the host round trip, priced on BC2's own census.

ROUNDFLOOR.json measured the instrument's float64 ARITHMETIC and said so. `exact_training`'s
docstring names the other half -- "a host round trip per softmax and per layer norm, in the
forward and in the backward's recompute" -- and that half needs a card to measure directly. It
does not need a card to PRICE, because of3t-bwattrib already measured the rate on real hardware:

    391.4 GB over 178 s  ->  2.199 GB/s
    (cross-check on its byte model: 226,492,416 elements at 0.906 GB each way is 4 B/element,
     and of3t's exact softmax input is float32 like ours, so the two agree)

That rate is of3t-bwattrib's, measured on pc against a Blackhole card. BindCraft 2 runs on qb2
and qb2's PCIe was never measured, so this is the campaign's only measured rate and it is
borrowed -- stated, not hidden. Everything else here is BindCraft 2's own.

THE TWO OPS DO NOT MOVE THE SAME WIDTH, and an earlier revision of this file wrongly used 4
bytes for both, overstating the layer norm's share by 2x:

  softmax    `_fp32_softmax_tail` typecasts the scores to float32 before the softmax
             (`tenstorrent.py:4296` and `:4299`), and `_exact_softmax_raw` returns
             `from_torch(y64.float(), dtype=v.dtype)`, so it is **float32, 4 B/element each way**
  layer_norm runs on AF2's activation tracks, which are built bfloat16
             (`af2.py:180-181`, `from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16)`; the
             float32 at `af2.py:315-324` is a residual-add temporary, not the track), so it is
             **bfloat16, 2 B/element each way**

Per-call traffic, stated before the number:
  forward    x off the card + y back                                 = 2 widths
  backward   g off + x re-read off + dx back                         = 3 widths (HIGH)
             `_v_exact_layer_norm.bw` does exactly that; the softmax backward holds y64 on the
             host from the forward and reads only g, so 2 widths is its LOW end
"""
import json

PCIE_GBPS = 391.4 / 178.0
BLOCKS, FWD_CENSUSES, BWD_CENSUSES = 48, 3, 1
ROUND_S = 39.08
ARITH_PC = {224: (80.1, 82.3), 288: (158.6, 160.8)}
BYTES = {"softmax": 4, "layer_norm": 2}          # per element, per direction


def census(n):
    """ladder.py's validated reduction, split by op because the widths differ."""
    return {"softmax": 2 * n * 4 * n * n, "layer_norm": 8 * n * n * 128}


def main():
    rows = []
    for n, (alo, ahi) in ARITH_PC.items():
        c = census(n)
        gb_lo = gb_hi = 0.0
        per_op = {}
        for op, el_block in c.items():
            w = BYTES[op]
            fwd_el = BLOCKS * FWD_CENSUSES * el_block
            bwd_el = BLOCKS * BWD_CENSUSES * el_block
            lo = (fwd_el * 2 * w + bwd_el * 2 * w) / 1e9
            hi = (fwd_el * 2 * w + bwd_el * 3 * w) / 1e9
            gb_lo += lo; gb_hi += hi
            per_op[op] = {"bytes_per_element_per_direction": w,
                          "forward_elements": fwd_el, "backward_elements": bwd_el,
                          "GB_per_round": [round(lo, 1), round(hi, 1)],
                          "s_per_round": [round(lo / PCIE_GBPS, 1), round(hi / PCIE_GBPS, 1)]}
        plo, phi = gb_lo / PCIE_GBPS, gb_hi / PCIE_GBPS
        tlo, thi = alo + plo, ahi + phi
        rows.append({"n": n, "per_op": per_op,
                     "pcie_GB_per_round": [round(gb_lo, 1), round(gb_hi, 1)],
                     "pcie_s_per_round": [round(plo, 1), round(phi, 1)],
                     "arithmetic_s_per_round_pc_quiet": [alo, ahi],
                     "added_s_per_round": [round(tlo, 1), round(thi, 1)],
                     "pcie_share_of_added": [round(plo / tlo, 3), round(phi / thi, 3)],
                     "round_s_with_instrument": [round(ROUND_S + tlo, 1),
                                                 round(ROUND_S + thi, 1)],
                     "slowdown_x": [round((ROUND_S + tlo) / ROUND_S, 2),
                                    round((ROUND_S + thi) / ROUND_S, 2)]})
    print(json.dumps({"pcie_GB_per_s_measured": round(PCIE_GBPS, 3),
                      "pcie_source": "of3t-bwattrib, 391.4 GB / 178 s, pc, Blackhole",
                      "bytes_per_element_per_direction": BYTES,
                      "round_s_measured_without_instrument": ROUND_S,
                      "round_source": "bcx-round, 25 rounds, n=224, AICLK 1350",
                      "call_structure_source": "STRUCTURE.json, 5 checks green",
                      "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
