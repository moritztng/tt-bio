#!/usr/bin/env python3
"""The bfp8 arithmetic for the channel move, derived from the two measurement files.

Reads `byte_sensitivity_*.json` and `cast_cost_*.json` and prices the three routes the "accept
bfp8 here" ask can mean. It computes rather than asserts, so the state doc quotes numbers that
regenerate from the JSONs on demand.

The bfp8 move time is not measured -- writing a correct bfp8 kernel is the thing being decided --
so it is BOUNDED two ways and both are printed:

  IDEAL   the op is a byte pipe with a fixed overhead. Fit t = a + k*b to the fp32 control
          (t(2b)/t(b) = r measured) and evaluate at 0.53b. This is the most generous bound
          available: it grants bfp8 movement the same cost per byte as bf16, which the stock
          permute shows is false, and it ignores that the gather's transaction count doubles.
  BLOCKF  the same, times the block-float penalty the wheel's own permute pays at this shape
          (its measured bf16 -> bfp8 ratio). That penalty is an independent measurement of a real
          bfp8 move of this exact tensor on this card, so it is the honest read.

0.53 rather than 0.50 because a bfp8 tile is 1088 B against bf16's 2048: 1024 mantissa bytes plus
64 of shared exponent (`tt_bio/tenstorrent.py::_chunk_l1_per_core`). The exponent section is also
why the kernel cannot express bfp8 today -- see the state doc.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

BS, CC = (Path(sys.argv[1]), Path(sys.argv[2]))
bs, cc = json.loads(BS.read_text()), json.loads(CC.read_text())
BFP8_BYTE_RATIO = 1088 / 2048

casts = {(r["n"], r["c"], r["buf"]): r for r in cc["rows"] if "error" not in r}

print(f"grid {bs['grid']}  host {bs['host']}  git {bs['git'][:9]}")
print(f"bfp8/bf16 byte ratio {BFP8_BYTE_RATIO:.4f}\n")
hdr = (f"{'shape':22s} {'bf16 ms':>8s} {'fp32/bf16':>9s} {'a/kb':>6s} "
       f"{'IDEAL':>7s} {'BLOCKF':>7s} {'paidB':>7s} {'unpaidC':>8s}")
print(hdr); print("-" * len(hdr))

rows = []
for r in bs["rows"]:
    if "error" in r or not r.get("usable"):
        tag = f"{r['direction']} N={r.get('n')} C={r.get('c')}"
        print(f"{tag:22s}   ---- excluded ({'error' if 'error' in r else 'brackets disagree'})")
        continue
    tag = f"{r['direction']} N={r['n']} C={r['c']} {r['buf_out'][:4]}"
    t1, ratio = r["bf16_ctrl_ms"], r["fp32_over_bf16"]

    # t = a + k*b fitted to the doubling: (a + 2kb) / (a + kb) = ratio
    # -> a/(kb) = (2 - ratio) / (ratio - 1)
    if ratio <= 1.0:
        print(f"{tag:22s} {t1:8.3f} {ratio:9.3f}   fp32 not slower: no byte term to fit")
        continue
    a_over_kb = (2 - ratio) / (ratio - 1)
    ideal = (a_over_kb + BFP8_BYTE_RATIO) / (a_over_kb + 1)          # x the bf16 move
    blockf = ideal * r["stock_bfp8_over_bf16"]

    key = (r["n"], r["c"], r["buf_out"])
    cst = casts.get(key)
    paid = unpaid_i = unpaid_b = None
    if cst:
        cast_frac = (cst["cast_in_ms"] + cst["cast_out_ms"]) / cst["move_bf16_ms"]
        paid = cast_frac + blockf            # route B, block-float read
    print(f"{tag:22s} {t1:8.3f} {ratio:9.3f} {a_over_kb:6.2f} "
          f"{ideal:7.3f} {blockf:7.3f} {('%.3f' % paid) if paid else '    n/a':>7s} "
          f"{blockf:8.3f}")
    rows.append(dict(tag=tag, ideal=ideal, blockf=blockf, paid=paid))

print("""
IDEAL/BLOCKF/paidB/unpaidC are all MULTIPLES OF THE CURRENT bf16 MOVE. Below 1.000 is a win.
  unpaidC = BLOCKF: producer and consumer both already bfp8, so no cast is needed. That is not
            this gate, it is a pair-track dtype change, and it also costs accuracy.
  paidB   = cast round trip + BLOCKF: cast in, move, cast back, which is what "accept bfp8 at this
            gate" actually requires while every caller is bf16.""")

def rng(k):
    v = [r[k] for r in rows if r[k] is not None]
    return f"{min(v):.3f}-{max(v):.3f}x" if v else "n/a"

print(f"\nacross {len(rows)} usable shapes:  IDEAL {rng('ideal')}   "
      f"BLOCKF/unpaidC {rng('blockf')}   paidB {rng('paid')}")
