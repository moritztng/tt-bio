#!/usr/bin/env python3
"""The widened dtype predicate must decline everything the four hand-written gates declined.

Host only, no device. The gates it replaced were `all operands == bfloat16`. This enumerates the
full operand-dtype cross product and checks three things by execution rather than by reading:

  1. no all-bf16 case is newly declined (the regression that would matter to a shipped fold),
  2. nothing outside FAST_DTYPES is admitted,
  3. every mixed set is declined -- the clause `probe2` measured as load bearing.
"""
import itertools, sys
import ttnn
from tt_bio.mm_generic import FAST_DTYPES, fast_dtypes_ok, tile_bytes

POOL = [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint32]
fails = []
n = 0
for arity in (3, 4):
    for combo in itertools.product(POOL, repeat=arity):
        n += 1
        got = fast_dtypes_ok(*combo)
        old = all(d == ttnn.bfloat16 for d in combo)
        uniform_fast = len(set(combo)) == 1 and set(combo) <= FAST_DTYPES
        if old and not got:
            fails.append(("regression: all-bf16 newly declined", combo))
        if got and not set(combo) <= FAST_DTYPES:
            fails.append(("admits a dtype outside FAST_DTYPES", combo))
        if got and len(set(combo)) != 1:
            fails.append(("admits a MIXED set (measured wrong at 12.55 rel_rms)", combo))
        if got != uniform_fast:
            fails.append(("disagrees with uniform-and-fast", combo))
print("combinations checked:", n)
print("tile_bytes bf16/fp32/bfp8:", tile_bytes(ttnn.bfloat16), tile_bytes(ttnn.float32),
      tile_bytes(ttnn.bfloat8_b))
print("admitted sets:", sorted({str(c[0]) for a in (3, 4)
                                for c in itertools.product(POOL, repeat=a) if fast_dtypes_ok(*c)}))
for why, combo in fails[:20]:
    print("FAIL", why, [str(d) for d in combo])
print("PASS" if not fails else "FAIL count=%d" % len(fails))
sys.exit(1 if fails else 0)
