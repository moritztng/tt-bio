"""K4 must change exactly two padded sizes and nothing else. Enumerate rather than assert."""
import importlib
import os
import sys
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree))

SEQS = list(range(32, 2049, 32)) + [1, 100, 257, 298, 321, 383, 385, 511, 513, 639, 641]
SEQS = sorted(set(SEQS))


def picks(band_div_k):
    os.environ["TT_BIO_SDPA_BAND_DIV_K"] = band_div_k
    os.environ["TT_BIO_SDPA_DIV_K"] = "1"
    for m in [m for m in list(sys.modules) if m.startswith("tt_bio")]:
        del sys.modules[m]
    T = importlib.import_module("tt_bio.tenstorrent")
    assert Path(T.__file__).resolve().is_relative_to(tree), T.__file__
    return {s: (T._sdpa_chunks_shipped(s, s), T._padded_sdpa_len(s)) for s in SEQS}


off, on = picks("0"), picks("1")

changed = [s for s in SEQS if off[s][0] != on[s][0]]
print(f"sizes probed: {len(SEQS)}  changed by K4: {len(changed)}")
for s in changed:
    padded = off[s][1]
    (q0, k0), (q1, k1) = off[s][0], on[s][0]
    print(f"  seq {s:5d} padded {padded:5d}  k {k0:4d} -> {k1:4d}   "
          f"q {q0} -> {q1}   old divides: {padded % k0 == 0}   new divides: {padded % k1 == 0}")

bad_q = [s for s in changed if off[s][0][0] != on[s][0][0]]
bad_div = [s for s in changed if on[s][1] % on[s][0][1] != 0]
print(f"\nq changed anywhere (must be empty): {bad_q}")
print(f"new k does not divide padded (must be empty): {bad_div}")
print(f"OK: {not bad_q and not bad_div and {off[s][1] for s in changed} == {320, 384}}")
