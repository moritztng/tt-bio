#!/usr/bin/env python3
"""Fused-SDPA reachability and tile-alignment census, by calling the shipped arithmetic.

Two questions per token count, and neither needs a device:

  reach    does `_tri_att_sdpa_hifi` serve or decline? It declines when the tile-padded key length
           is not a multiple of the shipped k_chunk (`tenstorrent.py:1149-1153`), and it declines
           as too_short below `_TRIATT_FUSED_HIFI_MIN_S`.
  ragged   is the LOGICAL token count a multiple of 32? If not, `padded - tokens` key columns sit
           in the physical tile tail at a bias of 0, exp(0) = 1, and they take real softmax mass.
           Priced at 71-76x against fp64 in perf/fused_sdpa/errstruct_lenladder.json, saturating
           rather than scaling with the unmasked fraction.

A margin read at a token count that is DECLINED is an A/A from a lever that never ran; a margin read
at a RAGGED one is measuring the masking bug. Run this before folding anything.

    python3 perf/ob_fused_rescore/reach_census.py                    # the OB0 cells + the anchors
    python3 perf/ob_fused_rescore/reach_census.py 164 333 547        # arbitrary counts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# name -> token count. Token counts for the ob_* cells are MEASURED by perf/openbind/tt_ob_run.py
# (see state/openbind-perf.md section 4); the ligand cells are residues + atomised CCD atoms, which
# is why none of them is a multiple of 32.
CELLS = [
    ("ob_apo_128", 128), ("ob_apo_256", 256),
    ("ob_lig_s_298", 312), ("ob_lig_m_298", 333), ("ob_lig_l_298", 342),
    ("ob_apo_512", 512), ("ob_lig_m_512", 547),
    ("ob_apo_768", 768), ("ob_apo_1024", 1024),
    ("9bk6_164 (OF3 FLOOR anchor)", 164),
    ("cdk2x2_298", 298), ("cdk2x2_512", 512),
    ("openbind-ubq-msa", 76), ("openbind-fkg-ligand-msa", 140),
]


def row(name, L, T):
    pad = T._padded_sdpa_len(L)
    kc = T._sdpa_chunks_shipped(L, L)[1]
    if L < T._TRIATT_FUSED_HIFI_MIN_S:
        reach = "TOO_SHORT"
    else:
        reach = "SERVED" if pad % kc == 0 else "DECLINED"
    ragged = pad - L
    return (name, L, pad, L % 32, kc, reach, ragged, 100.0 * ragged / pad)


def main():
    from tt_bio import tenstorrent as T
    cells = ([(a, int(a)) for a in sys.argv[1:]] if len(sys.argv) > 1 else CELLS)
    print("%-30s %7s %7s %6s %8s %10s %7s %8s"
          % ("cell", "tokens", "padded", "tok%32", "k_chunk", "reach", "ragged", "share"))
    for name, L in cells:
        n, L, pad, m, kc, reach, rag, share = row(name, L, T)
        print("%-30s %7d %7d %6d %8d %10s %7d %7.1f%%" % (n, L, pad, m, kc, reach, rag, share))
    print("\nMIN_S %d  CHUNK_TILE %d  CHUNK_MAX %d"
          % (T._TRIATT_FUSED_HIFI_MIN_S, T.SDPA_CHUNK_TILE, T.SDPA_CHUNK_MAX))


if __name__ == "__main__":
    main()
