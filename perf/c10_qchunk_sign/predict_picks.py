"""CPU-only prediction of what the rule picks, replayed from the shipped source with no device.

Imported from tt_bio.tenstorrent so the arithmetic is the SHIPPED function, not a copy of it.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

CORES = 110          # p300c Blackhole 11x10, the grid this card reports
WORK = 16            # token DiT: batch 1 x 16 heads, from the b2z2 Blackhole census
SHIPPED_CAP = 256


def rows(cores=CORES):
    import tt_bio.tenstorrent as T
    out = []
    for size, q_len in [(512, 512), (298, 320)]:
        shipped = T._capped_sdpa_chunk_size(q_len)
        picked = T._grid_q_chunk(q_len, WORK, shipped, cores)
        padded = T._padded_sdpa_len(q_len)
        out.append(dict(size=size, q_len=q_len, padded=padded, work=WORK, cores=cores,
                        shipped_chunk=shipped, rule_chunk=picked,
                        shipped_units=WORK * -(-padded // shipped),
                        rule_units=WORK * (padded // picked),
                        moved=picked != shipped))
    # the atom site: one tile of queries, so the rule provably cannot move it
    for size, q_len, work in [(512, 32, 560), (298, 32, 326)]:
        shipped = T._capped_sdpa_chunk_size(q_len)
        out.append(dict(size=size, site='atom', q_len=q_len, work=work, cores=cores,
                        shipped_chunk=shipped, rule_chunk=T._grid_q_chunk(q_len, work, shipped, cores),
                        moved=T._grid_q_chunk(q_len, work, shipped, cores) != shipped))
    return out


if __name__ == '__main__':
    print(json.dumps(rows(), indent=2))
