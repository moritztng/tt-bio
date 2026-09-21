"""Depth arms for MSA-TERM: the same 672-bucket target at the rungs a real ColabFold
alignment on a 580 aa target actually lands on.

Depth is charged in rungs (tt_bio/token_axis.py MSA_PAD_LADDER, then multiples of 1024), so
3553 rows pads to 4096 and 9947 pads to 10240. Nobody has measured past 2048. Rows are the
query with point mutations: MSA cost is set by the tensor shape, not by the residues in it.
"""
import random, sys
from pathlib import Path
import yaml

sys.path.insert(0, ".")
from tt_bio.cache import seq_hash

ROOT = Path("perf/pvxrerun")
DEPTHS = {"d2048": 2048, "d3553": 3553, "d9947": 9947}

def write_a3m(seq, depth, out, rng):
    aa = "ACDEFGHIKLMNPQRSTVWY"
    rows = [">query\n" + seq]
    for i in range(depth - 1):
        s = list(seq)
        for _ in range(max(1, len(seq) // 12)):
            s[rng.randrange(len(s))] = aa[rng.randrange(20)]
        rows.append(">hit%d\n%s" % (i, "".join(s)))
    out.write_text("\n".join(rows) + "\n")

def main():
    rng = random.Random(101)
    src = yaml.safe_load(open(ROOT / "yaml" / "b78_tok658_bk672.yaml"))
    target = src["sequences"][0]["protein"]["sequence"]
    binder = src["sequences"][1]["protein"]["sequence"]
    for tag, depth in DEPTHS.items():
        msa = ROOT / ("msa_" + tag)
        msa.mkdir(parents=True, exist_ok=True)
        # the binder keeps its shallow alignment; the 580 aa target carries the deep one,
        # and the fold is charged the deeper of the two (protenix_data.py:434)
        write_a3m(binder, 35, msa / (seq_hash(binder) + ".a3m"), rng)
        write_a3m(target, depth, msa / (seq_hash(target) + ".a3m"), rng)
        d = ROOT / ("in_" + tag)
        d.mkdir(parents=True, exist_ok=True)
        (d / ("01_b78_" + tag + ".yaml")).write_text(yaml.safe_dump(src, sort_keys=False))
        (d / ("02_b78_" + tag + ".yaml")).write_text(yaml.safe_dump(src, sort_keys=False))
        from tt_bio.token_axis import msa_pad_amount
        print("%s: target depth %d -> padded %d" % (tag, depth, depth + msa_pad_amount(depth)))

main()
