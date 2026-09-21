"""Build the customer-shaped fixture set: 580 aa target + binder, one yaml per token bucket.

The customer's target is redacted, so chain A is the first 580 residues of the 615 aa
protein already in examples/615.yaml, and each binder is a fragment of human CDK2 at the
length that lands in the bucket we want. Per-fold time is a function of the bucketed token
count and the MSA depth, not of which residues sit in the chain, so a real fragment at the
right length is a fair stand-in for a real binder at that length.
"""
import random, sys
from pathlib import Path
import yaml

sys.path.insert(0, ".")
from tt_bio.cache import seq_hash

ROOT = Path("perf/pvxrerun")
YAMLS = ROOT / "yaml"
MSA = ROOT / "msa"
DEPTH = 35

CDK2 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEF"
        "LHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVV"
        "TLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQ"
        "DFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")

def write_a3m(seq, rng):
    """A depth-DEPTH unpaired a3m for one chain. Rows are the query with a few point
    mutations; MSA cost depends on the tensor shape, not on which residues the rows carry."""
    aa = "ACDEFGHIKLMNPQRSTVWY"
    rows = [">query\n" + seq]
    for i in range(DEPTH - 1):
        s = list(seq)
        for _ in range(max(1, len(seq) // 12)):
            s[rng.randrange(len(s))] = aa[rng.randrange(20)]
        rows.append(">hit%d\n%s" % (i, "".join(s)))
    (MSA / (seq_hash(seq) + ".a3m")).write_text("\n".join(rows) + "\n")

def main():
    YAMLS.mkdir(parents=True, exist_ok=True)
    MSA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(101)
    target = yaml.safe_load(open("examples/615.yaml"))["sequences"][0]["protein"]["sequence"][:580]
    assert len(target) == 580, len(target)
    write_a3m(target, rng)
    for L in (78, 108, 140, 168):
        binder = CDK2[:L]
        write_a3m(binder, rng)
        tok = 580 + L
        bucket = -(-tok // 32) * 32
        doc = {"version": 1, "sequences": [
            {"protein": {"id": "A", "sequence": target}},
            {"protein": {"id": "B", "sequence": binder}},
        ]}
        name = "b%d_tok%d_bk%d" % (L, tok, bucket)
        (YAMLS / (name + ".yaml")).write_text(yaml.safe_dump(doc, sort_keys=False))
        print("%s: target 580 + binder %d = %d tokens -> bucket %d" % (name, L, tok, bucket))
    print("msa cache:", len(list(MSA.glob("*.a3m"))), "a3m at depth", DEPTH)

main()
