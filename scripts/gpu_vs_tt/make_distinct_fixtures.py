#!/usr/bin/env python3
"""Generate the distinct-targets control fixtures for the MPS same-input audit.

The control question: does the H200's MPS N=8 throughput hold when the 8 concurrent
workers fold 8 DIFFERENT targets instead of 8 copies of prot117? Eight single-chain
proteins of similar length (103-124 aa), each with a synthetic 35-row a3m so n_msa
matches every other concurrency point (35) and the harness's identical-bytes check
(query row == sequence) holds. Mutation-only rows (no indels), identity ~0.6-0.95,
seeded by target name so the fixtures are reproducible. Sequences 5CYT (leading
unresolved X dropped), 1BNI and 7RSA are from RCSB; the rest are in-repo examples.

    python3 scripts/gpu_vs_tt/make_distinct_fixtures.py
"""

import hashlib
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fixtures" / "distinct"

# name -> (sequence, source)
TARGETS = {
    "bk6_104": ("MSGGTPEERLAQLEKEIQALYDAADEVVDEVEEKDGKMTVTRTLTIGDGTVTLVETLKIVDGAPVKDG"
                "EIEVICNPECEELGKRLKALAKEYEKAQEEVEKAKA", "examples/9bk6.yaml chain A"),
    "cytc_104": ("GDVAKGKKTFVQKCAQCHTVENGGKHKVGPNLWGLFGRKTGQAEGYSYTDANKSKGIVWNNDTLME"
                 "YLENPKKYIPGTKMIFAGIKKKGERQDLVAYLKSATS",
                 "RCSB 5CYT chain 1, cytochrome c, unresolved leading X dropped"),
    "fkbp_107": ("GVQVETISPGDGRTFPKRGQTCVVHYTGMLEDGKKFDSSRDRNKPFKFMLGKQEVIRGWEEGVAQM"
                 "SVGQRAKLTISPDYAYGATGHPGIIPPHATLVFDVELLKLE",
                 "examples/affinity_fkg.yaml (FKBP12)"),
    "barn_110": ("AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGK"
                 "SGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR",
                 "RCSB 1BNI chain 1, barnase"),
    "mlta_112": ("MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQ"
                 "IESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA",
                 "examples/multimer.yaml chain A"),
    "mltb_116": ("MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAV"
                 "ESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVRDHRVSAQSA",
                 "examples/multimer.yaml chain B"),
    "prt7_117": ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAM"
                 "ISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG",
                 "examples/prot.yaml (the same-target point's own sequence)"),
    "rnsa_124": ("KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKN"
                 "GQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV",
                 "RCSB 7RSA chain 1, RNase A"),
}

AA = "ACDEFGHIKLMNPQRSTVWY"
# conservative-ish substitution groups (Dayhoff-flavoured); a mutated position draws
# from the same group with 70% probability, else any residue.
GROUPS = ["AGPST", "CV", "DENO", "FWYH", "KRH", "ILMV", "QEKRN"]


def mutate(seq: str, rng: random.Random, identity: float) -> str:
    out = []
    for c in seq:
        if rng.random() < identity:
            out.append(c)
            continue
        pool = AA
        for g in GROUPS:
            if c in g and rng.random() < 0.7:
                pool = g
                break
        choices = [x for x in pool if x != c]
        out.append(rng.choice(choices) if choices else c)
    return "".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (seq, src) in TARGETS.items():
        assert set(seq) <= set(AA), f"{name}: non-standard residue in {seq!r}"
        rng = random.Random(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
        rows = [f">{name} query ({src})", seq]
        for i in range(34):
            ident = 0.6 + 0.35 * rng.random()
            rows += [f">syn{i:02d} identity~{ident:.2f}", mutate(seq, rng, ident)]
        (OUT / f"{name}.a3m").write_text("\n".join(rows) + "\n")
        (OUT / f"{name}.seq").write_text(seq + "\n")
        print(f"{name}: {len(seq)} aa, 35 rows")


if __name__ == "__main__":
    main()
