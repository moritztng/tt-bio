"""Write the c730 fold (tt-bio's spd-data input, its own MSA) in each kit's input format, so the GPU folds what the TT
bench folds: chain A 580 + chain B 150, unpaired + paired rows from ~/spd-data/msa (MANIFEST.tsv row c730).

    python perf/kitcmp/make_inputs.py OUTDIR [--data ~/spd-data] [--root /root/kc/in]

OUTDIR gets msa/ (a3m per chain, unpaired and paired, plus Boltz-2 csv) and boltz2.yaml, opendde.json, openfold3.json,
each pointing at <root>/msa/... (the path the box sees). Seeds and rep count are the arms script's, not the input's.
"""
import argparse, ast, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--data", type=Path, default=Path("~/spd-data").expanduser())
ap.add_argument("--root", default="/root/kc/in")
a = ap.parse_args()

row = next(l.split("\t") for l in (a.data / "MANIFEST.tsv").read_text().splitlines() if l.startswith("c730\t"))
rows = ast.literal_eval(row[4].split("=", 1)[1])
unp = [k for k in rows if "/" not in k]                        # chain order = order in the manifest (A, B)
par = [k for k in rows if "/" in k]
seqs = []
for l in (a.data / "inputs/c730.yaml").read_text().splitlines():
    if l.strip().startswith("sequence:"):
        seqs.append(l.split(":", 1)[1].strip())
ids = ["A", "B"]
m = a.out / "msa"; m.mkdir(parents=True, exist_ok=True)


def a3m(p):
    t = (a.data / "msa" / p).read_text().splitlines()
    return [(t[i], t[i + 1]) for i in range(0, len(t) - 1, 2) if t[i].startswith(">")]


for c, s, u, p in zip(ids, seqs, unp, par):
    U, P = a3m(u), a3m(p)
    assert U[0][1].replace("-", "") == s and P[0][1].replace("-", "") == s, f"chain {c}: query row is not the sequence"
    shutil.copy(a.data / "msa" / u, m / f"{c}.unpaired.a3m")
    shutil.copy(a.data / "msa" / p, m / f"{c}.paired.a3m")
    # Boltz-2 csv: rows sharing a key are paired across chains, key -1 is unpaired (boltz.data.parse.csv).
    lines = ["key,sequence"] + [f"{i},{q}" for i, (_, q) in enumerate(P)]
    lines += [f"-1,{q}" for _, q in U[1:]]
    (m / f"{c}.csv").write_text("\n".join(lines) + "\n")
    print(c, len(s), "aa, unpaired", len(U), "paired", len(P))

R = a.root
(a.out / "boltz2.yaml").write_text("version: 1\nsequences:\n" + "".join(
    f"- protein:\n    id: {c}\n    sequence: {s}\n    msa: {R}/msa/{c}.csv\n" for c, s in zip(ids, seqs)))
(a.out / "opendde.json").write_text(json.dumps([{"name": "c730", "sequences": [
    {"proteinChain": {"sequence": s, "count": 1, "pairedMsaPath": f"{R}/msa/{c}.paired.a3m",
                      "unpairedMsaPath": f"{R}/msa/{c}.unpaired.a3m"}} for c, s in zip(ids, seqs)]}], indent=1))
(a.out / "openfold3.json").write_text(json.dumps({"queries": {"c730": {"chains": [
    {"molecule_type": "protein", "chain_ids": [c], "sequence": s,
     "main_msa_file_paths": [f"{R}/msa/{c}.unpaired.a3m"], "paired_msa_file_paths": [f"{R}/msa/{c}.paired.a3m"]}
    for c, s in zip(ids, seqs)]}}}, indent=1))
print("wrote", a.out)
