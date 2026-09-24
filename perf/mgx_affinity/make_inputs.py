#!/usr/bin/env python3
"""Inputs for the affinity-at-scale row, committed so every surface scores the same files.

    python3 perf/mgx_affinity/make_inputs.py --davis davis.tsv

Three sets:

  size/    the fleet's CDK2 size fixture (298 aa, tiled and truncated to N, the construction
           perf/nesso1/inputs/ladder uses, so rung N here is rung N there) crossed with three
           ligands chosen for size: the ladder's own small drug (19 heavy atoms), sirolimus
           (CCD RAP, a 65-heavy-atom macrocycle) and cobalamin (CCD B12, 91 heavy atoms, the
           cofactor the combos row uses). CCD codes rather than SMILES for the two big ones, so
           the heavy-atom count is the dictionary's and not a string quoted from memory.
  screen/  one target against 100 ligands. The 68 DAVIS compounds (every drug DAVIS measured,
           with its Kd against this target) plus 30 constructed peptide-like ligands from
           perf/nesso1/inputs/screen, one cofactor-sized CCD entry (B12), and one SMILES that
           does not parse, so a screen shows what one bad entry does to the batch.
           Two targets: YSK4 (1328 aa, the largest DAVIS kinase with 50 non-censored Kd) for
           Nesso-1, and LCK (509 aa, 47 non-censored Kd, the 512-class production size) for
           both surfaces, since a Boltz-2 affinity job refolds the target per ligand.
  real/    3ABQ (ethanolamine ammonia-lyase a2b2, 1518 aa, mgx-reference's 1536 fixture) with
           cobalamin, a B12-dependent enzyme with its own cofactor class. The tiled CDK2 rungs
           have no real pocket once the target is three copies long; this one does.
  kd.json  measured Kd (nM) per screen record, 10000 = DAVIS's censoring value.

The DAVIS table is TDC's (dataverse datafile 5219748). Only the rows used are kept in kd.json.
"""
import argparse
import csv
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = re.search(r"sequence: (\S+)", (ROOT / "perf/nesso1/inputs/ladder/aa1536/cdk2_1536.yaml")
                 .read_text()).group(1)[:298]
SMALL = re.search(r"smiles: '([^']+)'", (ROOT / "perf/nesso1/inputs/ladder/aa1536/cdk2_1536.yaml")
                  .read_text()).group(1)
LIGANDS = {"small": ("smiles", SMALL), "rap": ("ccd", "RAP"), "b12": ("ccd", "B12")}
SIZES = {"small": (512, 1024, 1536, 1664, 1792, 2048, 2560, 3072, 3584, 4096), "rap": (512, 1536, 2048, 2560, 3072),
         "b12": (512, 1536, 2048, 2560, 3072)}
BAD_SMILES = "C1CC(N"  # unclosed ring and branch: RDKit returns None
TARGETS = ("YSK4", "LCK")


def yaml(seq, kind, value):
    v = f"'{value}'" if kind == "smiles" else value
    return (f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n"
            f"  - ligand:\n      id: B\n      {kind}: {v}\nproperties:\n  - affinity:\n"
            f"      binder: B\n")


def tiled(n):
    return (BASE * (n // len(BASE) + 1))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis", type=pathlib.Path, help="DAVIS tsv; without it only the size rungs")
    a = ap.parse_args()

    size = HERE / "inputs/size"
    size.mkdir(parents=True, exist_ok=True)
    for lig, ns in SIZES.items():
        for n in ns:
            (size / f"cdk2_{n}_{lig}.yaml").write_text(yaml(tiled(n), *LIGANDS[lig]))
    real = HERE / "inputs/real"
    real.mkdir(exist_ok=True)
    (real / "3abq_1536_b12.yaml").write_text(
        (ROOT / "perf/mgx/ref/fixtures/3abq_1536.yaml").read_text()
        + "  - ligand:\n      id: E\n      ccd: B12\nproperties:\n  - affinity:\n      binder: E\n")
    if a.davis is None:
        return

    rows = list(csv.DictReader(open(a.davis), delimiter="\t"))
    constructed = sorted((ROOT / "perf/nesso1/inputs/screen").glob("*.yaml"))[:30]
    kd = {}
    for t in TARGETS:
        mine = sorted((r for r in rows if r["ID2"] == t), key=lambda r: int(r["ID1"]))
        assert len(mine) == 68, (t, len(mine))
        seq = mine[0]["X2"]
        d = HERE / f"inputs/screen/{t.lower()}"
        d.mkdir(parents=True, exist_ok=True)
        kd[t] = {"seq_len": len(seq), "records": {}}
        for i, r in enumerate(mine):
            rid = f"d{i:02d}_{r['ID1']}"
            (d / f"{rid}.yaml").write_text(yaml(seq, "smiles", r["X1"]))
            kd[t]["records"][rid] = float(r["Y"])
        for i, p in enumerate(constructed):
            smi = re.search(r"smiles: '([^']+)'", p.read_text()).group(1)
            (d / f"c{i:02d}.yaml").write_text(yaml(seq, "smiles", smi))
        (d / "x_b12.yaml").write_text(yaml(seq, "ccd", "B12"))
        (d / "x_bad.yaml").write_text(yaml(seq, "smiles", BAD_SMILES))
        assert len(list(d.glob("*.yaml"))) == 100
    (HERE / "inputs/kd.json").write_text(json.dumps(kd, indent=1) + "\n")


if __name__ == "__main__":
    main()
