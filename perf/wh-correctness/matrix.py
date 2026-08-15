#!/usr/bin/env python3
"""The JapanFold correctness matrix: every cell, its expected outcome, and a resumable runner.

One cell is one submission to the live public API with a stated expectation. A `reject`
cell passes only on a 400 that names the limit or the missing capability; an `ok` cell
passes only if the job succeeds AND every structure it returns survives
`check_structure.py`, including the composition check against the submitted input.

Composition fixtures are deliberately small (<= 80 residues). This axis asks whether a
model accepts a chemistry and folds it sanely, not how it scales; size lives on its own
axis and on its own fixtures.

    matrix.py --list                       # the cell table, with expectations
    matrix.py --run --group composition    # run one group, 3 at a time, resumable
    matrix.py --report                     # pass/fail per cell from the JSONL

Resume is by cell name against the JSONL, so a killed run costs at most the cells that
were in flight.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIX = HERE / "fixtures"

FOLD_MODELS = ["boltz2", "esmfold2", "esmfold2-fast", "protenix-v2", "opendde", "opendde-abag"]

# 64 aa, protein G B1 domain doubled to a length that folds fast and is not a stub.
P64 = ("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTEKPEVIDAS")
# A different 60 aa chain, for the heterodimer cell.
Q60 = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNI")

# The repo's own 1ahw antibody-antigen fixture (examples/1ahw_abag.yaml), comments and
# the version key stripped: the API takes the sequences block.
ABAG_1AHW = 'sequences:\n  - protein:\n      id: A\n      sequence: TNTVAAYNLTWKSTNFKTILEWEPKPVNQVYTVQISTKSGDWKSKCFYTTDTECDLTDEIVKDVKQTYLARVFSYPAGNEPLYENSPEFTPYLETNLGQPTIQSFEQVGTKVNVTVEDERTLVRRNNTFLSLRDVFGKDLIYTLYYWKSSSSGKKTAKTNTNEFLIDVDKGENYCFSVQAVIPSRTVNRKSTDSPVECMG\n  - protein:\n      id: H\n      sequence: EIQLQQSGAELVRPGALVKLSCKASGFNIKDYYMHWVKQRPEQGLEWIGLIDPENGNTIYDPKFQGKASITADTSSNTAYLQLSSLTSEDTAVYYCARDNSYYFDYWGQGTTLTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKI\n  - protein:\n      id: L\n      sequence: DIKMTQSPSSMYASLGERVTITCKASQDIRKYLNWYQQKPWKSPKTLIYYATSLADGVPSRFSGSGSGQDYSLTISSLESDDTATYYCLQHGESPYTFGGGTKLEINRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC\n'



def yaml_cell(body: str) -> str:
    return body


# name -> (yaml input, which caps the input needs). The cap names match
# tt_bio/platform/limits.py, which is what decides accept vs reject per model.
COMPOSITIONS = {
    "single":      (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n", set()),
    "homodimer":   (f"sequences:\n  - protein: {{id: [A, B], sequence: {P64}}}\n", {"multichain"}),
    "heterodimer": (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    f"  - protein: {{id: B, sequence: {Q60}}}\n", {"multichain"}),
    "ligand_smiles": (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                      "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n", {"ligands"}),
    "ligand_ccd":  (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    "  - ligand: {id: L, ccd: ATP}\n", {"ligands"}),
    "dna_duplex":  ("sequences:\n  - dna: {id: A, sequence: ATGCATGCATGCATGCATGC}\n"
                    "  - dna: {id: B, sequence: GCATGCATGCATGCATGCAT}\n", {"nucleic"}),
    "rna":         ("sequences:\n  - rna: {id: A, sequence: GGCUAGCUAGCUAGCUAGCC}\n", {"nucleic"}),
    "protein_dna": (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    "  - dna: {id: B, sequence: ATGCATGCATGCATGCATGC}\n"
                    "  - dna: {id: C, sequence: GCATGCATGCATGCATGCAT}\n", {"nucleic"}),
    "affinity":    (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n"
                    "properties:\n  - affinity: {binder: L}\n", {"affinity"}),
    "constraints": (f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
                    "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n"
                    "constraints:\n  - pocket: {binder: L, contacts: [[A, 10], [A, 14]]}\n",
                    {"constraints"}),
    # 1ahw: tissue-factor antigen + Fab 5G9 heavy/light, the repo's own antibody-antigen
    # fixture. It is the only composition here that is in OpenDDE's distribution, so it is
    # what separates "OpenDDE is broken" from "OpenDDE was handed a target it is not for".
    "abag":        (ABAG_1AHW, {"multichain"}),
    # The gap read out of limits.py: `modifications` is advertised for boltz2 and the two
    # esmfold2 ids only, and `_check_model_caps` has no branch for it, so a modified residue
    # sent to protenix-v2 or either opendde is neither served nor refused by contract.
    "modres":      (f"sequences:\n  - protein: {{id: A, sequence: {P64}, modifications: "
                    "[{position: 12, ccd: SEP}]}\n", {"modifications"}),
    # Position 11 is THR and TPO is phosphothreonine, so this is the modification a user
    # would actually ask for. It exists to remove the one objection to the `modres` cell,
    # whose L->SEP is a chemically odd substitution: if a model drops this one too, the
    # drop is about the model and not about the request.
    "modres_tpo":  (f"sequences:\n  - protein: {{id: A, sequence: {P64}, modifications: "
                    "[{position: 11, ccd: TPO}]}\n", {"modifications"}),
}

# What each model advertises, as of 2026-08-16 on the live catalog. Every expectation in
# the matrix is derived from this, and it is asserted against the live /v1/models on every
# run, so a catalog change breaks the matrix instead of silently re-baselining it.
MODEL_CAPS = {
    "boltz2":        {"msa", "ligands", "nucleic", "affinity", "constraints", "multichain",
                      "modifications", "potentials", "pae"},
    "esmfold2":      {"msa", "multichain", "modifications"},
    "esmfold2-fast": {"multichain", "modifications"},
    "protenix-v2":   {"msa", "ligands", "nucleic", "multichain", "pae"},
    "opendde":       {"msa", "multichain"},
    "opendde-abag":  {"msa", "multichain"},
}
MAX_RESIDUES = {"boltz2": 1024, "esmfold2": 1024, "esmfold2-fast": 1024,
                "protenix-v2": 980, "opendde": 788, "opendde-abag": 779}


def assert_catalog_unchanged() -> None:
    """The matrix's expectations are only as true as the catalog they were read from."""
    r = subprocess.run(["curl", "-s", "-m", "30", "https://api.japanfold.com/v1/models"],
                       capture_output=True, text=True)
    live = json.loads(r.stdout)
    got = {m["id"]: set(m.get("caps", [])) for m in live["models"]}
    if got != MODEL_CAPS:
        raise SystemExit(f"catalog moved: live {got} != recorded {MODEL_CAPS}")
    gotmax = {m["id"]: m["max_residues"] for m in live["models"]}
    if gotmax != MAX_RESIDUES:
        raise SystemExit(f"max_residues moved: live {gotmax} != recorded {MAX_RESIDUES}")

# Malformed and hostile inputs. Each carries what the cell is actually asking.
HOSTILE = {
    "bad_constraint_chain": (
        f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
        "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n"
        "constraints:\n  - pocket: {binder: L, contacts: [[Z, 10]]}\n",
        "constraint names chain Z, which does not exist"),
    "constraint_oob_index": (
        f"sequences:\n  - protein: {{id: A, sequence: {P64}}}\n"
        "  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n"
        "constraints:\n  - pocket: {binder: L, contacts: [[A, 9999]]}\n",
        "residue index past the end of the chain"),
    "ligand_only": ("sequences:\n  - ligand: {id: L, smiles: 'CC(=O)Oc1ccccc1C(=O)O'}\n",
                    "no polymer at all"),
    "one_residue": ("sequences:\n  - protein: {id: A, sequence: M}\n",
                    "a chain too short to have a backbone"),
    "ambiguous_letters": ("sequences:\n  - protein: {id: A, sequence: "
                          "MBJOUXZMBJOUXZMBJOUXZMBJOUXZMBJOUXZMBJOUXZ}\n",
                          "B/J/O/U/X/Z must map to UNK and keep the length"),
    "lowercase": (f"sequences:\n  - protein: {{id: A, sequence: {P64.lower()}}}\n",
                  "must not be read as masked or gapped"),
    "all_x": ("sequences:\n  - protein: {id: A, sequence: " + "X" * 256 + "}\n",
              "the garbage canary: a constant confidence field must be caught"),
}


def cells(group: str) -> list[dict]:
    out = []
    if group in ("composition", "all"):
        for cname, (yml, need) in COMPOSITIONS.items():
            for model in FOLD_MODELS:
                served = need <= MODEL_CAPS[model]
                # `modifications` has no branch in `_check_model_caps`, so a modified residue
                # sent to a model that does not advertise it is neither served by contract nor
                # refused. Measured 2026-08-16: protenix-v2, opendde and opendde-abag all
                # return 202. The contract is undefined, so the cell observes and does not grade.
                expect = "ok" if served else "reject"
                if cname.startswith("modres") and not served:
                    expect = "unknown"
                out.append({"cell": f"comp_{cname}_{model}", "kind": "predict",
                            "expect": expect,
                            "payload": {"model": model, "name": cname, "input": yml},
                            "yaml": yml, "group": "composition"})
    if group in ("hostile", "all"):
        for hname, (yml, why) in HOSTILE.items():
            out.append({"cell": f"hostile_{hname}", "kind": "predict", "expect": "unknown",
                        "payload": {"model": "boltz2", "name": hname, "input": yml},
                        "yaml": yml, "why": why, "group": "hostile"})
    return out


def run_cell(c: dict, outjs: Path, deadline: int) -> int:
    d = RESULTS / "payloads"
    d.mkdir(parents=True, exist_ok=True)
    pf = d / f"{c['cell']}.json"
    pf.write_text(json.dumps(c["payload"]))
    inf = d / f"{c['cell']}.input.yaml"
    inf.write_text(c["yaml"])
    # "unknown" cells are recorded, not graded: the point is to observe what the service
    # does with an input nobody has decided the contract for yet.
    expect = "ok" if c["expect"] == "unknown" else c["expect"]
    cmd = [sys.executable, str(HERE / "jf_cell.py"), "--cell", c["cell"], "--kind", c["kind"],
           "--expect", expect, "--payload", str(pf), "--input", str(inf),
           "--out", str(outjs), "--artifacts", str(RESULTS / "artifacts"),
           "--deadline", str(deadline)]
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--markdown", action="store_true",
                    help="the composition x model table, generated from the JSONL so the "
                         "state doc is never hand-transcribed")
    ap.add_argument("--group", default="all")
    ap.add_argument("--only", help="comma-separated cell-name substrings")
    ap.add_argument("--expect-only", choices=("ok", "reject", "unknown"),
                    help="run one expectation class; reject cells are instant, so they buy "
                         "most of the matrix before a single fold is spent")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="the per-IP cap is 8 active jobs; stay well under it")
    ap.add_argument("--deadline", type=int, default=1800)
    ap.add_argument("--out", type=Path, default=RESULTS / "matrix.jsonl")
    a = ap.parse_args()

    todo = cells(a.group)
    if a.only:
        keys = [k.strip() for k in a.only.split(",")]
        todo = [c for c in todo if any(k in c["cell"] for k in keys)]
    if a.expect_only:
        todo = [c for c in todo if c["expect"] == a.expect_only]

    if a.list:
        for c in todo:
            print(f"{c['cell']:44s} {c['expect']}")
        print(f"{len(todo)} cells")
        return 0

    if a.markdown:
        rows = {}
        if a.out.exists():
            for line in a.out.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["cell"]] = r
        # A cell is one of: PASS, FAIL (with the reason), 400 (refused as the catalog says),
        # or "-" for not yet run. The reason column is the point, so it goes underneath.
        print("| input | " + " | ".join(FOLD_MODELS) + " |")
        print("|---" * (len(FOLD_MODELS) + 1) + "|")
        notes = []
        for cname in COMPOSITIONS:
            cells_out = []
            for model in FOLD_MODELS:
                r = rows.get(f"comp_{cname}_{model}")
                if r is None:
                    cells_out.append("-")
                elif r.get("submit_status") == 400:
                    cells_out.append("400" if r.get("pass") else "**400 unexpected**")
                elif r.get("pass"):
                    cells_out.append("fold")
                else:
                    cells_out.append("**FAIL**")
                    what = r.get("status") or f"submit {r.get('submit_status')}"
                    notes.append(f"- `comp_{cname}_{model}`: {what} -- {r.get('why', '')[:220]}")
            print(f"| {cname} | " + " | ".join(cells_out) + " |")
        if notes:
            print("\nFailures:\n")
            print("\n".join(notes))
        return 0

    if a.report:
        rows = {}
        if a.out.exists():
            for line in a.out.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["cell"]] = r
        for c in todo:
            r = rows.get(c["cell"])
            v = "-" if r is None else ("PASS" if r.get("pass") else "FAIL")
            print(f"{c['cell']:44s} {c['expect']:8s} {v:5s} {r.get('why','') if r else ''}"[:160])
        print(f"{sum(1 for c in todo if c['cell'] in rows)}/{len(todo)} run")
        return 0

    assert_catalog_unchanged()
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["cell"])
    todo = [c for c in todo if c["cell"] not in done]
    print(f"{len(todo)} cells to run ({len(done)} already recorded)", flush=True)
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(lambda c: run_cell(c, a.out, a.deadline), todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
