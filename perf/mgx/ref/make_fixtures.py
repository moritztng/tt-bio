#!/usr/bin/env python3
"""Build the shared MGX reference fixtures at 1024, 1280 and 1536 tokens.

Two kinds per rung:

  * a real multi-chain complex from the PDB, protein only, every chain standard residues, so a
    structural RMSD between two folds means something;
  * the tiled cdk2x2 fixture the size ladder and perf/whceil already fold, so their outputs can
    be scored against the same reference.

The alignment is part of the input. Every chain gets ONE unpaired a3m, searched once through
the ColabFold API with the same call tt-bio makes (`tt_bio.data.msa.run_mmseqs2`, use_env=True,
no pairing), and committed. The yaml names it with `msa:`, and an explicit `msa:` wins over any
search in every predict model, so the TT fold and the GPU fold read the same rows. No model gets
cross-chain pairing: pairing is done differently by each upstream, and a reference that pairs on
one side only would score the pairing, not the model.

Run from the repo root:  python3 perf/mgx/ref/make_fixtures.py
Re-running re-uses committed a3m files; delete one to re-search it.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIX = HERE / "fixtures"
MSA = FIX / "msa"
GT = FIX / "gt"

# rung -> PDB entry. Picked from an RCSB search for X-ray <= 2.2 A, protein only, >= 2 distinct
# protein entities, at most 4 chains, total SEQRES length just under the rung (bucketing to 32
# lands each on its rung), and no selenocysteine or other non-standard residue in any chain.
COMPLEXES = {
    1024: ("7AQX", "VSG2 homodimer with two nanobodies (NB9), T. brucei / llama"),
    1280: ("2AD6", "methanol dehydrogenase a2b2, M. methylotrophus"),
    1536: ("3ABQ", "ethanolamine ammonia-lyase a2b2, E. coli"),
}
CDK2X2 = ROOT / "perf" / "size512" / "fixtures"
STD = set("ACDEFGHIKLMNPQRSTVWY")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def entities(pdb: str) -> list[tuple[list[str], str]]:
    """[(auth chain ids, sequence)] from the RCSB FASTA, which carries SEQRES, not the model."""
    lines = fetch(f"https://www.rcsb.org/fasta/entry/{pdb}").decode().split("\n")
    out = []
    for h, s in zip(lines[0::2], lines[1::2]):
        if not h.startswith(">"):
            continue
        chains = h.split("|")[1].replace("Chains ", "").replace("Chain ", "")
        ids = [c.strip().split("[")[0].strip() for c in chains.split(",")]
        bad = set(s) - STD
        if bad:
            sys.exit(f"{pdb} {ids}: non-standard residues {bad}")
        out.append((ids, s))
    return sorted(out, key=lambda e: e[0])


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def search(seqs: dict[str, str]) -> None:
    """One unpaired ColabFold search for every a3m not already committed."""
    todo = {k: s for k, s in seqs.items() if not (MSA / f"{k}.a3m").is_file()}
    if not todo:
        return
    sys.path.insert(0, str(ROOT))
    from tt_bio.data.msa import run_mmseqs2

    res = run_mmseqs2(list(todo.values()), str(HERE / ".msa_tmp"), use_env=True,
                      use_pairing=False)
    for (k, s), a3m in zip(todo.items(), res):
        rows = a3m.strip().split("\n")
        assert rows[1] == s, f"{k}: query row is not the sequence"
        (MSA / f"{k}.a3m").write_text(a3m if a3m.endswith("\n") else a3m + "\n")


def yaml_text(title: str, chains: list[tuple[list[str], str, str]]) -> str:
    out = ["version: 1", f"# {title}", "sequences:"]
    for ids, seq, msa in chains:
        idv = ids[0] if len(ids) == 1 else "[" + ", ".join(ids) + "]"
        out += ["  - protein:", f"      id: {idv}", f"      sequence: {seq}", f"      msa: {msa}"]
    return "\n".join(out) + "\n"


def match_rows(a3m: str, length: int) -> tuple[str, int]:
    """Keep the a3m records whose aligned row has exactly `length` match columns (insertions are
    lowercase or '.'), return the kept text and how many were dropped. Every upstream parser
    rejects a ragged a3m outright, so a malformed row cannot silently reach one side only."""
    recs = re.findall(r"(>[^\n]*\n)([^>]*)", a3m)
    keep = [h + s for h, s in recs if len(re.sub(r"[a-z.\s]", "", s)) == length]
    return "".join(keep), len(recs) - len(keep)


def main() -> None:
    MSA.mkdir(parents=True, exist_ok=True)
    GT.mkdir(parents=True, exist_ok=True)
    manifest = {"note": "paths are relative to the tt-bio repo root; run folds from there",
                "msa_source": "ColabFold API via tt_bio.data.msa.run_mmseqs2, use_env=True, "
                              "use_pairing=False; one unpaired a3m per chain, no pairing",
                "fixtures": {}}
    plan = {}
    for rung, (pdb, title) in COMPLEXES.items():
        ents = entities(pdb)
        plan[rung] = (pdb, title, ents)
    search({f"{pdb.lower()}_{ids[0]}": s for pdb, _, ents in plan.values() for ids, s in ents})

    for rung, (pdb, title, ents) in plan.items():
        name = f"{pdb.lower()}_{rung}"
        chains = []
        for ids, s in ents:
            p = MSA / f"{pdb.lower()}_{ids[0]}.a3m"
            assert match_rows(p.read_text(), len(s))[1] == 0, f"{p} has rows off the query length"
            chains.append((ids, s, str(p.relative_to(ROOT))))
        tokens = sum(len(s) * len(ids) for ids, s in ents)
        (FIX / f"{name}.yaml").write_text(yaml_text(f"PDB {pdb}: {title}. {tokens} tokens.", chains))
        gt = GT / f"{pdb.lower()}.cif.gz"
        if not gt.is_file():
            gt.write_bytes(fetch(f"https://files.rcsb.org/download/{pdb}.cif.gz"))
        manifest["fixtures"][name] = dict(
            rung=rung, kind="pdb-complex", pdb=pdb, tokens=tokens,
            yaml=str((FIX / f"{name}.yaml").relative_to(ROOT)), ground_truth=str(gt.relative_to(ROOT)),
            chains=[dict(ids=ids, length=len(s), msa=m, msa_rows=Path(ROOT / m).read_text().count(">"),
                         msa_sha256_16=sha(ROOT / m)) for ids, s, m in chains])

    for rung in sorted(COMPLEXES):
        src = CDK2X2 / f"cdk2x2_{rung}.yaml"
        a3m = CDK2X2 / f"cdk2x2_{rung}.a3m"
        seq = next(l.split(":", 1)[1].strip() for l in src.read_text().split("\n")
                   if l.strip().startswith("sequence:"))
        assert a3m.read_text().split("\n")[1] == seq, f"{a3m} query row does not match {src}"
        name = f"cdk2x2_{rung}"
        # The ladder's 1280 a3m carries rows short of 1280 match columns; pin a copy without them.
        kept, dropped = match_rows(a3m.read_text(), len(seq))
        note = "with its a3m pinned"
        if dropped:
            a3m = MSA / f"{name}.a3m"
            a3m.write_text(kept)
            note = f"with its a3m pinned minus {dropped} rows whose match columns are not {len(seq)}"
        m = str(a3m.relative_to(ROOT))
        (FIX / f"{name}.yaml").write_text(yaml_text(
            f"CDK2 (1HCL) tiled to {len(seq)} aa, one chain: the size ladder's own fixture "
            f"({src.relative_to(ROOT)}) {note}.", [(["A"], seq, m)]))
        manifest["fixtures"][name] = dict(
            rung=rung, kind="tiled-cdk2x2", tokens=len(seq),
            yaml=str((FIX / f"{name}.yaml").relative_to(ROOT)), ground_truth=None,
            chains=[dict(ids=["A"], length=len(seq), msa=m, msa_rows=a3m.read_text().count(">"),
                         msa_rows_dropped=dropped, msa_sha256_16=sha(a3m))])

    (FIX / "fixtures.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for n, f in manifest["fixtures"].items():
        print(n, f["tokens"], [(c["ids"], c["length"], c["msa_rows"]) for c in f["chains"]])


if __name__ == "__main__":
    main()
