#!/usr/bin/env python3
"""AbAg-XM independent leak audit (closeout spec 1.4): homology leakage of the
164 targets against pre-cutoff data.

Date purity is already established (all targets 2026 releases; cutoffs
2021-09-30 OpenDDE / 2023-06-01 Boltz-2). This audit measures HOMOLOGY:

  Leg A (CDR-H3): MMseqs2 search of each target's CDR-H3
    (~/abag_xm/cdrh3/cdrh3_all.fasta) against CDR-H3s of SAbDab entries
    released before each cutoff. The SAbDab universe is ABAG-Rank's
    full_sabdab_train_manifest.csv complex_ids (their train set IS
    pre-cutoff SAbDab; the opig TSV endpoint is now a SAbDab2 SPA, pass-5
    finding). Heavy chains are ANARCI-typed (IMGT) from pdb_seqres chains;
    CDR-H3 = IMGT 105-117.
  Leg B (antigen): MMseqs2 search of each target's antigen sequence (fold
    YAML chain A) against ALL PDB protein chains released pre-cutoff
    (pdb_seqres.txt + entries.idx dates).
  Ranker leg: assert none of the 164 pdb_ids is an ABAG-Rank train
    complex_id, and report antigen max-identity vs all protein chains of
    the train entries.

Flag rule (spec): contamination RISK iff CDR-H3 max-identity >= 90% AND
antigen max-identity >= 90% against the SAME pre-cutoff entry; marginals
reported for both cutoffs.

    python3 scripts/abag_xm_leak_audit.py            # full run (qb1)
    python3 scripts/abag_xm_leak_audit.py --limit_chains 2000   # smoke

Output: leak_audit.parquet (target, cdrh3_max_id_pre2021, ag_max_id_pre2021,
cdrh3_max_id_pre2023, ag_max_id_pre2023, abagrank_train_max_id, flag) + a
markdown summary. ANARCI results are cached in --work_dir (reruns cheap).
"""
import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from abag_xm_cdrh3_cluster import cdrh3_from_numbering  # noqa: E402
from abag_xm_hq_bracket import _yaml_seqs  # noqa: E402

CUTOFFS = {"pre2021": datetime.date(2021, 9, 30),
           "pre2023": datetime.date(2023, 6, 1)}
FLAG_IDENT = 90.0
AA = "ACDEFGHIKLMNPQRSTVWY"


def _parse_entries(path):
    """entries.idx -> {pdb_id: release_date} (col 3, MM/DD/YY)."""
    dates = {}
    with open(path) as f:
        f.readline()
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                dates[parts[0].lower()] = datetime.datetime.strptime(
                    parts[2].strip(), "%m/%d/%y").date()
            except ValueError:
                continue
    return dates


def _parse_seqres(path):
    """pdb_seqres.txt -> [(pdb_id, chain_id, seq)] for mol:protein chains."""
    out = []
    pid = cid = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if pid and seq_parts:
                    out.append((pid, cid, "".join(seq_parts)))
                head = line[1:].split()
                pid, cid = head[0].split("_")
                pid = pid.lower()
                seq_parts = []
                if "mol:protein" not in line:
                    pid = None  # skip non-protein records
            elif pid:
                seq_parts.append(line.strip())
    if pid and seq_parts:
        out.append((pid, cid, "".join(seq_parts)))
    return [(p, c, "".join(x if x in AA else "X" for x in s))
            for p, c, s in out if len(s) >= 20]


def _train_ids(path):
    ids = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            ids.add(row["complex_id"].lower())
    return ids


def _read_fasta(path):
    seqs = {}
    name = None
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = ""
            elif name:
                seqs[name] += line.strip()
    return seqs


def _write_fasta(items, path):
    with open(path, "w") as f:
        for name, seq in items:
            f.write(f">{name}\n{seq}\n")


def _cdrh3_db(chains, work, ncpu, limit=0):
    """ANARCI-type `chains` [(id, seq)], return [(db_id, cdrh3_seq, pdb_id)].

    db_id = "<pdb>_<chain>_h<j>". Cached to work/anarci_cdrh3.json.
    """
    cache = work / "anarci_cdrh3.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    from anarci import run_anarci
    if limit:
        chains = chains[:limit]
    _r0, r1, r2, _r3 = run_anarci(chains, scheme="imgt", ncpu=ncpu)
    out = []
    for i, (sid, _seq) in enumerate(chains):
        hits = r2[i] if r2[i] else []
        pdb, chain = sid.rsplit("_", 1)
        for j, hdet in enumerate(hits):
            if hdet.get("chain_type") != "H":
                continue
            numbered = r1[i][j][0] if r1[i] and len(r1[i]) > j else None
            cdr = cdrh3_from_numbering(numbered)
            if cdr and len(cdr) >= 3:
                out.append((f"{sid}_h{j}", cdr, pdb))
    cache.write_text(json.dumps(out))
    return out


def _mmseqs_search(mmseqs, query_fasta, db_fasta, out_prefix, work):
    """easy-search -> {query_id: [(target_id, pident, alnlen), ...]}."""
    cmd = [mmseqs, "easy-search", str(query_fasta), str(db_fasta),
           str(out_prefix), str(work), "-s", "7.5", "--max-seqs", "500",
           "--format-output", "query,target,pident,alnlen"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[error] mmseqs {' '.join(cmd)}\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    hits = {}
    with open(out_prefix) as f:
        for line in f:
            q, t, pid, aln = line.rstrip("\n").split("\t")
            hits.setdefault(q, []).append((t, float(pid), int(aln)))
    return hits


def _max_per_entry(hits, entry_of):
    """{query: {entry: max_pident}} over hits; entry_of: db target id -> pdb."""
    per = {}
    for q, hs in hits.items():
        m = {}
        for t, pid, _aln in hs:
            e = entry_of(t)
            if e and pid > m.get(e, 0.0):
                m[e] = pid
        per[q] = m
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqres", default=str(Path.home() / "abag_xm/leak_audit/pdb_seqres.txt"))
    ap.add_argument("--entries", default=str(Path.home() / "abag_xm/leak_audit/entries.idx"))
    ap.add_argument("--train_manifest", default=str(Path.home() / "ABAG-Rank/data/full_sabdab_train_manifest.csv"))
    ap.add_argument("--targets_parquet", default=str(ROOT / "docs/implementation-parity-data/abag-xm-targets.parquet"))
    ap.add_argument("--yaml_dir", default=str(ROOT / "examples/abag_xm"))
    ap.add_argument("--cdrh3_fasta", default=str(Path.home() / "abag_xm/cdrh3/cdrh3_all.fasta"))
    ap.add_argument("--mmseqs", default=str(Path.home() / "localcolabfold/.pixi/envs/default/bin/mmseqs"))
    ap.add_argument("--ncpu", type=int, default=8)
    ap.add_argument("--work_dir", default=str(Path.home() / "abag_xm/leak_audit/work"))
    ap.add_argument("--out_parquet", default=str(Path.home() / "abag_xm/leak_audit/leak_audit.parquet"))
    ap.add_argument("--out_md", default=str(Path.home() / "abag_xm/leak_audit/leak_audit.md"))
    ap.add_argument("--limit_chains", type=int, default=0, help="smoke test: cap ANARCI chains")
    a = ap.parse_args()

    import pandas as pd
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "tmp").mkdir(exist_ok=True)

    print("parsing entries.idx ...", flush=True)
    dates = _parse_entries(a.entries)
    print("parsing pdb_seqres.txt ...", flush=True)
    chains = _parse_seqres(a.seqres)
    print(f"  {len(dates)} entries, {len(chains)} protein chains")
    train = _train_ids(a.train_manifest)
    print(f"  {len(train)} ABAG-Rank train complex_ids")

    tdf = pd.read_parquet(a.targets_parquet)
    targets = sorted(tdf["pdb_id"])
    overlap = set(targets) & train
    print(f"  ranker-leg assert: {len(overlap)} of 164 targets in ABAG-Rank train (must be 0)")
    assert not overlap, f"targets in ABAG-Rank train: {sorted(overlap)}"

    # ---------------- leg A DB: CDR-H3s of train entries ----------------
    train_chains = [(f"{p}_{c}", s) for p, c, s in chains if p in train]
    print(f"ANARCI over {len(train_chains)} train-entry chains ...", flush=True)
    cdr_db = _cdrh3_db(train_chains, work, a.ncpu, a.limit_chains)
    print(f"  {len(cdr_db)} DB CDR-H3 sequences")
    cdr_entry = {dbid: pdb for dbid, _s, pdb in cdr_db}
    for name, cut in CUTOFFS.items():
        items = [(dbid, s) for dbid, s, pdb in cdr_db
                 if dates.get(pdb, datetime.date.max) < cut]
        _write_fasta(items, work / f"db_cdrh3_{name}.fasta")
        print(f"  {name}: {len(items)} DB CDR-H3s ({len({p for _d, _s, p in cdr_db if dates.get(p, datetime.date.max) < cut})} entries)")

    # ---------------- leg B DBs: PDB protein chains pre-cutoff ----------------
    for name, cut in CUTOFFS.items():
        items = [(f"{p}_{c}", s) for p, c, s in chains
                 if dates.get(p, datetime.date.max) < cut]
        _write_fasta(items, work / f"db_pdb_{name}.fasta")
        print(f"  {name}: {len(items)} PDB protein chains", flush=True)
    _write_fasta(train_chains, work / "db_train_chains.fasta")

    # ---------------- queries ----------------
    target_cdr = _read_fasta(a.cdrh3_fasta)  # ids "pid_hN", 117 unique
    ag_queries = []
    for t in targets:
        ys = _yaml_seqs(Path(a.yaml_dir) / f"{t}.yaml")
        if ys.get("A"):
            ag_queries.append((t, ys["A"]))
    _write_fasta(ag_queries, work / "query_antigen.fasta")
    print(f"queries: {len(target_cdr)} CDR-H3, {len(ag_queries)} antigens", flush=True)

    # ---------------- searches ----------------
    results = {}
    for name in CUTOFFS:
        print(f"leg A {name} ...", flush=True)
        h = _mmseqs_search(a.mmseqs, a.cdrh3_fasta, work / f"db_cdrh3_{name}.fasta",
                           work / f"hits_cdrh3_{name}.tsv", work / "tmp")
        results[f"cdrh3_{name}"] = _max_per_entry(h, lambda t: cdr_entry.get(t))
        print(f"leg B {name} ...", flush=True)
        h = _mmseqs_search(a.mmseqs, work / "query_antigen.fasta", work / f"db_pdb_{name}.fasta",
                           work / f"hits_ag_{name}.tsv", work / "tmp")
        results[f"ag_{name}"] = _max_per_entry(h, lambda t: t.split("_")[0])
    print("ranker leg ...", flush=True)
    h = _mmseqs_search(a.mmseqs, work / "query_antigen.fasta", work / "db_train_chains.fasta",
                       work / "hits_ag_train.tsv", work / "tmp")
    results["ag_train"] = _max_per_entry(h, lambda t: t.split("_")[0])

    # ---------------- per-target table ----------------
    # map target -> its CDR-H3 query ids via the manifest's cdrh3_sequences
    seq_to_qid = {}
    for qid, s in target_cdr.items():
        seq_to_qid.setdefault(s, qid)
    tcdr = dict(zip(tdf["pdb_id"], tdf["cdrh3_sequences"]))
    rows = []
    for t in targets:
        row = {"target": t}
        qids = [seq_to_qid[s] for s in tcdr.get(t, []) if s in seq_to_qid]
        for name in CUTOFFS:
            cmax, cent = 0.0, None
            for q in qids:
                for e, pid in results[f"cdrh3_{name}"].get(q, {}).items():
                    if pid > cmax:
                        cmax, cent = pid, e
            agm = results[f"ag_{name}"].get(t, {})
            amax = max(agm.values()) if agm else 0.0
            aent = max(agm, key=agm.get) if agm else None
            row[f"cdrh3_max_id_{name}"] = round(cmax, 2)
            row[f"ag_max_id_{name}"] = round(amax, 2)
            row[f"cdrh3_best_entry_{name}"] = cent
            row[f"ag_best_entry_{name}"] = aent
            shared = {e for e, pid in (results[f"cdrh3_{name}"].get(qids[0], {}) if qids else {}).items()
                      if pid >= FLAG_IDENT} if qids else set()
            for q in qids[1:]:
                shared |= {e for e, pid in results[f"cdrh3_{name}"].get(q, {}).items()
                           if pid >= FLAG_IDENT}
            flagged = [e for e in shared if agm.get(e, 0.0) >= FLAG_IDENT]
            row[f"flag_{name}"] = ",".join(sorted(flagged))
        tr = results["ag_train"].get(t, {})
        row["abagrank_train_max_id"] = round(max(tr.values()), 2) if tr else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(a.out_parquet, index=False)
    print(f"wrote {a.out_parquet} ({len(out)} rows)")

    # ---------------- summary ----------------
    md = ["# AbAg-XM leak audit (spec 1.4)", ""]
    md.append(f"DBs: CDR-H3 from ANARCI over {len(train_chains)} chains of "
              f"{len(train)} ABAG-Rank-train SAbDab entries (opig TSV dead; "
              f"pass-5 route); antigen leg vs all PDB protein chains "
              f"(pdb_seqres + entries.idx). Flag rule: CDR-H3 >=90% AND "
              f"antigen >=90% identity vs the SAME pre-cutoff entry.")
    md.append(f"Ranker leg: 0 of 164 targets appear in ABAG-Rank train "
              f"complex_ids (asserted).")
    md.append("")
    md.append("| leg | cutoff | median max-id | max max-id | n>=90% | n>=70% |")
    md.append("|---|---|---|---|---|---|")
    for col, label in (("cdrh3_max_id_pre2021", "CDR-H3"),
                       ("cdrh3_max_id_pre2023", "CDR-H3"),
                       ("ag_max_id_pre2021", "antigen"),
                       ("ag_max_id_pre2023", "antigen"),
                       ("abagrank_train_max_id", "antigen vs ABAG-Rank train")):
        s = out[col]
        cut = "2021-09-30" if "pre2021" in col else ("2023-06-01" if "pre2023" in col else "any (train)")
        md.append(f"| {label} | {cut} | {s.median():.1f} | {s.max():.1f} "
                  f"| {(s >= 90).sum()} | {(s >= 70).sum()} |")
    md.append("")
    for name in CUTOFFS:
        fl = out[out[f"flag_{name}"] != ""]
        md.append(f"flagged ({name}): {len(fl)} targets"
                  + (": " + ", ".join(f"{r.target} [{r[f'flag_{name}']}]"
                                      for r in fl.itertuples()) if len(fl) else ""))
    Path(a.out_md).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
