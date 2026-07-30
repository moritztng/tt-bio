#!/usr/bin/env python3
"""AbAg-XM antigen de-duplication audit (addendum A2, CoFold Arena panel rules).

CoFold Arena enforces: one entry per antibody (earliest release), at most ONE
antibody per antigen UniProt accession, one physical copy per mmCIF. ARK-164 was
not built under those rules, so this script MEASURES the violation rate instead
of assuming it: every target's antigen entity (manifest fold_entity_id_2) is
mapped to UniProt accession(s) via the RCSB GraphQL API (batched, cached),
accession multiplicity across the panel is reported, and -- if any accession
maps to >1 target -- headline metrics are recomputed BOTH ways (full 164 panel
and earliest-release deduplicated panel). PRIMARY stays the full panel: panel
identity with ARK-164 anchors the 66.5%/66.4% harness validation.

Null-mapping antigens (short peptides, engineered chains) are a reported class,
NOT auto-duplicates of each other; an all-vs-all MMseqs2 search at 90% identity
(run separately, results ingested via --mmseqs_tsv) catches duplicates the
accession route misses.

    python3 scripts/abag_xm_antigen_dedup.py [--mmseqs_tsv antigen_seqs.m8]

Outputs docs/abag-xm-antigen-dedup.{md,csv}, docs/abag-xm-antigen-dedup.parquet
(the fifth release table) and docs/abag-xm-antigen-seqs.fasta (MMseqs2 input).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from abag_xm_ranker_bootstrap import (RANKERS, GENS, NS, THRESHOLDS,  # noqa: E402
                                      fold_constants)

RCSB_GQL = "https://data.rcsb.org/graphql"
QUERY = """{ entries(entry_ids: [%s]) { rcsb_id
  polymer_entities { rcsb_polymer_entity_container_identifiers { entity_id auth_asym_ids uniprot_ids }
    entity_poly { type pdbx_seq_one_letter_code_can } } } }"""


def rcsb_fetch(pdb_ids, cache_path):
    """entity records per entry, cached so re-runs are offline."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [p for p in pdb_ids if p.upper() not in cache]
    for i in range(0, len(missing), 55):
        chunk = missing[i:i + 55]
        q = QUERY % ", ".join(f'"{p}"' for p in chunk)
        req = urllib.request.Request(RCSB_GQL, data=json.dumps({"query": q}).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=60))
        if "errors" in d:
            raise RuntimeError(f"RCSB GraphQL errors: {d['errors']}")
        for e in d["data"]["entries"]:
            cache[e["rcsb_id"]] = e["polymer_entities"]
        time.sleep(1)  # be polite; 3 requests total
        print(f"  fetched {len(chunk)} entries ({i + len(chunk)}/{len(missing)})")
    cache_path.write_text(json.dumps(cache, indent=1))
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(Path.home() / ".coworker" / "state"
                                             / "abag-xm-closeout" / "abag-xm-targets.parquet"))
    ap.add_argument("--scores", default=str(Path.home() / ".coworker" / "state"
                                            / "abag-xm-closeout" / "ranker_scores.csv"))
    ap.add_argument("--cache", default=str(Path.home() / ".coworker" / "state"
                                           / "abag-xm-closeout" / "rcsb_uniprot_cache.json"))
    ap.add_argument("--antigen_fasta", default=str(Path.home() / ".coworker" / "state"
                                                   / "abag-xm-closeout" / "antigen_refs.fasta"),
                    help="fold-yaml 'A' (antigen) sequences per target")
    ap.add_argument("--mmseqs_tsv", default=None,
                    help="MMseqs2 easy-search m8 output for the all-vs-all antigen search")
    ap.add_argument("--out", default=str(ROOT / "docs" / "abag-xm-antigen-dedup"))
    a = ap.parse_args()

    tg = pd.read_parquet(a.targets)
    assert len(tg) == 164 and tg.pdb_id.is_unique
    tg["release_dt"] = pd.to_datetime(tg.release_date.str.replace("Z", "+0000"),
                                      format="%Y-%m-%dT%H:%M:%S%z")

    # ---- 1. accession mapping ---------------------------------------------------
    # superseded entries (9m8k->25st, 9m8l->25su) are null at RCSB; the manifest
    # mmcif_path already points at the superseding entry, whose entity numbering
    # the fold columns use, so query that one.
    tg["rcsb_entry"] = [str(s).upper() if pd.notna(s) else p.upper()
                        for p, s in zip(tg.pdb_id, tg.superseded_by)]
    cache = rcsb_fetch(sorted(set(tg.rcsb_entry)), Path(a.cache))

    # Antigen-entity resolution by SEQUENCE, not by manifest side order: ARK's
    # interface rows put the antigen on entity_id_1 for some targets (9dsg) and
    # entity_id_2 for others (21av). The fold yaml's "A" chain IS the antigen
    # (the leak audit's antigen leg used the same definition); match it against
    # every entity's canonical sequence by containment.
    refs = {}
    for blk in Path(a.antigen_fasta).read_text().split(">"):
        if blk.strip():
            hdr, *lines = blk.splitlines()
            refs[hdr.strip()] = "".join(lines).strip()
    assert len(refs) == 164, f"{len(refs)} antigen reference sequences != 164"

    recs = []
    fasta = []
    side_counts = {"entity_1": 0, "entity_2": 0, "other": 0}
    match_scores = {}
    for _, t in tg.iterrows():
        ref = refs[t.pdb_id]
        ents = cache[t.rcsb_entry]
        by_id = {int(pe["rcsb_polymer_entity_container_identifiers"]["entity_id"]): pe
                 for pe in ents}
        # k-mer containment: fraction of the reference's 10-mers present in the
        # entity sequence. Indel-tolerant (construct edits only break the k-mers
        # spanning a junction), alignment-free, deterministic. Exact containment
        # scores 1.0; antibody entities score ~0 against an antigen reference.
        K = min(10, len(ref))  # peptide antigens can be shorter than 10 aa
        kmers = {ref[i:i + K] for i in range(len(ref) - K + 1)}
        scores = {}
        for eid, pe in by_id.items():
            eseq = (pe["entity_poly"] or {}).get("pdbx_seq_one_letter_code_can") or ""
            if not eseq:
                continue
            scores[eid] = sum(k in eseq for k in kmers) / max(len(kmers), 1)
        eid = max(scores, key=scores.get)
        second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        assert scores[eid] >= 0.5 and scores[eid] - second >= 0.3, \
            f"{t.pdb_id}: antigen entity ambiguous, scores {scores}"
        match_scores[t.pdb_id] = round(scores[eid], 4)
        if eid == int(t.fold_entity_id_1):
            side_counts["entity_1"] += 1
        elif eid == int(t.fold_entity_id_2):
            side_counts["entity_2"] += 1
        else:
            side_counts["other"] += 1
        ent = by_id[eid]
        ids = ent["rcsb_polymer_entity_container_identifiers"]
        # the antigen entity must carry one of the two declared interface chains
        assert ({t.fold_auth_chain_id_1, t.fold_auth_chain_id_2}
                & set(ids["auth_asym_ids"])), \
            f"{t.pdb_id}: antigen entity {eid} chains {ids['auth_asym_ids']} " \
            f"disjoint from declared pair"
        unp = ids["uniprot_ids"] or []
        seq = (ent["entity_poly"] or {}).get("pdbx_seq_one_letter_code_can") or ""
        recs.append(dict(pdb_id=t.pdb_id, rcsb_entry=t.rcsb_entry,
                         antigen_entity_id=eid, antigen_match_score=scores[eid],
                         antigen_is_fold_entity_2=eid == int(t.fold_entity_id_2),
                         antigen_auth_chains=ids["auth_asym_ids"], uniprot_ids=unp,
                         uniprot_null=len(unp) == 0,
                         entity_poly_type=(ent["entity_poly"] or {}).get("type"),
                         antigen_seq_len=len(seq), antigen_seq=seq))
        fasta.append((t.pdb_id, seq))
    dd = pd.DataFrame(recs).merge(tg[["pdb_id", "release_dt", "release_date",
                                      "has_peptide_antigen", "superseded_by",
                                      "cdrh3_cluster"]], on="pdb_id")
    print(f"antigen side: {side_counts}")

    # ---- 2. multiplicity ---------------------------------------------------------
    dd["primary_accession"] = dd.uniprot_ids.map(lambda u: u[0] if u else None)
    acc2targets = (dd[~dd.uniprot_null].explode("uniprot_ids")
                   .groupby("uniprot_ids").pdb_id.apply(sorted))
    prim_groups = dd[~dd.uniprot_null].groupby("primary_accession").pdb_id.apply(sorted)
    dup_accessions = prim_groups[prim_groups.map(len) > 1]
    dd["accession_multiplicity"] = dd.primary_accession.map(
        prim_groups.map(len)).fillna(1).astype(int)
    dd["dup_group_id"] = None
    for gi, (acc, members) in enumerate(dup_accessions.items()):
        dd.loc[dd.pdb_id.isin(members), "dup_group_id"] = f"G{gi:02d}:{acc}"

    # ---- 3. antibody-side rule + mmCIF copies ------------------------------------
    # same-antibody-same-antigen: shared cdrh3_cluster AND any shared accession
    ab_flags = []
    for _, t1 in dd.iterrows():
        for _, t2 in dd.iterrows():
            if t1.pdb_id >= t2.pdb_id:
                continue
            if t1.cdrh3_cluster != t2.cdrh3_cluster:
                continue
            if set(t1.uniprot_ids) & set(t2.uniprot_ids):
                ab_flags.append((t1.pdb_id, t2.pdb_id))
    dd["antibody_dup_flag"] = dd.pdb_id.isin({p for pair in ab_flags for p in pair})
    # one physical copy per mmCIF: superseding entries must not both be in the panel
    sup = dd[dd.superseded_by.notna()][["pdb_id", "superseded_by"]]
    sup_bad = [f"{r.pdb_id}->{r.superseded_by}" for _, r in sup.iterrows()
               if r.superseded_by.lower() in set(dd.pdb_id)]
    assert not sup_bad, f"superseded pair both in panel: {sup_bad}"

    # ---- 4. MMseqs2 fallback for null-mapping antigens ---------------------------
    seq_edges = set()   # frozenset pairs, alignment >= 50 aa, >= 90% identity
    short_hits = []     # sub-50-aa alignments: peptide-fragment class, never merged
    fa_path = Path(a.out).parent / "abag-xm-antigen-seqs.fasta"
    fa_path.write_text("".join(f">{p}\n{s}\n" for p, s in fasta))
    if a.mmseqs_tsv:
        m8 = pd.read_csv(a.mmseqs_tsv, sep="\t", header=None,
                         names=["q", "t", "pident", "alen", "mism", "gap", "qs", "qe",
                                "ts", "te", "eval", "bits"])
        m8 = m8[(m8.q != m8.t) & (m8.pident >= 0.90)]  # mmseqs pident is 0-1
        seen = set()
        for _, r in m8.iterrows():
            e = frozenset((r.q, r.t))
            if e in seen:
                continue
            seen.add(e)
            if r.alen >= 50:
                seq_edges.add(e)
            else:
                short_hits.append((r.q, r.t, 100 * float(r.pident), int(r.alen)))

    # accession-sharing edges; union with sequence edges; connected components
    acc_edges = set()
    for members in prim_groups:
        if len(members) > 1:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    acc_edges.add(frozenset((members[i], members[j])))
    parent = {p: p for p in dd.pdb_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in acc_edges | seq_edges:
        x, y = sorted(e)
        parent[find(x)] = find(y)
    comp = {}
    for p in dd.pdb_id:
        comp.setdefault(find(p), []).append(p)
    seq_clusters = sorted((sorted(v) for v in comp.values() if len(v) > 1),
                          key=lambda v: (len(v), v), reverse=True)
    novel_edges = sorted(seq_edges - acc_edges)

    # ---- 5. deduplicated panels -----------------------------------------------------
    # accession dedup: earliest release per primary accession, tie-break lowest
    # pdb_id; null-mapping antigens are singleton groups (never deduplicated away)
    keep = set(dd[dd.uniprot_null].pdb_id)
    for _, members in prim_groups.items():
        sub = dd[dd.pdb_id.isin(members)].sort_values(["release_dt", "pdb_id"])
        keep.add(sub.iloc[0].pdb_id)
    dd["keeps_in_dedup"] = dd.pdb_id.isin(keep)
    dedup_ids = sorted(keep)
    # sequence dedup: earliest release per connected component over (accession OR
    # >=90%-identity >=50 aa sequence) edges -- the strictest "same antigen" view,
    # which also merges the null-mapping antigens the accession route cannot see
    keep_seq = set()
    for members in comp.values():
        sub = dd[dd.pdb_id.isin(members)].sort_values(["release_dt", "pdb_id"])
        keep_seq.add(sub.iloc[0].pdb_id)
    dd["keeps_in_seqdedup"] = dd.pdb_id.isin(keep_seq)
    seqdedup_ids = sorted(keep_seq)
    # cluster id per target (for the release table)
    cl_id = {}
    for ci, members in enumerate(seq_clusters):
        for p in members:
            cl_id[p] = f"C{ci:02d}"
    dd["antigen_seq_cluster"] = dd.pdb_id.map(cl_id).fillna("-")

    # ---- 6. both-ways headline metrics --------------------------------------------
    df = pd.read_csv(a.scores)
    scorable = df.groupby("target")["dockq"].apply(lambda s: s.notna().any())
    scorable_ids = sorted(scorable[scorable].index)

    def panel_metrics(panel_ids):
        ids = [t for t in panel_ids if t in scorable_ids]
        out = {}
        sp_all = {}
        for gen in GENS:
            sub = df[(df.gen == gen) & (df.target.isin(ids))]
            per = {k: {q: [] for q in (["oracle", "random"] + RANKERS)}
                   for k in [(thr, N) for thr in THRESHOLDS for N in NS]}
            sp = {rk: [] for rk in RANKERS}
            for t, g in sub.groupby("target"):
                fc = fold_constants(g, t, gen)
                for (thr, N), (orc, rnd, rnk) in fc.items():
                    per[(thr, N)]["oracle"].append(orc)
                    per[(thr, N)]["random"].append(rnd)
                    for rk in RANKERS:
                        per[(thr, N)][rk].append(rnk[rk])
                dq = g.dockq.to_numpy()
                if len(np.unique(dq)) > 1:
                    rd = pd.Series(dq).rank().to_numpy()
                    for rk in RANKERS:
                        rv = pd.Series(g[rk].to_numpy()).rank().to_numpy()
                        sp[rk].append(np.corrcoef(rd, rv)[0, 1])
            for (thr, N), q in per.items():
                orc, rnd = np.mean(q["oracle"]), np.mean(q["random"])
                for rk in RANKERS:
                    rnk = np.mean(q[rk])
                    gap = (rnk - rnd) / (orc - rnd) if abs(orc - rnd) > 1e-9 else np.nan
                    out[(gen, thr, N, rk)] = dict(oracle=orc, random=rnd,
                                                  ranked=rnk, gap_recovered=gap,
                                                  n_targets=len(q["oracle"]))
            sp_all[gen] = {rk: float(np.nanmean(v)) for rk, v in sp.items()}
        return out, sp_all

    full_metrics, full_sp = panel_metrics(scorable_ids)
    dedup_metrics, dedup_sp = ({}, {}) if dup_accessions.empty else panel_metrics(dedup_ids)
    seqdedup_metrics, seqdedup_sp = panel_metrics(seqdedup_ids)

    # ---- 7. outputs ------------------------------------------------------------------
    out_dd = dd.drop(columns=["antigen_seq"])
    out_dd["uniprot_ids_str"] = out_dd.uniprot_ids.map(lambda u: ";".join(u))
    out_dd["antigen_auth_chains_str"] = out_dd.antigen_auth_chains.map(";".join)
    csv_cols = ["pdb_id", "rcsb_entry", "antigen_entity_id", "antigen_match_score",
                "antigen_is_fold_entity_2", "antigen_auth_chains_str",
                "uniprot_ids_str", "uniprot_null", "entity_poly_type",
                "antigen_seq_len", "primary_accession", "accession_multiplicity",
                "dup_group_id", "antigen_seq_cluster", "keeps_in_dedup",
                "keeps_in_seqdedup", "antibody_dup_flag",
                "has_peptide_antigen", "release_date", "superseded_by", "cdrh3_cluster"]
    out_dd[csv_cols].rename(columns={"antigen_auth_chains_str": "antigen_auth_chains",
                                     "uniprot_ids_str": "uniprot_ids"}
                            ).to_csv(Path(a.out).with_suffix(".csv"), index=False)
    out_dd.drop(columns=["uniprot_ids_str", "antigen_auth_chains_str"]
                ).to_parquet(Path(a.out).with_suffix(".parquet"), index=False)

    lines = ["# AbAg-XM antigen de-duplication audit (addendum A2)",
             "",
             "CoFold Arena panel rules applied as an AUDIT of the 164-target panel: one "
             "entry per antibody, at most one antibody per antigen UniProt accession, one "
             "physical copy per mmCIF. Antigen entities were resolved by SEQUENCE (the fold "
             "yaml's A chain matched by containment against every RCSB entity; ARK's "
             "interface rows do not put the antigen on a consistent side) and mapped to "
             "UniProt via the RCSB GraphQL API (cached). PRIMARY "
             "reporting stays the full 164-target panel: panel identity with ARK-164 "
             "anchors the 66.5%/66.4% harness validation. The deduplicated view is the "
             "sensitivity analysis.",
             "",
             "## Accession mapping",
             "",
             f"- {int((~dd.uniprot_null).sum())}/164 antigens map to >=1 UniProt "
             f"accession; {int(dd.uniprot_null.sum())} are null-mapping "
             "(engineered/construct antigens without a SIFTS UniProt mapping). "
             "Null-mapping antigens are a reported class, NOT auto-duplicates "
             "of each other. Note: the manifest has_peptide_antigen flag is "
             "target-level (ANY interface of the entry); the fold antigen of a "
             "flagged target can still be a full protein (e.g. 9d73, 274 aa), "
             "and short true-peptide fold antigens (7-24 aa) DO carry UniProt "
             "accessions via their parent protein.",
             f"- Multi-accession antigen entities: "
             f"{int((dd.uniprot_ids.map(len) > 1).sum())} (first accession = primary key).",
             "",
             "## Accession multiplicity (primary accession per target)",
             ""]
    mult = dd.accession_multiplicity.value_counts().sort_index()
    for m, c in mult.items():
        lines.append(f"- multiplicity {m}: {c} targets")
    lines.append("")
    if dup_accessions.empty:
        lines.append("No antigen accession is shared by >1 target: the panel already "
                     "satisfies the one-antibody-per-antigen-accession rule, per-target "
                     "averages are independent in this respect, and no deduplicated view "
                     "is needed.")
    else:
        lines.append(f"{len(dup_accessions)} accessions are shared by >1 target "
                     f"({sum(len(m) for m in dup_accessions)} targets). Duplicate groups:")
        lines.append("")
        lines.append("| group | accession | members (kept first = earliest release) |")
        lines.append("|---|---|---|")
        for gi, (acc, members) in enumerate(dup_accessions.items()):
            sub = dd[dd.pdb_id.isin(members)].sort_values(["release_dt", "pdb_id"])
            lines.append(f"| G{gi:02d} | {acc} | "
                         + ", ".join(f"{p} ({r})" + (" KEEP" if j == 0 else "")
                                     for j, (p, r) in enumerate(zip(sub.pdb_id, sub.release_date)))
                         + " |")
        lines.append("")
        lines.append("Sensitivity (ANY shared accession, not just primary): "
                     f"{int(acc2targets.map(len).gt(1).sum())} accessions shared.")
        lines.append("")
    # antibody-side
    if ab_flags:
        lines.append(f"## Same-antibody-same-antigen pairs: {len(ab_flags)}")
        lines.append("")
        for p1, p2 in ab_flags:
            lines.append(f"- {p1} x {p2} (shared cdrh3_cluster + antigen accession)")
        lines.append("")
    else:
        lines += ["## Same-antibody-same-antigen pairs: 0", ""]
    lines.append(f"One physical copy per mmCIF: {int(dd.superseded_by.notna().sum())} "
                 "targets carry a superseded_by pointer; no superseding entry is in the "
                 "panel (asserted).")
    lines.append("")
    # mmseqs
    lines += ["## Sequence-level fallback (MMseqs2 all-vs-all, >=90% identity)", ""]
    if a.mmseqs_tsv:
        lines.append(f"{len(seq_edges)} unique cross-target pairs at >=90% identity "
                     f"over >=50 aa; {len(novel_edges)} of them span targets the "
                     "accession route does NOT group (different accessions, or "
                     "null-mapping antigens). Clusters below merge accession-sharing "
                     "and sequence hits into one graph (connected components).")
        lines.append("")
        lines.append("| cluster | members (accession or null) | accession-consistent? |")
        lines.append("|---|---|---|")
        acc_of = dict(zip(dd.pdb_id, dd.primary_accession.fillna("null")))
        for ci, members in enumerate(seq_clusters):
            accs = {acc_of[p] for p in members}
            consistent = "yes" if len(accs) == 1 else "**NOVEL span**"
            lines.append(f"| C{ci:02d} ({len(members)}) | "
                         + ", ".join(f"{p} [{acc_of[p]}]" for p in members)
                         + f" | {consistent} |")
        lines.append("")
        if short_hits:
            lines.append(f"Peptide-fragment hits (<50 aa alignment, not merged into "
                         f"clusters): {len(short_hits)}")
            lines.append("")
            for q, t, pid, alen in sorted(short_hits):
                lines.append(f"- {q} x {t}: {pid:.1f}% over {alen} aa "
                             f"({acc_of[q]} vs {acc_of[t]})")
            lines.append("")
    else:
        lines.append("PENDING: fasta written to docs/abag-xm-antigen-seqs.fasta; run "
                     "`mmseqs easy-search docs/abag-xm-antigen-seqs.fasta docs/abag-xm-antigen-seqs.fasta "
                     "out.m8 tmp --min-seq-id 0.9` and re-run with --mmseqs_tsv out.m8.")
    lines.append("")
    # both-ways metrics
    lines += ["## Headline metrics, full panel vs antigen-deduplicated", ""]
    if dup_accessions.empty:
        lines.append("Not applicable (no duplicates). Full-panel numbers are the "
                     "published ones (abag-xm-ranker-cis.md).")
    else:
        n_acc = len([t for t in dedup_ids if t in scorable_ids])
        n_seq = len([t for t in seqdedup_ids if t in scorable_ids])
        lines.append(f"Full panel: {len(scorable_ids)} scorable targets. "
                     f"Accession-deduplicated: {n_acc} scorable ({len(dedup_ids)} total). "
                     f"Sequence-deduplicated (accession + >=90%-identity clusters merged): "
                     f"{n_seq} scorable ({len(seqdedup_ids)} total). Point estimates over "
                     "the same seeded budget-N fold constants as the bootstrap doc; CIs "
                     "for the full panel are in abag-xm-ranker-cis.md.")
        lines.append("")
        lines.append("| generator | N | thr | metric | full | acc-dedup | seq-dedup |")
        lines.append("|---|---|---|---|---|---|---|")
        for gen in GENS:
            for N in NS:
                for thr in THRESHOLDS:
                    f0 = full_metrics[(gen, thr, N, "ranking_score")]
                    d0 = dedup_metrics[(gen, thr, N, "ranking_score")]
                    s0 = seqdedup_metrics[(gen, thr, N, "ranking_score")]
                    f1 = full_metrics[(gen, thr, N, "deeprank_ab")]
                    d1 = dedup_metrics[(gen, thr, N, "deeprank_ab")]
                    s1 = seqdedup_metrics[(gen, thr, N, "deeprank_ab")]
                    for label, fv, dv, sv in (
                            ("oracle", f0["oracle"], d0["oracle"], s0["oracle"]),
                            ("random", f0["random"], d0["random"], s0["random"]),
                            ("ranked (ranking_score)", f0["ranked"], d0["ranked"],
                             s0["ranked"]),
                            ("ranked (deeprank_ab)", f1["ranked"], d1["ranked"],
                             s1["ranked"]),
                            ("deeprank_ab gap-recovered", f1["gap_recovered"],
                             d1["gap_recovered"], s1["gap_recovered"])):
                        lines.append(f"| {gen} | {N} | {thr} | {label} | {fv:.3f} | "
                                     f"{dv:.3f} ({100*(dv-fv):+.1f}) | {sv:.3f} "
                                     f"({100*(sv-fv):+.1f}) |")
        lines.append("")
        lines.append("Per-target Spearman (mean across panel; the diagnostic whose "
                     "independence assumption duplicates would break):")
        lines.append("")
        lines.append("| generator | ranker | full | acc-dedup | seq-dedup |")
        lines.append("|---|---|---|---|---|")
        for gen in GENS:
            for rk in ("ranking_score", "deeprank_ab", "abag_rank"):
                lines.append(f"| {gen} | {rk} | {full_sp[gen][rk]:.3f} | "
                             f"{dedup_sp[gen][rk]:.3f} | {seqdedup_sp[gen][rk]:.3f} |")
        lines.append("")
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}.md / .csv / .parquet; fasta at {fa_path.name}")
    print(f"null-mapping: {int(dd.uniprot_null.sum())}, dup accessions: "
          f"{len(dup_accessions)}, ab-same-ag pairs: {len(ab_flags)}, "
          f"acc-dedup keeps: {len(dedup_ids)}, seq-dedup keeps: {len(seqdedup_ids)}, "
          f"novel seq edges: {len(novel_edges)}")


if __name__ == "__main__":
    main()
