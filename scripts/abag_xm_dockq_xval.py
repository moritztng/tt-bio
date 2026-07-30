#!/usr/bin/env python3
"""Cross-validate shipped DockQ 2.1.3 labels against tinyprot (addendum A3).

Every AbAg-XM number rests on the DockQ labels and this project already shipped
one real DockQ bug (tt-bio-dockq-irmsd-key-casing-bug). This script scores a
stratified sample of (target, generator, sample) triples with BOTH
implementations on the SAME declared chain pair:

  - shipped label: DockQ 2.1.3 via scripts/abag_xm_dockq_interface.py, read back
    from the labels JSON (dockq/fnat/iRMSD/LRMSD + resolved chain ids).
  - tinyprot 0.1.0 (MIT, github.com/bjing2016/tinyprot) dockQ() on the same
    pair, chains residue-intersected to identical grids (make_compatible).

Subcommands:
  select  stratified triple sample from ranker_scores.csv + targets parquet (pc)
  score   run tinyprot on the selected triples (qb1, tinyprot venv)

Loader note: model CIFs from OpenDDE/Protenix lack _entity_poly_seq, so
Structure.from_mmcif cannot parse them. load_chains builds poly_seq from
_atom_site directly (same pattern as Structure.from_single_chain_mmcif) and
works uniformly on natives and all three generators' outputs.
"""
import argparse, gzip, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

GT_DIR = Path.home() / "abag_xm" / "ground_truth"
LABELS_DIR = Path.home() / "abag_xm" / "tier_a" / "labels"
GEN_DIR = {"boltz2": "boltz2", "opendde-abag": "opendde_abag",
           "protenix-v2": "protenix_v2"}
ANTI_PHOSPHO = {"9ly2", "9ly3", "9lz2"}   # null dockq; nothing to compare

# forced-coverage classes: >=1 triple each (plan §A3.2)
FORCED = {"multicopy": ["21av"], "seq_mismatch": ["9mz8"]}


# --------------------------------------------------------------------- select
def cmd_select(a):
    rng = np.random.RandomState(20260730)
    df = pd.read_csv(a.scores)
    df = df[df.gen.isin(GEN_DIR) & df.dockq.notna() &
            ~df.target.isin(ANTI_PHOSPHO)].copy()
    tg = pd.read_parquet(a.targets)
    fmt = tg.set_index("pdb_id")[["has_HL", "has_scFv", "has_peptide_antigen"]]

    picks = []          # list of (gen, row-index)
    forced_marks = {}   # (gen, idx) -> class name
    for gen, g in df.groupby("gen"):
        band_lo = g[(g.dockq >= 0.15) & (g.dockq <= 0.30)]
        band_hi = g[(g.dockq >= 0.70) & (g.dockq <= 0.90)]
        rest = g.drop(band_lo.index.union(band_hi.index))
        n_lo, n_hi = 8, 6
        take_lo = band_lo.iloc[rng.choice(len(band_lo), min(n_lo, len(band_lo)),
                                          replace=False)]
        take_hi = band_hi.iloc[rng.choice(len(band_hi), min(n_hi, len(band_hi)),
                                          replace=False)]
        # remaining 26 spread across the rest proportional to dockq decile
        dec = pd.cut(rest.dockq, bins=np.arange(0, 1.01, 0.1), include_lowest=True)
        counts = dec.value_counts().sort_index()
        alloc = (counts / counts.sum() * 26).round().astype(int)
        while alloc.sum() > 26:
            alloc[alloc.idxmax()] -= 1
        while alloc.sum() < 26:
            alloc[alloc.idxmax()] += 1
        take_rest = []
        for d, n in alloc.items():
            pool = rest[dec == d]
            if n > 0 and len(pool):
                take_rest.append(pool.iloc[rng.choice(len(pool),
                                                      min(n, len(pool)),
                                                      replace=False)])
        sel = pd.concat([take_lo, take_hi] + take_rest)
        picks.extend((gen, i) for i in sel.index)

    def _class_of(t):
        if t in FORCED["multicopy"]:
            return "multicopy"
        if t in FORCED["seq_mismatch"]:
            return "seq_mismatch"
        r = fmt.loc[t]
        if bool(r.has_peptide_antigen):
            return "peptide"
        if bool(r.has_scFv):
            return "scfv"
        if bool(r.has_HL):
            return "hl"
        return ""

    # forced coverage: swap in one triple per missing class (replace a random
    # unforced pick in the same gen as the donor triple)
    have = set()
    for (gen, i) in picks:
        c = _class_of(df.loc[i, "target"])
        if c:
            have.add(c)
        if df.loc[i, "dockq"] < 0.05:
            have.add("low_failure")
    need = {"multicopy", "seq_mismatch", "peptide", "scfv", "hl",
            "low_failure"} - have
    for cls in sorted(need):
        if cls == "low_failure":
            pool = df[df.dockq < 0.05]
        elif cls in FORCED:
            pool = df[df.target.isin(FORCED[cls])]
        else:
            col = {"peptide": "has_peptide_antigen", "scfv": "has_scFv",
                   "hl": "has_HL"}[cls]
            members = fmt.index[fmt[col]].tolist()
            pool = df[df.target.isin(members)]
        donor = pool.iloc[rng.choice(len(pool))]
        gen, idx = donor.gen, donor.name
        # replace an unforced pick in the same gen
        cands = [k for k, (g, i) in enumerate(picks)
                 if g == gen and (g, i) not in forced_marks]
        k = cands[rng.choice(len(cands))]
        picks[k] = (gen, idx)
        forced_marks[(gen, idx)] = cls

    out = df.loc[[i for _, i in picks], ["target", "gen", "rank", "dockq"]].copy()
    out["forced_class"] = [forced_marks.get((g, i), "") for g, i in picks]
    out = out.sort_values(["gen", "target", "rank"]).reset_index(drop=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} triples -> {a.out}")
    print(out.forced_class.value_counts().to_string())


# ---------------------------------------------------------------------- score
def _tiny():
    global read_mmcif, _parse_polymer, Structure, dockQ
    from tinyprot.mmcif import read_mmcif
    from tinyprot.parsing import _parse_polymer
    from tinyprot.structure import Structure
    from tinyprot.metrics import dockQ


def _read_cif(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return read_mmcif(f)
    with open(path) as f:
        return read_mmcif(f)


def load_chains(path, model_num="1"):
    """tinyprot chains keyed by label_asym_id + auth->label map.

    poly_seq is built from _atom_site (microheterogeneity: first residue wins,
    mirroring tinyprot's own drop_duplicates) so model CIFs without
    _entity_poly_seq parse exactly like RCSB natives.
    """
    cif = _read_cif(path)
    atom = cif["_atom_site"]
    if "pdbx_PDB_model_num" in atom:
        atom = atom[atom["pdbx_PDB_model_num"].astype(str) == str(model_num)]
    atom = atom.copy()
    atom["label_seq_id"] = pd.to_numeric(atom["label_seq_id"], errors="coerce")
    cif["_atom_site"] = atom
    auth2label = {}
    for auth, label in (atom[["auth_asym_id", "label_asym_id"]]
                        .drop_duplicates().itertuples(index=False)):
        auth2label.setdefault(str(auth), str(label))
    chains = {}
    poly = atom.dropna(subset=["label_seq_id"])
    for asym, sub in poly.groupby("label_asym_id"):
        poly_seq = (sub[["label_seq_id", "label_comp_id"]]
                    .drop_duplicates(subset=["label_seq_id"], keep="first")
                    .sort_values("label_seq_id").reset_index(drop=True))
        poly_seq.columns = ["num", "mon_id"]
        poly_seq["num"] = poly_seq["num"].astype(int)
        chains[str(asym)] = _parse_polymer(cif, entity_id=1, asym_id=str(asym),
                                           poly_seq=poly_seq)
    return chains, auth2label


def make_compatible(ref_chain, pred_chain):
    """Residue-intersect ref/pred to identical grids for tinyprot's aname assert.

    Keep positions present on both sides (by ridx = label_seq_id), standard on
    both sides, with equal rname. Equal rname per position implies equal CCD
    atom grids, so the dockQ() aname assert passes by construction. Mirrors
    DockQ v2's common-residue-universe semantics (its alignment path resolves
    point mismatches; tinyprot cannot, so mismatched positions are dropped --
    at most a handful of residues, recorded as n_dropped).
    """
    rs, ps = ref_chain.get_std_mask(), pred_chain.get_std_mask()
    rmap = {int(r): i for i, r in enumerate(ref_chain.ridx) if rs[i]}
    pmap = {int(r): i for i, r in enumerate(pred_chain.ridx) if ps[i]}
    common = sorted(set(rmap) & set(pmap))
    keep = [r for r in common
            if ref_chain.rname[rmap[r]] == pred_chain.rname[pmap[r]]]
    rmask = np.zeros(len(ref_chain.rname), bool)
    pmask = np.zeros(len(pred_chain.rname), bool)
    rmask[[rmap[r] for r in keep]] = True
    pmask[[pmap[r] for r in keep]] = True
    n_dropped = min(int(rs.sum()), int(ps.sum())) - len(keep)
    return ref_chain.residue_slice(rmask), pred_chain.residue_slice(pmask), n_dropped


def score_triple(target, gen, rank, labels_dir=LABELS_DIR, gt_dir=GT_DIR):
    d = json.loads((labels_dir / f"{GEN_DIR[gen]}_{target}.json").read_text())
    s = next(s for s in d["samples"] if s["rank"] == rank)
    blk = s["dockq"]
    native_path = gt_dir / f"{target}.cif"
    model_path = Path(blk["model"])
    if not model_path.exists():
        model_path = Path(s["cif"])

    native_chains, n_a2l = load_chains(native_path)
    model_chains, m_a2l = load_chains(model_path)
    rl1 = n_a2l.get(blk["native_chain1"], blk["native_chain1"])
    rl2 = n_a2l.get(blk["native_chain2"], blk["native_chain2"])
    pl1 = m_a2l.get(blk["model_chain1"], blk["model_chain1"])
    pl2 = m_a2l.get(blk["model_chain2"], blk["model_chain2"])

    ref, pred = Structure(), Structure()
    ref.chains, pred.chains = {}, {}
    mapping, dropped = {}, {}
    for rl, pl in ((rl1, pl1), (rl2, pl2)):
        if rl not in native_chains:
            raise KeyError(f"{target}: native label chain {rl!r} (auth "
                           f"{blk['native_chain1']}/{blk['native_chain2']}) not "
                           f"in {sorted(native_chains)}")
        if pl not in model_chains:
            raise KeyError(f"{target}: model chain {pl!r} not in "
                           f"{sorted(model_chains)}")
        rc, pc, nd = make_compatible(native_chains[rl], model_chains[pl])
        ref.chains[rl], pred.chains[pl] = rc, pc
        mapping[rl] = pl
        dropped[rl] = nd

    out = dockQ(ref, pred, mapping=mapping)
    pair = tuple(sorted(mapping))
    row = out.get(pair)
    res = {"target": target, "gen": gen, "rank": rank,
           "model": str(model_path),
           "ref_pair": f"{pair[0]}_{pair[1]}",
           "v2_dockq": blk["dockq"], "v2_fnat": blk["fnat"],
           "v2_irmsd": blk["iRMSD"], "v2_lrmsd": blk["LRMSD"],
           "n_dropped_chain1": dropped[rl1], "n_dropped_chain2": dropped[rl2],
           "status": "ok" if row else "no_interface_in_tinyprot"}
    if row:
        res.update({"tp_dockq": row["DockQ"], "tp_fnat": row["fnat"],
                    "tp_irmsd": row["iRMSD"], "tp_lrmsd": row["LRMSD"]})
    return res


def cmd_score(a):
    _tiny()
    trip = pd.read_csv(a.triples)
    rows = []
    for t in trip.itertuples(index=False):
        try:
            r = score_triple(t.target, t.gen, int(t.rank))
        except Exception as e:
            r = {"target": t.target, "gen": t.gen, "rank": int(t.rank),
                 "status": f"error: {type(e).__name__}: {e}"}
        rows.append(r)
        print(f"[{len(rows)}/{len(trip)}] {t.target} {t.gen} r{t.rank} "
              f"-> {r['status']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows -> {a.out}")
    ok = out[out.status == "ok"]
    print(f"ok {len(ok)}/{len(out)}")
    if len(ok):
        d = (ok.tp_dockq - ok.v2_dockq).abs()
        print(f"max|dDockQ| {d.max():.4f}  mean signed "
              f"{(ok.tp_dockq - ok.v2_dockq).mean():+.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--scores", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_select)
    p = sub.add_parser("score")
    p.add_argument("--triples", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_score)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
