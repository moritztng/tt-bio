"""Build the TFG reference panel: SAbDab antibody-antigen targets, their upstream input JSONs and constraints.

Follows OpenDDE's conditional-inference report, section 2.1: SAbDab entries released up to 2021-09-30;
Contact = the n closest CDR-antigen CA pairs at 3.5-7 A, each residue used once, requested at 3.5-8 A;
Pocket = the n antigen residues touched by the most antibody heavy atoms within 4.5 A, paratope "all",
min_fraction 0.5. Targets need five of each. Their 805-target split is not published, so this draws a
seeded, antigen-deduplicated sample from the same pool; the failing subset is chosen later by the
unguided screen (median DockQ < 0.23 over five seed-101 samples).

usage: build_inputs.py SABDAB_SUMMARY_CSV OUT_DIR [--n 130] [--max-tokens 900] [--seed 0]
Writes OUT_DIR/<id>/{native.cif, <id>_{unconstrained,contact,pocket}.json, meta.json} and OUT_DIR/panel.json.
"""
import argparse
import csv
import gzip
import json
import random
import urllib.request
from pathlib import Path

import gemmi
import numpy as np

N_CONSTRAINTS = 5
CHAIN_IDS = "ABCDEFGH"


def fetch(pdb, dest):
    if not dest.exists():
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb}.cif.gz", timeout=60) as r:
            dest.write_bytes(gzip.decompress(r.read()))
    return gemmi.read_structure(str(dest))


def chain_record(st, auth):
    """One-letter entity sequence and {label_seq: residue} for the first model's polymer of chain `auth`."""
    ch = st[0].find_chain(auth)
    if ch is None:
        return None
    poly = ch.get_polymer()
    if len(poly) == 0:
        return None
    ent = st.get_entity_of(poly)
    if ent is None or ent.polymer_type != gemmi.PolymerType.PeptideL or not ent.full_sequence:
        return None
    seq = ""
    for mon in ent.full_sequence:
        info = gemmi.find_tabulated_residue(gemmi.Entity.first_mon(mon))
        code = info.one_letter_code.upper() if info and info.is_amino_acid() else "X"
        seq += code if code != " " else "X"
    if "X" in seq:
        return None
    res = {r.label_seq: r for r in poly if r.label_seq is not None}
    return seq, res


def heavy(res):
    return [a for a in res if a.element.name != "H"]


def ca(res):
    a = res.find_atom("CA", "*")
    return np.array(a.pos.tolist()) if a else None


def build_target(row, cache):
    pdb = row["PDB"][-4:].lower()
    ab = [c for c in (row["Hchain"], row["Lchain"]) if c and c != "NA"]
    if len(set(ab)) != len(ab):
        return None, "scFv (H == L)"
    ag = row["antigen_chain"]
    st = fetch(pdb, cache / f"{pdb}.cif")
    st.setup_entities()
    recs = [chain_record(st, c) for c in ab + [ag]]
    if any(r is None for r in recs):
        return None, "non-standard or missing polymer"
    ab_recs, (ag_seq, ag_res) = recs[:-1], recs[-1]

    # CDR residues by locating SAbDab's CDR strings in the entity sequence.
    cdr = []
    for k, (seq, res) in enumerate(ab_recs):
        tag = "H" if k == 0 and row["Hchain"] not in ("", "NA") else "L"
        for i in (1, 2, 3):
            s = row.get(f"CDR-{tag}{i}", "")
            p = seq.find(s) if s and s != "NA" else -1
            if p < 0:
                return None, f"CDR-{tag}{i} not found in sequence"
            cdr += [(k, p + 1 + j) for j in range(len(s))]

    # Contact: closest CDR-antigen CA pairs at 3.5-7 A, each residue once.
    pairs = []
    for k, pos in cdr:
        r = ab_recs[k][1].get(pos)
        x = ca(r) if r else None
        if x is None:
            continue
        for apos, ar in ag_res.items():
            y = ca(ar)
            if y is not None:
                d = float(np.linalg.norm(x - y))
                if 3.5 <= d <= 7.0:
                    pairs.append((d, k, pos, apos))
    pairs.sort()
    used, contacts = set(), []
    for d, k, pos, apos in pairs:
        if (k, pos) in used or ("ag", apos) in used:
            continue
        used |= {(k, pos), ("ag", apos)}
        contacts.append((k, pos, apos, d))
        if len(contacts) == N_CONSTRAINTS:
            break
    if len(contacts) < N_CONSTRAINTS:
        return None, f"only {len(contacts)} contacts"

    # Pocket: antigen residues touched by the most antibody heavy atoms within 4.5 A.
    ab_xyz = np.array([a.pos.tolist() for _, res in ab_recs for r in res.values() for a in heavy(r)])
    touch = []
    for apos, ar in ag_res.items():
        xyz = np.array([a.pos.tolist() for a in heavy(ar)])
        if len(xyz) == 0:
            continue
        d = np.linalg.norm(ab_xyz[:, None, :] - xyz[None], axis=-1)
        n = int((d.min(axis=1) <= 4.5).sum())
        if n:
            touch.append((-n, apos))
    touch.sort()
    if len(touch) < N_CONSTRAINTS:
        return None, f"only {len(touch)} epitope residues"
    pocket = [apos for _, apos in touch[:N_CONSTRAINTS]]

    seqs = [s for s, _ in ab_recs] + [ag_seq]
    n_tok = sum(map(len, seqs))
    ag_ent = str(len(seqs))
    movable = list(CHAIN_IDS[: len(ab_recs)])
    contact_json = {
        "contact": [
            {"entity1": ag_ent, "copy1": 1, "position1": str(apos), "atom1": "CA",
             "entity2": str(k + 1), "copy2": 1, "position2": str(pos), "atom2": "CA",
             "min_distance": 3.5, "max_distance": 8.0}
            for k, pos, apos, _ in contacts
        ],
        "movable_chains": movable,
    }
    pocket_json = {
        "movable_chains": movable,
        "epitope": {
            "residues": [{"entity": ag_ent, "copy": 1, "position": str(p), "residue": ag_seq[p - 1]} for p in pocket],
            "paratope": "all",
            "min_fraction": 0.5,
        },
    }
    meta = {
        "pdb": pdb, "instance": row["INSTANCE"], "type": row["type"], "date": row["date"],
        "resolution": row["resolution"], "antigen_name": row["antigen_name"],
        "native_chains": ab + [ag], "pred_chains": list(CHAIN_IDS[: len(seqs)]),
        "antibody_chains": movable, "antigen_chains": [CHAIN_IDS[len(seqs) - 1]],
        "n_tokens": n_tok, "contact_ca_distances": [round(c[3], 3) for c in contacts],
        "pocket_touch_counts": [-t for t, _ in touch[:N_CONSTRAINTS]],
    }
    return (seqs, contact_json, pocket_json, meta, st), None


def write_native(st, native_chains, pred_chains, path):
    """Native trimmed to the scored chains, renamed to the prediction's chain ids, numbered by label_seq."""
    out = gemmi.Structure()
    out.cell, out.spacegroup_hm = st.cell, st.spacegroup_hm
    model = gemmi.Model("1")
    for src, dst in zip(native_chains, pred_chains):
        ch = gemmi.Chain(dst)
        for r in st[0].find_chain(src).get_polymer():
            r = r.clone()
            r.seqid = gemmi.SeqId(r.label_seq, " ")
            ch.add_residue(r)
        model.add_chain(ch)
    out.add_model(model)
    out.setup_entities()
    out.make_mmcif_document().write_file(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=130)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    cache = out / "_cif"
    cache.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(a.summary))
            if r["date"] <= "2021/09/30" and r["antigen_type"] == "PROTEIN" and r["method"] == "XRAY"
            and r["type"] in ("FAB", "FV", "SD-H") and "|" not in r["antigen_chain"]
            and r["resolution"] not in ("", "NA") and float(r["resolution"].split(",")[0]) <= 3.0]
    first = {}
    for r in rows:
        first.setdefault(r["PDB"], r)
    pool = sorted(first.values(), key=lambda r: r["PDB"])
    random.Random(a.seed).shuffle(pool)

    panel, seen_ag, skipped = [], set(), {}
    for row in pool:
        if len(panel) >= a.n:
            break
        name = row["antigen_name"].strip().lower()
        if name in seen_ag:
            continue
        try:
            built, why = build_target(row, cache)
        except Exception as e:  # malformed entry: skip, but say why
            built, why = None, f"error {type(e).__name__}: {e}"
        if built is None:
            skipped[row["PDB"]] = why
            continue
        seqs, contact, pocket, meta, st = built
        if meta["n_tokens"] > a.max_tokens:
            skipped[row["PDB"]] = f"{meta['n_tokens']} tokens"
            continue
        if seqs[-1] in seen_ag:
            continue
        seen_ag |= {name, seqs[-1]}
        tid = meta["pdb"]
        d = out / tid
        d.mkdir(exist_ok=True)
        base = {"modelSeeds": [101], "sequences": [
            {"proteinChain": {"count": 1, "id": [c], "sequence": s}} for c, s in zip(meta["pred_chains"], seqs)]}
        for cond, cons in (("unconstrained", None), ("contact", contact), ("pocket", pocket)):
            job = dict(base, name=f"{tid}_{cond}")
            if cons:
                job["constraint"] = cons
            (d / f"{tid}_{cond}.json").write_text(json.dumps([job], indent=1))
        write_native(st, meta["native_chains"], meta["pred_chains"], d / "native.cif")
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        panel.append(tid)
        print(f"{len(panel):3d} {tid} {meta['type']:5s} {meta['n_tokens']:4d} tok  {meta['antigen_name'][:40]}", flush=True)
    (out / "panel.json").write_text(json.dumps({"targets": panel, "skipped": skipped, "seed": a.seed,
                                                "pool": len(pool)}, indent=1))
    print(f"panel {len(panel)} from pool {len(pool)}; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
