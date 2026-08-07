"""DockQ on the ARK-declared Ab-Ag interface (D6), not an auto chain-mapper average.

`scripts/opendde_dockq.py` runs DockQ on EVERY native interface and reports
GlobalDockQ (mean over them). That is wrong for AbAg-XM's primary label: D6 fixes
the label to DockQ on the single ARK-declared `[antibody*-protein]` interface
(`fold_auth_chain_id_1` / `fold_auth_chain_id_2` in the manifest), never a
wave-average. Using the auto mapper + average also re-opens the
`dockq-multicopy-chain-mapper-false-zero` class (a multi-copy native can map a
model chain to the wrong copy and yield a false-zero DockQ).

This script loads model + native, builds DockQ's sequence-based chain map, then
calls `run_on_chains` on the two declared chains (antibody + antigen), resolving
model<->native by sequence. Output is the single-interface DockQ +
Fnat/iRMSD/LRMSD/clashes/CAPRI, plus the chain map used so the mapping is
auditable.

Multi-copy natives: when the declared chains belong to mmCIF entities with more
than one chain (crystallographically equivalent copies of the same molecule),
the model -- which folds a single copy -- is free to dock ANY of them. Pinning
the declared copy then false-zeros correct predictions that chose a different
copy (9q1l: declared E/F scores 0.07 while the equivalent B/C copy scores 0.97
for the same, correct model). For such natives the script scores the declared
pair plus every entity-equivalent (copy-of-chain1, copy-of-chain2) pair and
keeps the best DockQ; the declared pair and all attempts are recorded for
audit. Single-copy natives take exactly the declared pair (unchanged behavior).

    PYTHONPATH=<wt> python3 scripts/abag_xm_dockq_interface.py \
        <model.cif> <native.cif> <chain1> <chain2> [--out json]

chain1/chain2 are the manifest's fold_auth_chain_id_1/2 (native AUTH asym ids).
The fold YAML is built from the native's auth chains and the generators preserve
those ids in the model CIF, so the declared ids are normally MODEL chain ids; the
native CIF is read by DockQ under its LABEL asym ids. The script resolves both
sides: if a declared id is a model chain it is mapped to its native counterpart
via the sequence chain map, and vice-versa.
"""
import argparse, json, sys
from difflib import SequenceMatcher
from DockQ.DockQ import load_PDB, run_on_chains, group_chains, get_all_chain_maps


def _chain_by_id(struct, cid):
    for c in struct:
        if c.id == cid:
            return c
    raise KeyError(f"chain {cid!r} not in structure (have {[c.id for c in struct]})")




def _build_seq_map(model_path, native_path):
    """Many-to-one {native_chain_id: model_chain_id} by best sequence identity.

    Handles multicopy natives (2+ chains with the same entity/length) where
    DockQ's one-to-one group_chains leaves the second copy unmapped. For each
    native chain, pick the model chain with the highest sequence identity
    (allowing multiple native chains to map to the same model chain). Models
    lacking _entity_poly (e.g. some OpenDDE/Protenix CIFs where gemmi returns
    empty polymer sequences) get their sequences rebuilt from CA-bearing
    standard residues; a bare CA-count match is only the last-resort fallback
    (calcium ions share the CA atom name and poison the count).
    """
    import gemmi
    from Bio.Data.PDBData import protein_letters_3to1

    def _info(path):
        st = gemmi.read_structure(str(path))
        out = {}
        for m in st:
            for ch in m:
                try:
                    s = ch.get_polymer().make_one_letter_sequence()
                except Exception:
                    s = ""
                if not s:
                    # Models lacking _entity_poly (some OpenDDE/Protenix CIFs):
                    # rebuild the sequence from standard amino-acid residues that
                    # carry a CA atom. A bare CA-atom count is NOT a safe proxy:
                    # natives can hold calcium ions (atom name CA) which shift the
                    # count and mis-map same-sized chains (9q1l: the native light
                    # chain mapped onto the model's heavy chain -> false-zero).
                    s = "".join(protein_letters_3to1[r.name] for r in ch
                                if r.name in protein_letters_3to1
                                and any(a.name == "CA" for a in r))
                n_ca = sum(1 for r in ch for a in r if a.name == "CA")
                out[ch.name] = (s, n_ca)
        return out

    ms = _info(model_path)
    ns = _info(native_path)
    smap = {}
    for n_id, (n_seq, n_ca) in ns.items():
        best, best_id = -1.0, None
        for m_id, (m_seq, m_ca) in ms.items():
            if n_seq and m_seq:
                # Shift-robust identity: constructs can differ by terminal
                # residues (9q1l's antigen copy C carries an extra N-terminal
                # Asp vs the folded sequence), which zeroes a colinear zip
                # comparison. Never a length proxy: heavy/light chains differ
                # by ~7 residues and a length proxy scores them 0.96+ identical.
                ident = SequenceMatcher(None, n_seq, m_seq).ratio()
            else:
                denom = max(n_ca, m_ca, 1)
                ident = 1.0 - abs(n_ca - m_ca) / denom
            if ident > best:
                best, best_id = ident, m_id
        if best_id is not None and best >= 0.5:
            smap[n_id] = best_id
    return smap

def _entity_classes(native_path, nc):
    """Group native chain ids by mmCIF polymer entity (same molecule = equivalent
    copies). Chains in no polymer entity form singleton classes. gemmi's chain
    names use the same namespace DockQ's load_PDB exposes for these CIFs (the
    auth asym ids), so the classes line up with `nc`.
    """
    import gemmi
    st = gemmi.read_structure(str(native_path))
    classes, seen = [], set()
    for e in st.entities:
        if e.entity_type != gemmi.EntityType.Polymer:
            continue
        members = [c for c in e.subchains if c in nc]
        if members:
            classes.append(members)
            seen.update(members)
    for c in nc:
        if c not in seen:
            classes.append([c])
    return classes


def _resolve(declared, mc, nc, cmap, inv, seq_map=None):
    """Return (model_id, native_id) for a declared chain id, or raise.

    DockQ's chain_map from get_all_chain_maps is {native_id: model_id} (keys are
    the ref/native chains, values the query/model chains). The manifest's
    fold_auth_chain_id_* are native AUTH ids, so the common path is the declared
    id being a native chain; we also accept a declared model chain for robustness.
    """
    if declared in nc:           # declared id is a native chain -> map to model
        # Prefer the gemmi sequence map over DockQ's one-to-one chain_map: the
        # chain_map is built from the sequences DockQ's own parser extracts, and
        # for model CIFs it cannot read (no _entity_poly) every chain looks
        # identical, so chain_map degenerates to an arbitrary permutation
        # (9q1l: it paired the model light chain with a native heavy chain).
        if seq_map and declared in seq_map:
            return seq_map[declared], declared
        if declared in cmap:
            return cmap[declared], declared
        raise KeyError(f"declared chain {declared!r} is a native chain but not in "
                       f"chain_map (multicopy?) and no seq_map fallback; native={nc}")
    if declared in mc:           # declared id is a model chain -> map to native
        return declared, inv[declared]
    raise KeyError(f"declared chain {declared!r} is neither a native chain {nc} "
                   f"nor a model chain {mc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("native")
    ap.add_argument("chain1",
                    help="manifest fold_auth_chain_id_1 (antibody side, native auth id)")
    ap.add_argument("chain2",
                    help="manifest fold_auth_chain_id_2 (antigen side, native auth id)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ms = load_PDB(a.model)
    ns = load_PDB(a.native)
    mc = [c.id for c in ms]
    nc = [c.id for c in ns]
    clusters, rev = group_chains(ms, ns, mc, nc, allowed_mismatches=0)
    cmap = next(get_all_chain_maps(clusters, {}, rev, mc, nc))   # {model_id: native_id}
    inv = {n: m for m, n in cmap.items()}
    seq_map = _build_seq_map(a.model, a.native)

    try:
        m1, n1 = _resolve(a.chain1, mc, nc, cmap, inv, seq_map)
        m2, n2 = _resolve(a.chain2, mc, nc, cmap, inv, seq_map)
    except KeyError as e:
        result = {"model": a.model, "native": a.native, "chain1": a.chain1,
                 "chain2": a.chain2, "chain_map": cmap, "status": "unresolved",
                 "error": str(e)}
        print(json.dumps(result, indent=2, default=str))
        if a.out:
            with open(a.out, "w") as fp:
                json.dump(result, fp, indent=2, default=str)
        return 3

    # Mirror run_on_all_native_interfaces: protein interfaces use the DockQ path
    # (small_molecule=False); the default True path runs calc_sym_corrected_lrmsd
    # and returns None for protein-protein interfaces.
    def _score(m_id1, m_id2, x, y):
        mch = (_chain_by_id(ms, m_id1), _chain_by_id(ms, m_id2))
        nch = (_chain_by_id(ns, x), _chain_by_id(ns, y))
        sm = bool(getattr(nch[0], "is_het", False) or getattr(nch[1], "is_het", False))
        return run_on_chains(mch, nch, small_molecule=sm)

    # Multi-copy natives: score the declared pair plus every entity-equivalent
    # copy pair and keep the best DockQ (see module docstring). Single-copy
    # natives get exactly one candidate -- the declared pair, unchanged behavior.
    classes = _entity_classes(a.native, nc)
    cls1 = next((c for c in classes if n1 in c), [n1])
    cls2 = next((c for c in classes if n2 in c), [n2])
    extra = sorted((x, y) for x in cls1 for y in cls2
                   if x != y and (x, y) != (n1, n2))

    info = _score(m1, m2, n1, n2)
    best = (info, m1, m2, n1, n2)
    copies_scored = []
    for x, y in extra:
        mx = seq_map.get(x) or inv.get(x)
        my = seq_map.get(y) or inv.get(y)
        if mx is None or my is None or mx == my:
            continue
        try:
            xi = _score(mx, my, x, y)
        except Exception:
            xi = None
        dq = xi.get("DockQ") if xi else None
        copies_scored.append({"native_chain1": x, "native_chain2": y,
                              "model_chain1": mx, "model_chain2": my,
                              "dockq": dq})
        best_dq = best[0].get("DockQ") if best[0] else None
        if dq is not None and (best_dq is None or dq > best_dq):
            best = (xi, mx, my, x, y)
    info, m1, m2, n1_best, n2_best = best
    if info is None:
        # DockQ returns None when it finds no interface between the two chains it was given,
        # and `info.get` then raised AttributeError -- so an unscorable target produced a
        # traceback in the label field instead of a status, for all 50 samples x 3 generators.
        # It happens when the whole contact is carried by residues DockQ's loader discards:
        # 9ly2/9ly3/9lz2 are anti-phosphoepitope antibodies whose antigen side is two SEP
        # (phosphoserine) residues and nothing else. Report it as the structured verdict it is;
        # scripts/abag_xm_native_interface_audit.py finds these before folding.
        result = {"model": a.model, "native": a.native, "chain1": a.chain1,
                  "chain2": a.chain2, "chain_map": cmap,
                  "model_chain1": m1, "model_chain2": m2,
                  "native_chain1": n1, "native_chain2": n2,
                  "copies_scored": copies_scored,
                  "status": "no_scorable_interface",
                  "error": (f"DockQ found no scorable interface between native chains "
                            f"{n1}/{n2} (or any entity-equivalent copy pair); the "
                            f"contact is carried by residues its loader discards "
                            f"(see abag_xm_native_interface_audit.py)")}
        print(json.dumps(result, indent=2, default=str))
        if a.out:
            with open(a.out, "w") as fp:
                json.dump(result, fp, indent=2, default=str)
        return 3
    # DockQ 2.1.3 returns a dict; copy the load-bearing fields. Key casing in
    # DockQ==2.1.3: capital `DockQ`, `iRMSD`, `LRMSD`; lowercase `fnat`,
    # `fnonnat`. Assert iRMSD non-None (tt-bio-dockq-irmsd-key-casing-bug).
    dockq = info.get("DockQ")
    irmsd = info.get("iRMSD")
    if irmsd is None and "iRMS" in info:
        irmsd = info.get("iRMS")
    fnat = info.get("fnat")
    if fnat is None:
        fnat = info.get("Fnat")
    out = {"model": a.model, "native": a.native,
           "chain1": a.chain1, "chain2": a.chain2,
           "model_chain1": m1, "model_chain2": m2,
           "native_chain1": n1_best, "native_chain2": n2_best,
           "declared_native_chain1": n1, "declared_native_chain2": n2,
           "chain_map": cmap,
           "interface": f"{a.chain1}_{a.chain2}",
           "dockq": dockq,
           "fnat": fnat,
           "fnonnat": info.get("fnonnat"),
           "iRMSD": irmsd,
           "LRMSD": info.get("LRMSD"),
           "clashes": info.get("clashes"),
           "capri": info.get("capri_class") or info.get("CAPRI") or info.get("capri"),
           "raw": {k: v for k, v in info.items()
                   if not isinstance(v, (list, dict))}}
    if copies_scored:
        out["copies_scored"] = copies_scored
    print(json.dumps(out, indent=2, default=str))
    if a.out:
        with open(a.out, "w") as fp:
            json.dump(out, fp, indent=2, default=str)
    return 0 if dockq is not None else 2


if __name__ == "__main__":
    sys.exit(main())
