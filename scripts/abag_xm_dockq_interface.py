"""DockQ on the ARK-declared Ab-Ag interface (D6), not an auto chain-mapper average.

`scripts/opendde_dockq.py` runs DockQ on EVERY native interface and reports
GlobalDockQ (mean over them). That is wrong for AbAg-XM's primary label: D6 fixes
the label to DockQ on the single ARK-declared `[antibody*-protein]` interface
(`fold_auth_chain_id_1` / `fold_auth_chain_id_2` in the manifest), never a
wave-average. Using the auto mapper + average also re-opens the
`dockq-multicopy-chain-mapper-false-zero` class (a multi-copy native can map a
model chain to the wrong copy and yield a false-zero DockQ).

This script loads model + native, builds DockQ's sequence-based chain map, then
calls `run_on_chains` on ONLY the two declared chains (antibody + antigen),
resolving model<->native by sequence. Output is the single-interface DockQ +
Fnat/iRMSD/LRMSD/clashes/CAPRI, plus the chain map used so the mapping is
auditable.

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
from DockQ.DockQ import load_PDB, run_on_chains, group_chains, get_all_chain_maps


def _chain_by_id(struct, cid):
    for c in struct:
        if c.id == cid:
            return c
    raise KeyError(f"chain {cid!r} not in structure (have {[c.id for c in struct]})")


def _resolve(declared, mc, nc, cmap, inv):
    """Return (model_id, native_id) for a declared chain id, or raise.

    DockQ's chain_map from get_all_chain_maps is {native_id: model_id} (keys are
    the ref/native chains, values the query/model chains). The manifest's
    fold_auth_chain_id_* are native AUTH ids, so the common path is the declared
    id being a native chain; we also accept a declared model chain for robustness.
    """
    if declared in nc:           # declared id is a native chain -> map to model
        return cmap[declared], declared
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

    try:
        m1, n1 = _resolve(a.chain1, mc, nc, cmap, inv)
        m2, n2 = _resolve(a.chain2, mc, nc, cmap, inv)
    except KeyError as e:
        result = {"model": a.model, "native": a.native, "chain1": a.chain1,
                 "chain2": a.chain2, "chain_map": cmap, "status": "unresolved",
                 "error": str(e)}
        print(json.dumps(result, indent=2, default=str))
        if a.out:
            with open(a.out, "w") as fp:
                json.dump(result, fp, indent=2, default=str)
        return 3

    model_chains = [_chain_by_id(ms, m1), _chain_by_id(ms, m2)]
    native_chains = [_chain_by_id(ns, n1), _chain_by_id(ns, n2)]
    # Mirror run_on_all_native_interfaces: protein interfaces use the DockQ path
    # (small_molecule=False); the default True path runs calc_sym_corrected_lrmsd
    # and returns None for protein-protein interfaces.
    small_molecule = bool(getattr(native_chains[0], "is_het", False) or
                          getattr(native_chains[1], "is_het", False))
    info = run_on_chains(tuple(model_chains), tuple(native_chains),
                         small_molecule=small_molecule)
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
           "native_chain1": n1, "native_chain2": n2, "chain_map": cmap,
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
    print(json.dumps(out, indent=2, default=str))
    if a.out:
        with open(a.out, "w") as fp:
            json.dump(out, fp, indent=2, default=str)
    return 0 if dockq is not None else 2


if __name__ == "__main__":
    sys.exit(main())
