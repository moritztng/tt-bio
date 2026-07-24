"""DockQ of an OpenDDE antibody-antigen fold vs the public ground-truth complex.

Uses the reference DockQ tool (DockQ==2.1.3, the Björn Wallner lab implementation that
defines the DockQ metric -- installed as an eval-time requirement into the run venv; not
a project runtime dependency). DockQ enumerates every native chain-pair interface,
maps model->native chains by sequence identity, and reports per-interface DockQ
(Fnat / iRMS / LRMS) plus GlobalDockQ (mean over native interfaces). For a Fab+antigen
complex the antibody-antigen interfaces are antigen--heavy and antigen--light; the
heavy--light interface is the internal Fab docking and is reported for completeness.

    PYTHONPATH=<worktree> python3 scripts/opendde_dockq.py <model.cif> <native.cif> [--out json]

Model CIF chain IDs (Protenix writes A,B,C,... in input order) need not match the native
(A,H,L here) -- DockQ maps by sequence. The native is examples/ground_truth_structures/9dsg.cif
(PDB 9dsg, SARS-CoV-2 RBD + neutralizing Fab, from the OpenDDE 2026ARK_AB benchmark set).
"""
import argparse, json, sys
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces, group_chains, get_all_chain_maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("native")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ms = load_PDB(a.model)
    ns = load_PDB(a.native)
    mc, nc = [c.id for c in ms], [c.id for c in ns]
    clusters, rev = group_chains(ms, ns, mc, nc, allowed_mismatches=0)
    cmap = next(get_all_chain_maps(clusters, {}, rev, mc, nc))
    res, total = run_on_all_native_interfaces(ms, ns, chain_map=cmap)
    out = {"model": a.model, "native": a.native, "chain_map": cmap,
           "total_dockq": total, "n_interfaces": len(res),
           "global_dockq": (total / len(res)) if res else 0.0,
           "interfaces": {}}
    for k, v in res.items():
        out["interfaces"][k] = {m: v.get(m) for m in
                                ("DockQ", "fnat", "fnonnat", "iRMSD", "LRMSD", "clashes")}
    print(json.dumps(out, indent=2, default=str))
    if a.out:
        with open(a.out, "w") as fp:
            json.dump(out, fp, indent=2, default=str)
    # exit nonzero if no interface had a real DockQ (e.g. no native contacts found)
    return 0 if any(v.get("DockQ") is not None for v in res.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
