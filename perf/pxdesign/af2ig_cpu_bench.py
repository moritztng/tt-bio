"""Host-CPU cost of PXDesign's AF2-IG filter stage, at the pipeline's real shape.

PXDesign runs AF2-IG as a fresh subprocess per invocation
(`pxdbench/tools/af2/main_af2_complex.py`, then `main_af2_monomer.py`), looping over
the designs in its `data_list`. Each design re-runs `prep_inputs` and one
`predict(num_recycles=3)`. So the stage cost splits into a fixed term paid once per
invocation (AF2 parameter load + XLA compile) and a marginal term paid per design.
The marginal is the number that matters: `state/pxdesign-stage-placement.md` defines
the pipeline comparison on marginal cost per design, and the N=1 cell is ~97% startup.

Config is upstream's production `af2` block verbatim
(`pxdbench/pxd_configs/eval.py:53` -- use_multimer False, model_ids [0],
use_initial_guess True, use_initial_atom_pos False, use_binder_template True) and
`num_recycles=3`, hardcoded at `main_af2_complex.py:76,136`.

This calls ColabDesign directly rather than upstream's `main_af2_*.py` because
`pxdbench/tools/__init__.py` imports protenix, which needs CUDA and is not installed
in the CPU-only JAX venv. The model call and its per-design featurization are
reproduced exactly; upstream's per-design post-processing (renumber, permutation,
Kabsch RMSD -- all pure-numpy host work, small) is excluded, which biases this
measurement DOWN, i.e. in favour of "no port needed".

Binder geometry is synthetic: a copy of the target crop's first 80 residues,
translated clear of the target. AF2 with a fixed recycle count has no data-dependent
control flow and no early exit, so its cost is a function of (complex length,
recycles, model) and not of residue identities or coordinate values -- the same
argument `state/pxdesign-perf.md` used for the diffusion sampler and
`perf/pxdesign/cpu_mpnn_bench.py` used for ProteinMPNN.
"""
import argparse, gc, json, os, time

import numpy as np


AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def read_crop_cif(path):
    """Parse one of perf/pxdesign/targets/laczc_*.cif into ordered residues.

    These are written by `make_targets.py` with a fixed atom_site header, one chain,
    renumbered 1..N. Resolve columns by header name, never by index
    (`design-model-output-validation-not-folding-invariants`).
    """
    cols, residues, order = [], {}, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("_atom_site."):
                cols.append(line.split(".", 1)[1].strip())
                continue
            if not line.startswith("ATOM"):
                continue
            f = line.split()
            rec = dict(zip(cols, f))
            key = int(rec["label_seq_id"])
            if key not in residues:
                residues[key] = {"comp": rec["label_comp_id"], "atoms": []}
                order.append(key)
            residues[key]["atoms"].append(
                (rec["label_atom_id"], rec["type_symbol"],
                 float(rec["Cartn_x"]), float(rec["Cartn_y"]), float(rec["Cartn_z"])))
    return [residues[k] for k in order]


def write_complex_pdb(path, target_res, binder_res, shift):
    """Target as chain A, binder as chain B translated by `shift`."""
    serial = 0
    with open(path, "w") as fh:
        for chain, res_list, dx in (("A", target_res, 0.0), ("B", binder_res, shift)):
            for i, res in enumerate(res_list, start=1):
                for (name, elem, x, y, z) in res["atoms"]:
                    serial += 1
                    aname = name if len(name) >= 4 else " %-3s" % name
                    fh.write("ATOM  %5d %s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                             % (serial, aname, res["comp"], chain, i,
                                x + dx, y, z, elem.rjust(2)))
            fh.write("TER\n")
        fh.write("END\n")
    return serial


def seq_of(res_list):
    return "".join(AA3TO1.get(r["comp"], "A") for r in res_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-dir", default="perf/pxdesign/targets")
    ap.add_argument("--target", type=int, nargs="+", default=[128])
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--ndesign", type=int, default=4,
                    help="designs per invocation; design 0 is cold (XLA compile)")
    ap.add_argument("--params", default=os.path.expanduser("~/pxd_tool_weights/af2"))
    ap.add_argument("--stage", default="complex", choices=["complex", "monomer"])
    ap.add_argument("--work", default="/tmp/af2ig_bench_work")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax
    from colabdesign import clear_mem, mk_afdesign_model

    rec = {
        "host": os.uname().nodename,
        "stage": args.stage,
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "nproc": os.cpu_count(),
        "loadavg_start": open("/proc/loadavg").read().split()[:3],
        "num_recycles": 3,
        "af2_cfg": {"use_multimer": False, "model_ids": [0], "use_initial_guess": True,
                    "use_initial_atom_pos": False, "use_binder_template": True},
        "binder_length": args.binder,
        "ndesign": args.ndesign,
        "cells": [],
    }
    os.makedirs(args.work, exist_ok=True)

    for tgt in args.target:
        cif = os.path.join(args.targets_dir, "laczc_%d.cif" % tgt)
        res = read_crop_cif(cif)
        assert len(res) == tgt, "%s has %d residues, expected %d" % (cif, len(res), tgt)
        target_res = res
        binder_res = res[: args.binder]
        pdb = os.path.join(args.work, "cell_%d.pdb" % tgt)
        natoms = write_complex_pdb(pdb, target_res, binder_res, shift=60.0)
        binder_seq = seq_of(binder_res)
        ntok = tgt + args.binder

        clear_mem()
        gc.collect()
        t0 = time.perf_counter()
        model = mk_afdesign_model(
            protocol="binder", num_recycles=3, data_dir=args.params,
            use_multimer=False, use_initial_guess=True, use_initial_atom_pos=False)
        t_model = time.perf_counter() - t0

        prep_s, pred_s = [], []
        for d in range(args.ndesign):
            t0 = time.perf_counter()
            model.prep_inputs(pdb_filename=pdb, chain="A", binder_chain="B",
                              use_binder_template=True, rm_target_seq=True,
                              rm_target_sc=False, rm_template_ic=True)
            t1 = time.perf_counter()
            model.predict(seq=binder_seq, models=[0], num_recycles=3, verbose=False)
            t2 = time.perf_counter()
            prep_s.append(t1 - t0)
            pred_s.append(t2 - t1)
            log = model.aux["log"]
            print("  target %4d (%4d tok) design %d/%d: prep %7.2f s  predict %8.2f s"
                  "   plddt %.3f i_ptm %.3f"
                  % (tgt, ntok, d, args.ndesign - 1, prep_s[-1], pred_s[-1],
                     log["plddt"], log.get("i_ptm", float("nan"))), flush=True)

        per_design = [p + q for p, q in zip(prep_s, pred_s)]
        warm = sorted(per_design[1:]) if len(per_design) > 1 else []
        cell = {
            "target_residues": tgt, "n_tokens": ntok, "complex_atoms": natoms,
            "model_construct_s": t_model,
            "cold_design_s": per_design[0],
            "prep_s": prep_s, "predict_s": pred_s, "per_design_s": per_design,
            "warm_per_design_s": warm,
            "marginal_s_per_design": warm[len(warm) // 2] if warm else None,
            "aa_spread_s": (warm[-1] - warm[0]) if warm else None,
            "aa_spread_pct": (100.0 * (warm[-1] - warm[0]) / warm[len(warm) // 2]
                              if warm else None),
            "fixed_s": t_model + (per_design[0] - (warm[len(warm) // 2] if warm else 0.0)),
            "loadavg": open("/proc/loadavg").read().split()[:3],
        }
        rec["cells"].append(cell)
        print("target %4d (%4d tok): marginal %s s/design, cold %.2f, construct %.2f, "
              "A/A spread %s"
              % (tgt, ntok,
                 ("%.2f" % cell["marginal_s_per_design"]) if warm else "n/a",
                 cell["cold_design_s"], t_model,
                 ("%.3f s = %.2f%%" % (cell["aa_spread_s"], cell["aa_spread_pct"]))
                 if warm else "n/a"), flush=True)

        if args.out:                      # write after every rung: partials survive
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(rec, fh, indent=1)
            print("wrote", args.out, flush=True)

        del model
        clear_mem()
        gc.collect()


if __name__ == "__main__":
    main()
