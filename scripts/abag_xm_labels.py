#!/usr/bin/env python3
"""Phase 4 label orchestration driver.

Runs the Phase 4 label scripts over a completed (target, gen) fold dir and writes
labels.json (one record per sample + fold-level PSS / basin clusters).

Per-sample (rank r, CIF): DockQ on the ARK-declared Ab-Ag interface (D6, via
abag_xm_dockq_interface -- not the auto-mapper GlobalDockQ average), PAE metrics,
epitope Jaccard, interface lDDT, per-CDR RMSD. Per-fold: pairwise DockQ/TM matrix
+ PSS, basin clustering.

Per-sample pTM is read from results.json[0]["all_runs"][rank]["ptm"]; the PAE npz
is <target>_model_<rank>_pae.npz (uniform across ranks, incl. rank 0).

The declared interface chains (fold_auth_chain_id_1/2) are resolved from the
manifest by pdb_id; if --chain1/--chain2 are passed they override the lookup.

Usage:
    PYTHONPATH=<wt> python3 scripts/abag_xm_labels.py <results_dir> <native.cif> <fold.yaml> [--n_samples N] [--out labels.json] [--per_sample_only]

--per_sample_only skips the pairwise matrix + basin clustering stages (quadratic
in N); labels.json keeps both keys with a "_skipped" marker so the schema is
otherwise unchanged.
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
MANIFEST = ROOT / "docs" / "implementation-parity-data" / "abag-xm-targets.parquet"


def _declared_chains(target):
    """Look up fold_auth_chain_id_1/2 from the manifest by pdb_id (D6)."""
    try:
        import pandas as pd
        df = pd.read_parquet(MANIFEST)
        row = df[df["pdb_id"] == target]
        if len(row) == 0:
            return None, None
        r = row.iloc[0]
        return r["fold_auth_chain_id_1"], r["fold_auth_chain_id_2"]
    except Exception:
        return None, None


def _run(script, args, out_path=None):
    """Run a label script; if out_path given, pass --out and read it back, else parse stdout JSON."""
    cmd = [sys.executable, str(SCRIPTS / f"{script}.py"), *args]
    if out_path:
        cmd += ["--out", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"_error": r.stderr.strip()[:400]}
    if out_path:
        try:
            return json.loads(Path(out_path).read_text())
        except Exception as e:
            return {"_error": f"read out failed: {e}; stderr={r.stderr.strip()[:200]}"}
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"_raw": r.stdout.strip()[:400]}


def _samples(results_dir, target):
    st = results_dir / "structures"
    samples = []
    r0 = st / f"{target}.cif"
    if r0.exists():
        samples.append((0, r0))
    for cif in sorted(st.glob(f"{target}_model_*.cif"), key=lambda p: p.name):
        try:
            r = int(cif.stem.split("_model_")[1])
            samples.append((r, cif))
        except Exception:
            continue
    return samples


def _ptm_by_rank(results_dir):
    """Read results.json[0]['all_runs'] -> {rank: ptm}."""
    rj = results_dir / "results.json"
    if not rj.exists():
        return {}
    try:
        data = json.loads(rj.read_text())
        if isinstance(data, list) and data:
            data = data[0]
        runs = data.get("all_runs", [])
        return {rec.get("rank"): rec.get("ptm") for rec in runs
                if isinstance(rec, dict) and rec.get("rank") is not None}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("native")
    ap.add_argument("yaml")
    ap.add_argument("--n_samples", type=int, default=0,
                    help="0 = all; else limit per-sample loop and pairwise matrix to first N")
    ap.add_argument("--chain1", default=None,
                    help="override manifest fold_auth_chain_id_1 (antibody side)")
    ap.add_argument("--chain2", default=None,
                    help="override manifest fold_auth_chain_id_2 (antigen side)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pair_workers", type=int, default=4,
                    help="parallel workers inside the pairwise DockQ matrix, which is 62%% of a "
                         "fold's label cost. Total processes are (label workers) x this.")
    ap.add_argument("--per_sample_only", action="store_true",
                    help="skip pairwise matrix + basin clustering (quadratic in N)")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    target = rd.name.split("results_")[1]
    c1, c2 = a.chain1, a.chain2
    if c1 is None or c2 is None:
        mc1, mc2 = _declared_chains(target)
        c1 = c1 or mc1
        c2 = c2 or mc2
    samples = _samples(rd, target)
    if a.n_samples and a.n_samples > 0:
        samples = samples[:a.n_samples]
    st = rd / "structures"
    ptm_map = _ptm_by_rank(rd)

    recs = []
    for rank, cif in samples:
        rec = {"target": target, "rank": rank, "cif": str(cif)}
        if c1 is not None and c2 is not None:
            rec["dockq"] = _run("abag_xm_dockq_interface",
                               [str(cif), a.native, str(c1), str(c2)])
        else:
            # Manifest lookup failed -- fall back to the auto-mapper average and
            # flag it so the label is never silently the wrong quantity.
            dq = _run("opendde_dockq", [str(cif), a.native])
            dq["_fallback_global_average"] = True
            rec["dockq"] = dq
        rec["epitope_jaccard"] = _run("abag_xm_epitope_jaccard", [str(cif), a.native, a.yaml])
        rec["interface_lddt"] = _run("abag_xm_interface_lddt", [str(cif), a.native, a.yaml])
        rec["cdr_rmsd"] = _run("abag_xm_cdr_rmsd", [str(cif), a.native, a.yaml])
        pae_npz = st / f"{target}_model_{rank}_pae.npz"
        ptm = ptm_map.get(rank)
        if pae_npz.exists() and ptm is not None:
            rec["pae_metrics"] = _run("abag_pae_metrics",
                                       [str(cif), str(pae_npz), a.yaml, str(ptm)])
        else:
            rec["pae_metrics"] = {"_skipped": f"pae={pae_npz.exists()} ptm={ptm is not None}"}
        recs.append(rec)

    # per-fold: pairwise matrix + PSS, then basin clustering on that matrix
    if a.per_sample_only:
        pm = {"_skipped": "per_sample_only"}
        bc = {"_skipped": "per_sample_only"}
    else:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            matrix_out = td / "matrix.json"
            # Pass the pairwise worker count explicitly. It was defaulting to 4 inside
            # abag_xm_pairwise_matrix, so the real process count was label_workers x 4 -- 16 on a
            # host running 4 label workers, which nobody chose and which no budget bounded. It
            # happens to fit 32 cores next to 4 folds; it would not fit a host where label workers
            # were raised further. Default preserved at 4 so this changes nothing today, but the
            # knob now exists and the resulting total is printed rather than implied.
            pm = _run("abag_xm_pairwise_matrix", [str(rd), target,
                       f"--n_samples={len(samples)}",
                       f"--n_workers={a.pair_workers}"], out_path=matrix_out)
            bc_out = td / "basin.json"
            bc = _run("abag_xm_basin_clust", [str(matrix_out)], out_path=bc_out)

    out = {"target": target, "results_dir": str(rd), "native": a.native,
           "yaml": a.yaml, "n_samples": len(samples),
           "samples": recs, "pairwise_matrix": pm, "basin_clust": bc}
    print(json.dumps(out, indent=2))
    if a.out:
        # Atomic, because the trailing labeler treats the mere existence of labels.json as
        # "this fold is done" and never revisits it. A kill during a plain write leaves a
        # truncated file that every later scan skips, so the cell would drop out of the rung
        # with no error anywhere. ~237 KB per file, so the window is small but real.
        tmp = Path(str(a.out) + ".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
