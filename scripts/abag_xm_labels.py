#!/usr/bin/env python3
"""Phase 4 label orchestration driver.

Runs the Phase 4 label scripts over a completed (target, gen) fold dir and writes
labels.json (one record per sample + fold-level PSS / basin clusters).

Per-sample (rank r, CIF): DockQ, PAE metrics, epitope Jaccard, interface lDDT,
per-CDR RMSD. Per-fold: pairwise DockQ/TM matrix + PSS, basin clustering.

Per-sample pTM is read from results.json[0]["all_runs"][rank]["ptm"]; the PAE npz
is <target>_model_<rank>_pae.npz (uniform across ranks, incl. rank 0).

Usage:
    PYTHONPATH=<wt> python3 scripts/abag_xm_labels.py <results_dir> <native.cif> <fold.yaml> [--n_samples N] [--out labels.json]
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rd = Path(a.results_dir)
    target = rd.name.split("results_")[1]
    samples = _samples(rd, target)
    if a.n_samples and a.n_samples > 0:
        samples = samples[:a.n_samples]
    st = rd / "structures"
    ptm_map = _ptm_by_rank(rd)

    recs = []
    for rank, cif in samples:
        rec = {"target": target, "rank": rank, "cif": str(cif)}
        rec["dockq"] = _run("opendde_dockq", [str(cif), a.native])
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
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        matrix_out = td / "matrix.json"
        pm = _run("abag_xm_pairwise_matrix", [str(rd), target,
                   f"--n_samples={len(samples)}"], out_path=matrix_out)
        bc_out = td / "basin.json"
        bc = _run("abag_xm_basin_clust", [str(matrix_out)], out_path=bc_out)

    out = {"target": target, "results_dir": str(rd), "native": a.native,
           "yaml": a.yaml, "n_samples": len(samples),
           "samples": recs, "pairwise_matrix": pm, "basin_clust": bc}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
