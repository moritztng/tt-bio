#!/usr/bin/env python3
"""ABAG-Rank adapter for the AbAg-XM campaign (Phase 5 prep).

Converts a completed AbAg-XM fold directory (one (target, generator) pair,
N=50 samples) into the AF3-style layout ABAG-Rank's `preprocess_inference.py`
expects, then runs preprocess (+ optionally inference) and emits per-sample
scores.

Our fold layout (harness-written):
    <fold_dir>/
        results.json              # list[1] with "all_runs": [per-rank conf dicts]
        structures/
            <target>.cif          # rank-0
            <target>_model_1.cif   # rank-k (k=1..N-1)
            <target>_model_k_pae.npz

ABAG-Rank expected layout (AF3-style):
    <input_dir>/
        ranking_scores.csv        # seed,sample,ranking_score
        seed-0_sample-k/
            model.cif
            confidences.json       # {ranking_score, ptm, iptm}

Chain IDs differ per generator (protenix renames H->B/L->C; boltz2 preserves
H/L). We map model chains to the YAML's A/H/L by exact sequence match (rank-0),
then reuse that mapping for all samples in the fold (chain IDs are stable
within a fold).

Usage:
    python scripts/abag_xm_abagrank_adapter.py \
        --fold_dir ~/abag_xm/tier_a/protenix_v2/protenix_results_21av \
        --target 21av \
        --yaml examples/abag_xm/21av.yaml \
        --out_h5 /tmp/21av_protenix_abagrank.h5 \
        --run_inference --out_scores /tmp/21av_protenix_abagrank_scores.json
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, csv
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Polypeptide import PPBuilder


def _yaml_sequences(yaml_path):
    """Return {chain_id: sequence} for the protein chains in the YAML."""
    import yaml
    seqs = {}
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    for entry in d.get("sequences", []):
        prot = entry.get("protein")
        if prot:
            seqs[prot["id"]] = prot["sequence"]
    return seqs


def _chain_sequences(cif_path):
    """Return {model_chain_id: sequence} parsed from a CIF (one-letter)."""
    parser = MMCIFParser(QUIET=True)
    s = parser.get_structure("m", str(cif_path))
    ppb = PPBuilder()
    out = {}
    for ch in s[0]:
        pps = ppb.build_peptides(ch)  # list of Polypeptide
        one = "".join(str(pp.get_sequence()) for pp in pps)
        out[ch.id] = one
    return out


def _map_chains(model_seqs, yaml_seqs):
    """Map model chain IDs to YAML roles {A, H, L} by exact sequence match.

    Falls back to length match if exact match fails (e.g., missing residues).
    Supports nanobody/VHH targets (no L chain in YAML) — light_id is None then.
    Returns (antigen_model_id, heavy_model_id, light_model_id_or_None).
    """
    # Exact match first
    role_for_yaml = {}
    for yid, yseq in yaml_seqs.items():
        for mid, mseq in model_seqs.items():
            if mseq and mseq == yseq:
                role_for_yaml[yid] = mid
                break
    # Length fallback for any unresolved
    for yid, yseq in yaml_seqs.items():
        if yid in role_for_yaml:
            continue
        yl = len(yseq)
        best, best_diff = None, 1e9
        for mid, mseq in model_seqs.items():
            if mid in role_for_yaml.values():
                continue
            d = abs(len(mseq) - yl)
            if d < best_diff:
                best, best_diff = mid, d
        if best is not None:
            role_for_yaml[yid] = best
    # Resolve roles: A=antigen, H=heavy, L=light (YAML convention; L optional for VHH)
    ag = role_for_yaml.get("A")
    heavy = role_for_yaml.get("H")
    light = role_for_yaml.get("L")  # None for nanobody/VHH targets
    if ag is None or heavy is None:
        raise RuntimeError(f"chain mapping incomplete: {role_for_yaml} "
                           f"model_seqs={ {k: len(v) for k,v in model_seqs.items()} } "
                           f"yaml_seqs={ {k: len(v) for k,v in yaml_seqs.items()} }")
    return ag, heavy, light


def _sample_cifs(fold_dir, target, n_expected):
    """Return ordered list of sample CIF paths [rank0, rank1, ..., rankN-1]."""
    sdir = fold_dir / "structures"
    cifs = [sdir / f"{target}.cif"]  # rank-0
    for k in range(1, n_expected):
        cifs.append(sdir / f"{target}_model_{k}.cif")
    # Verify existence
    for c in cifs:
        if not c.exists():
            raise FileNotFoundError(f"missing sample CIF: {c}")
    return cifs


def _sample_pae_npz(sdir, target, k):
    """Return the PAE npz path for sample rank k.

    Naming: <target>_model_<k>_pae.npz for ALL k (including rank-0, whose CIF
    is <target>.cif but whose PAE is <target>_model_0_pae.npz).
    """
    p = sdir / f"{target}_model_{k}_pae.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing PAE npz for sample {k}: {p}")
    return p


def _build_shim(fold_dir, target, all_runs, cifs, work_dir):
    """Build the AF3-style shim layout in work_dir. Returns (seed, n_samples).

    Injects the per-sample PAE matrix (from <target>_model_k_pae.npz) into each
    confidences.json under the 'pae' key, since ABAG-Rank's preprocess reads
    `data.get('pae', [])` from confidences.json (AF3 convention) and our PAE
    lives in sidecar npz files, not the confidences json.
    """
    import numpy as np
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    sdir = fold_dir / "structures"
    seed = 0
    # ranking_scores.csv
    with open(work / "ranking_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "sample", "ranking_score"])
        for k, run in enumerate(all_runs):
            rs = run.get("confidence_score", run.get("iptm", 0.0))
            w.writerow([seed, k, float(rs)])
    # per-sample dirs
    for k, (run, cif) in enumerate(zip(all_runs, cifs)):
        sdir_k = work / f"seed-{seed}_sample-{k}"
        sdir_k.mkdir(exist_ok=True)
        # model.cif as a symlink (avoid copy)
        link = sdir_k / "model.cif"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(cif.resolve())
        # PAE matrix from npz -> nested list for JSON
        npz = np.load(_sample_pae_npz(sdir, target, k))
        pae = npz["pae"].astype(np.float32)
        pae_list = pae.tolist()
        # confidences.json (with pae matrix injected)
        conf = {
            "ranking_score": float(run.get("confidence_score", run.get("iptm", 0.0))),
            "ptm": float(run.get("ptm", 0.0)),
            "iptm": float(run.get("iptm", 0.0)),
            "pae": pae_list,
        }
        with open(sdir_k / "confidences.json", "w") as f:
            json.dump(conf, f)
    return seed, len(all_runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out_h5", required=True)
    ap.add_argument("--abagrank_dir", default=os.path.expanduser("~/ABAG-Rank"))
    ap.add_argument("--add_esm", action="store_true")
    ap.add_argument("--run_inference", action="store_true")
    ap.add_argument("--out_scores", default=None)
    ap.add_argument("--keep_shim", action="store_true",
                    help="Keep the shim dir (for debugging)")
    args = ap.parse_args()

    fold_dir = Path(args.fold_dir)
    abag = Path(args.abagrank_dir)
    yaml_seqs = _yaml_sequences(args.yaml)

    # 1. read results.json -> all_runs
    with open(fold_dir / "results.json") as f:
        results = json.load(f)
    assert isinstance(results, list) and len(results) == 1, \
        f"unexpected results.json shape: {type(results)} len {len(results) if isinstance(results,list) else None}"
    all_runs = results[0]["all_runs"]
    n = len(all_runs)
    print(f"[adapter] target={args.target} n_samples={n} add_esm={args.add_esm}")

    # 2. list sample CIFs
    cifs = _sample_cifs(fold_dir, args.target, n)

    # 3. map model chain IDs (rank-0)
    model_seqs = _chain_sequences(cifs[0])
    ag_id, heavy_id, light_id = _map_chains(model_seqs, yaml_seqs)
    print(f"[adapter] chain map: antigen={ag_id} heavy={heavy_id} "
          f"light={light_id} (model_seqs={ {k: len(v) for k,v in model_seqs.items()} })")

    # 4. build shim
    work_dir = tempfile.mkdtemp(prefix=f"abagrank_shim_{args.target}_")
    seed, ns = _build_shim(fold_dir, args.target, all_runs, cifs, work_dir)
    print(f"[adapter] shim built at {work_dir} ({ns} samples)")

    try:
        # 5. preprocess
        pp = abag / "preprocess_inference.py"
        # nanobody/VHH: single heavy chain (no light); else H,L
        ab_chains = heavy_id if light_id is None else f"{heavy_id},{light_id}"
        cmd = [sys.executable, str(pp),
               "--input_dir", work_dir,
               "--output_h5", args.out_h5,
               "--antibody_chains", ab_chains,
               "--antigen_chains", ag_id,
               "--target_id", args.target,
               "--run_dirs", ""]
        if args.add_esm:
            cmd.append("--add_esm")
        print(f"[adapter] preprocess: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-2000:])
        if r.returncode != 0:
            print(r.stderr[-3000:], file=sys.stderr)
            sys.exit(r.returncode)

        # 6. inference (optional)
        if args.run_inference:
            ckpt = abag / "checkpoints" / ("ABAG_Rank_checkpoint.pt" if args.add_esm
                                          else "ABAG_Rank_noESM_checkpoint.pt")
            cfg = abag / "configs" / ("config_ABAG_rank.yaml" if args.add_esm
                                     else "config_ABAG_rank_no_esm.yaml")
            inf_out = tempfile.mkdtemp(prefix=f"abagrank_inf_{args.target}_")
            icmd = [sys.executable, str(abag / "run_inference.py"),
                    "--h5_file", args.out_h5,
                    "--checkpoint", str(ckpt),
                    "--config", str(cfg),
                    "--output_dir", inf_out,
                    "--target_id", args.target]
            print(f"[adapter] inference: {' '.join(icmd)}")
            ir = subprocess.run(icmd, capture_output=True, text=True)
            print(ir.stdout[-2000:])
            if ir.returncode != 0:
                print(ir.stderr[-3000:], file=sys.stderr)
                sys.exit(ir.returncode)
            # parse scores from inf_out (run_inference writes a ranking file)
            scores = _parse_inference_output(inf_out, args.target)
            print(f"[adapter] inference done; {len(scores)} scores")
            if args.out_scores:
                with open(args.out_scores, "w") as f:
                    json.dump(scores, f, indent=2)
                print(f"[adapter] wrote {args.out_scores}")
    finally:
        if not args.keep_shim:
            shutil.rmtree(work_dir, ignore_errors=True)


def _parse_inference_output(inf_out, target):
    """Parse run_inference.py output into {sample_name: model_predicted_dockq}.

    run_inference.py writes <target>_ranked_by_model.csv with columns:
    target_id, sample_name, af3_ranking_score, model_predicted_dockq,
    combined_score, rank.
    """
    inf_out = Path(inf_out)
    # Prefer the ranked_by_model CSV (the ranker's own scores)
    for p in sorted(inf_out.rglob(f"{target}_ranked_by_model.csv")):
        import csv as _csv
        with open(p) as f:
            r = _csv.DictReader(f)
            rows = list(r)
        if rows and "model_predicted_dockq" in rows[0]:
            return {row["sample_name"]: float(row["model_predicted_dockq"])
                    for row in rows}
    # Fallback: any CSV with a score-like column
    for p in inf_out.rglob("*.csv"):
        import csv as _csv
        with open(p) as f:
            r = _csv.DictReader(f)
            rows = list(r)
        if not rows:
            continue
        for col in ("model_predicted_dockq", "predicted_dockq", "score"):
            if col in rows[0]:
                key = "sample_name" if "sample_name" in rows[0] else "sample"
                return {row[key]: float(row[col]) for row in rows}
    # Fallback: JSON
    for p in inf_out.rglob("*.json"):
        with open(p) as f:
            return json.load(f)
    print(f"[adapter] WARNING: could not parse inference output; listing {inf_out}:")
    for p in inf_out.rglob("*"):
        print("  ", p)
    return {}


if __name__ == "__main__":
    main()
