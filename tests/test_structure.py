import argparse
import sys
from itertools import permutations
from pathlib import Path

import gemmi
import numpy as np


def _kabsch_deviations(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Per-point residual (Å) after optimal rigid superposition of P onto Q."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return np.linalg.norm(Pc @ R.T - Qc, axis=1)


def _tm_score(deviations, L_target: int) -> float:
    """TM-score for CA deviations (Angstrom) after superposition, normalized by the
    ground-truth length L_target. Length-independent measure of topological correctness:
    >0.5 means the same fold, ~0.17 is random. Complements RMSD, which a single misplaced
    flexible tail can inflate on an otherwise-correct fold."""
    d = np.asarray(deviations, dtype=float)
    d0 = max(0.5, 1.24 * (L_target - 15) ** (1.0 / 3.0) - 1.8)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)) * len(d) / L_target)


def get_ca_atoms(cif_path: str):
    """Extract CA coordinates per chain, keyed by entity position (label_seq_id).

    label_seq_id is the 1-based position in the entity sequence, matching the
    sequential residue numbering tt-bio writes for predictions. When a file omits
    label_seq_id (e.g. ESMFold2's minimal cif), fall back to a per-chain running
    index — identical for two files that both list residues in sequence order.

    Parsed with gemmi (not biopython MMCIFParser) so minimal predicted cifs that
    omit _atom_site.occupancy still load. Returns {chain_id: {pos(int): xyz(np)}}.
    """
    st = gemmi.read_structure(cif_path)
    st.remove_alternative_conformations()
    chains = {}
    for chain in st[0]:
        ca_by_pos, running = {}, 0
        for res in chain:
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            running += 1
            pos = res.label_seq if res.label_seq is not None else running
            ca_by_pos[int(pos)] = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
        if ca_by_pos:
            chains[chain.name] = ca_by_pos
    return chains


def _find_results_dir(protein_name: str) -> Path:
    """Locate the tt-bio predict output dir for protein_name in the cwd.

    Model-agnostic: matches the model-named layout <model>_results_<name> (e.g.
    protenix_results_prot, boltz2_results_trpcage, opendde_results_1ahw_abag)
    and the neutral results_<name> form, so the harness never hardcodes a model
    prefix. Raises if no results dir is found."""
    candidates = sorted(Path(".").glob(f"*results_{protein_name}"))
    if not candidates:
        raise FileNotFoundError(
            f"No results dir found for {protein_name} " f"(looked for *results_{protein_name})")
    return candidates[0]


def compute_rmsd(protein_name: str, model_idx: int = 0, results_dir: Path | None = None):
    """Compute RMSD with optimal chain matching (handles partial coverage)."""
    if results_dir is None:
        results_dir = _find_results_dir(protein_name)
    if model_idx == 0:
        pred_file = results_dir / "structures" / f"{protein_name}.cif"
    else:
        pred_file = results_dir / "structures" / f"{protein_name}_model_{model_idx}.cif"
    truth_file = Path(f"examples/ground_truth_structures/{protein_name}.cif")
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    if not truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {truth_file}")
    
    pred_chains = get_ca_atoms(str(pred_file))
    truth_chains = get_ca_atoms(str(truth_file))
    
    # Filter out empty chains
    pred_chains = {k: v for k, v in pred_chains.items() if v}
    truth_chains = {k: v for k, v in truth_chains.items() if v}
    
    print(f"Predicted chains: {[(c, len(pred_chains[c])) for c in sorted(pred_chains)]}")
    print(f"Ground truth chains: {[(c, len(truth_chains[c])) for c in sorted(truth_chains)]}")
    
    pred_ids = sorted(pred_chains)
    truth_ids = sorted(truth_chains)
    
    if len(pred_ids) != len(truth_ids):
        raise ValueError(
            f"Chain count mismatch: {len(pred_ids)} predicted vs {len(truth_ids)} ground truth"
        )
    
    # Try all permutations of truth chains, pick lowest RMSD
    best_rmsd = float('inf')
    best_matching = None
    best_n_common = 0
    
    def get_atoms_for_pair(pred_by_id, truth_by_id):
        """Match residues by rank when lengths are equal, by label_seq_id otherwise."""
        pred_list = list(pred_by_id.values())
        truth_list = list(truth_by_id.values())
        if len(pred_list) == len(truth_list):
            # Same length: match residue-by-residue in sequential order
            return pred_list, truth_list
        # Different lengths (partial coverage): match by label_seq_id
        common = sorted(set(pred_by_id) & set(truth_by_id))
        return [pred_by_id[k] for k in common], [truth_by_id[k] for k in common]

    for truth_perm in permutations(truth_ids):
        matching = list(zip(pred_ids, truth_perm))
        
        # Collect CA atoms across all chain pairs
        atoms_pred, atoms_truth = [], []
        for pred_id, truth_id in matching:
            ap, at = get_atoms_for_pair(pred_chains[pred_id], truth_chains[truth_id])
            atoms_pred.extend(ap)
            atoms_truth.extend(at)
        
        if len(atoms_pred) < 3:
            continue

        rmsd = float(np.sqrt((_kabsch_deviations(np.array(atoms_pred), np.array(atoms_truth)) ** 2).mean()))
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_matching = matching
            best_n_common = len(atoms_pred)
    
    if best_matching is None:
        raise ValueError("No valid chain matching found")
    
    total_pred = sum(len(v) for v in pred_chains.values())
    total_truth = sum(len(v) for v in truth_chains.values())

    # Recompute matched atoms for the winning chain matching, then derive per-CA
    # deviations under its rotation/translation for a TM-score (RMSD alone can be
    # inflated by one flexible tail on an otherwise-correct fold).
    atoms_pred, atoms_truth = [], []
    for pred_id, truth_id in best_matching:
        ap, at = get_atoms_for_pair(pred_chains[pred_id], truth_chains[truth_id])
        atoms_pred.extend(ap)
        atoms_truth.extend(at)
    dev = _kabsch_deviations(np.array(atoms_pred), np.array(atoms_truth))
    best_rmsd = float(np.sqrt((dev ** 2).mean()))
    tm = _tm_score(dev, total_truth)

    print(f"\nBest chain matching ({best_n_common} common residues "
          f"of {total_pred} predicted / {total_truth} ground truth):")
    for pred_id, truth_id in best_matching:
        ap, _ = get_atoms_for_pair(pred_chains[pred_id], truth_chains[truth_id])
        print(f"  {pred_id} -> {truth_id} ({len(ap)} common residues)")

    print(f"\n{'='*50}")
    print(f"Protein: {protein_name}, Model: {model_idx}")
    print(f"RMSD: {best_rmsd:.4f} Å   TM-score: {tm:.4f}")
    print(f"{'='*50}\n")

    return best_rmsd, tm


def _num_models(protein_name: str, results_dir: Path | None = None) -> int:
    """Number of written samples (best is model 0, ranked by confidence)."""
    if results_dir is None:
        results_dir = _find_results_dir(protein_name)
    d = results_dir / "structures"
    return 1 + sum(1 for _ in d.glob(f"{protein_name}_model_*.cif")) if d.exists() else 1


def evaluate(protein_name: str, max_rmsd: float | None = None, min_tm: float | None = None, results_dir: Path | None = None):
    """Ground-truth accuracy gate for a foldable target.

    tt-bio writes the best-confidence sample as ``{name}.cif`` (model 0) and the
    rest as ``{name}_model_{rank}.cif`` (ranked by confidence). The RELEASE GATE
    scores model 0 — the confidence-selected best, NOT a raw sample-0 — because
    that is the structure a user actually gets. We also report the best/worst over
    all samples so a poorly-calibrated confidence head (best-conf far from oracle-
    best) is visible.

    Self-consistency (seed-vs-reference RMSD) is NOT a substitute: it passes even
    when the fold is wrong. A foldable target must land near its experimental
    structure. Fold with production settings (--sampling_steps ~200,
    --diffusion_samples >= 5) before gating; too few steps/samples fails a good model.

    Returns (best_conf_rmsd, best_conf_tm). Raises AssertionError if thresholds set
    and the confidence-selected model misses them.
    """
    n = _num_models(protein_name, results_dir)
    rows = []
    for m in range(n):
        rmsd, tm = compute_rmsd(protein_name, m, results_dir)
        rows.append((m, rmsd, tm))
    conf_m, conf_rmsd, conf_tm = rows[0]  # model 0 == best confidence
    oracle = min(rows, key=lambda r: r[1])

    print(f"{'#'*60}")
    print(f"GROUND-TRUTH ACCURACY — {protein_name} ({n} samples)")
    for m, rmsd, tm in rows:
        tag = " <- best confidence (gated)" if m == 0 else ""
        print(f"  model {m}: RMSD {rmsd:6.3f} Å   TM {tm:.3f}{tag}")
    print(f"  oracle-best sample: model {oracle[0]} RMSD {oracle[1]:.3f} Å TM {oracle[2]:.3f}")
    ok = True
    if max_rmsd is not None:
        good = conf_rmsd <= max_rmsd
        ok &= good
        print(f"  gate RMSD <= {max_rmsd} Å: {'PASS' if good else 'FAIL'} ({conf_rmsd:.3f})")
    if min_tm is not None:
        good = conf_tm >= min_tm
        ok &= good
        print(f"  gate TM >= {min_tm}: {'PASS' if good else 'FAIL'} ({conf_tm:.3f})")
    print(f"{'#'*60}\n")
    if (max_rmsd is not None or min_tm is not None) and not ok:
        raise AssertionError(
            f"{protein_name}: confidence-selected fold missed the ground-truth gate "
            f"(RMSD {conf_rmsd:.3f} Å, TM {conf_tm:.3f})")
    return conf_rmsd, conf_tm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground-truth RMSD/TM between predicted and experimental structures")
    parser.add_argument("protein", help="Protein name (e.g., hemoglobin, prot)")
    parser.add_argument("--model", type=int, default=None, help="Score a single model index (default: full gate over all samples)")
    parser.add_argument("--max-rmsd", type=float, default=None, help="Gate: fail (exit 1) if the confidence-selected fold exceeds this CA-RMSD (Å)")
    parser.add_argument("--min-tm", type=float, default=None, help="Gate: fail (exit 1) if the confidence-selected fold is below this TM-score")
    args = parser.parse_args()
    if args.model is not None:
        compute_rmsd(args.protein, args.model)
    else:
        try:
            evaluate(args.protein, max_rmsd=args.max_rmsd, min_tm=args.min_tm)
        except AssertionError as e:
            print(f"GATE FAIL: {e}", file=sys.stderr)
            sys.exit(1)
