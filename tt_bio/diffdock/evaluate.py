"""DiffDock's evaluation metrics, as a library.

Upstream puts them inline in ``evaluate.py`` around its own inference loop, so nothing there
is callable without a dataset and a model. This module is the same arithmetic over poses that
already exist on disk, which is what both the accuracy gate and a user grading their own run
need.

Three groups, all card-free:

* **RMSD** -- :func:`symmetry_rmsd` is upstream's ``get_symmetry_rmsd`` (spyrmsd, heavy atoms,
  no minimisation), falling back to the naive atom-order RMSD on failure exactly as upstream
  does (``evaluate.py:475-482``). The fallback is *counted*: a silent naive RMSD flatters a
  symmetric ligand by several Angstrom.
* **Steric** -- :func:`min_cross_distance` (closest ligand-receptor approach) and
  :func:`min_self_distance` / :func:`self_intersect_fraction`, upstream's
  ``evaluate.py:486-520,633-634``.
* **The metric block** -- :func:`success_rates` reproduces upstream's ``performance_metrics``
  dict key for key, including the top-5 and top-10 variants, so a number here can be compared
  with a published one without re-deriving what it means.

Poses are ranked by the confidence model, so ``rmsds[i]`` is complex ``i``'s samples **in
confidence order**: ``rmsds[:, 0]`` is rank 1 and ``min(rmsds[:, :5])`` is upstream's top-5.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

CLASH_CUTOFF_A = 0.4  # upstream's self-intersection bar (evaluate.py:633)


# ----------------------------------------------------------------------------- RMSD


def _heavy(mol):
    from rdkit import Chem

    return Chem.RemoveHs(mol)


def symmetry_rmsd(ref, pred) -> tuple[float, bool]:
    """``(rmsd, used_naive)`` between two RDKit molecules of the same ligand.

    Symmetry-corrected through spyrmsd, which is what upstream calls and what makes a
    two-fold symmetric ring score the pose rather than the atom numbering. On any spyrmsd
    failure the caller gets the naive atom-order RMSD and ``used_naive=True``; upstream
    prints "Using non corrected RMSD" and carries on, and the count is the honest version
    of that print.
    """
    try:
        from spyrmsd import rmsd as sprmsd
        from spyrmsd.molecule import Molecule
    except ImportError as e:
        raise ImportError("pose grading needs spyrmsd: pip install 'tt-bio[eval]'") from e

    a, b = _heavy(ref), _heavy(pred)
    try:
        ma, mb = Molecule.from_rdkit(a), Molecule.from_rdkit(b)
        ma.strip()
        mb.strip()
        return float(sprmsd.symmrmsd(ma.coordinates, mb.coordinates, ma.atomicnums,
                                     mb.atomicnums, ma.adjacency_matrix, mb.adjacency_matrix,
                                     minimize=False)), False
    except Exception:
        x, y = a.GetConformer().GetPositions(), b.GetConformer().GetPositions()
        if x.shape != y.shape:
            raise
        return float(np.sqrt(((x - y) ** 2).sum(axis=1).mean())), True


def best_symmetry_rmsd(refs, pred) -> tuple[float, bool]:
    """Minimum :func:`symmetry_rmsd` over several reference conformers.

    The PoseBusters set ships every crystal copy of the ligand in ``<id>_ligands.sdf`` because
    it contains crystal contacts, and upstream's evaluator takes the minimum over all of them
    (``datasets/pdbbind.py:392``). Grading against one copy costs about 5 points of top-1.
    """
    vals = [symmetry_rmsd(r, pred) for r in refs]
    best = min(vals, key=lambda v: v[0])
    return best[0], any(v[1] for v in vals)


def centroid_distance(ref, pred) -> float:
    """Distance between the heavy-atom centroids, upstream's ``centroid_distance``."""
    x = _heavy(ref).GetConformer().GetPositions().mean(axis=0)
    y = _heavy(pred).GetConformer().GetPositions().mean(axis=0)
    return float(np.linalg.norm(x - y))


# --------------------------------------------------------------------------- steric


def min_cross_distance(ligand_xyz, receptor_xyz) -> float:
    """Closest approach between any ligand atom and any receptor atom, in Angstrom."""
    d = np.linalg.norm(np.asarray(ligand_xyz)[:, None, :] - np.asarray(receptor_xyz)[None], axis=-1)
    return float(d.min()) if d.size else math.inf


def min_self_distance(ligand_xyz) -> float:
    """Closest approach between two distinct atoms of the same pose (upstream's self distance)."""
    x = np.asarray(ligand_xyz)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    d = np.where(np.eye(d.shape[0], dtype=bool), np.inf, d)
    return float(d.min()) if d.size else math.inf


def self_intersect_fraction(min_self_distances, cutoff=CLASH_CUTOFF_A) -> float:
    """Percentage of poses whose two closest atoms are under ``cutoff`` (upstream reports %)."""
    v = np.asarray(min_self_distances, dtype=float)
    return float(round(100 * (v < cutoff).sum() / len(v), 2)) if v.size else 0.0


# -------------------------------------------------------------------- the metric block


def _pct(mask) -> float:
    m = np.asarray(mask)
    return float(round(100 * m.sum() / m.size, 2)) if m.size else 0.0


def _percentiles(v, prefix, out):
    for p in (25, 50, 75):
        out[f"{prefix}_percentile_{p}"] = float(np.percentile(v, p).round(2))


def success_rates(rmsds, centroid_distances=None, min_self_distances=None, run_times=None,
                  prefix="") -> dict:
    """Upstream's ``performance_metrics`` dict for a whole set.

    ``rmsds`` is ``[n_complexes, n_samples]`` in confidence order. The ``rmsds_below_2`` family
    is upstream's pooled rate over every sample, ``min_rmsds_below_2`` the any-sample rate, and
    ``top5``/``top10`` the rank-truncated ones that the paper plots. Percentages, like upstream.
    """
    r = np.asarray(rmsds, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    n, N = r.shape
    out = {f"{prefix}mean_rmsd": float(r.mean().round(3)),
           f"{prefix}rmsds_below_2": _pct(r < 2),
           f"{prefix}rmsds_below_5": _pct(r < 5),
           f"{prefix}min_rmsds_below_2": _pct(r.min(axis=1) < 2),
           f"{prefix}min_rmsds_below_5": _pct(r.min(axis=1) < 5)}
    _percentiles(r, f"{prefix}rmsds", out)

    if centroid_distances is not None:
        c = np.asarray(centroid_distances, dtype=float)
        c = c[:, None] if c.ndim == 1 else c
        out[f"{prefix}mean_centroid"] = float(c.mean().round(2))
        out[f"{prefix}centroid_below_2"] = _pct(c < 2)
        out[f"{prefix}centroid_below_5"] = _pct(c < 5)
        _percentiles(c, f"{prefix}centroid", out)

    for k in (5, 10):
        if N < k:
            continue
        top = r[:, :k].min(axis=1)
        out[f"{prefix}top{k}_rmsds_below_2"] = _pct(top < 2)
        out[f"{prefix}top{k}_rmsds_below_5"] = _pct(top < 5)
        _percentiles(top, f"{prefix}top{k}_rmsds", out)
        # Upstream's top-k centroid and self distance are the ones belonging to the BEST-RMSD
        # pose among the first k, not the minimum of each column on its own.
        best = np.argsort(r[:, :k], axis=1)[:, 0]
        if centroid_distances is not None:
            tc = np.asarray(centroid_distances, dtype=float)
            if tc.ndim == 2:
                tc = tc[np.arange(n), best]
                out[f"{prefix}top{k}_centroid_below_2"] = _pct(tc < 2)
                out[f"{prefix}top{k}_centroid_below_5"] = _pct(tc < 5)
                _percentiles(tc, f"{prefix}top{k}_centroid", out)
        if min_self_distances is not None:
            ms = np.asarray(min_self_distances, dtype=float)
            if ms.ndim == 2:
                out[f"{prefix}top{k}_self_intersect_fraction"] = self_intersect_fraction(
                    ms[np.arange(n), best])

    if min_self_distances is not None:
        out[f"{prefix}self_intersect_fraction"] = self_intersect_fraction(
            np.asarray(min_self_distances, dtype=float).reshape(-1))
    if run_times is not None:
        t = np.asarray(run_times, dtype=float)
        out[f"{prefix}run_times_mean"] = float(t.mean().round(2))
        out[f"{prefix}run_times_std"] = float(t.std().round(2))
    out[f"{prefix}n_complexes"] = int(n)
    out[f"{prefix}n_samples"] = int(N)
    return out


# --------------------------------------------------------- PoseBusters physical validity


def posebusters_validity(rows, config="redock", table_path=None) -> dict:
    """PoseBusters validity over ``rows`` of ``{mol_pred, mol_true, mol_cond}`` paths.

    A tt-bio extension: upstream grades its own two steric numbers and not the posebusters
    suite. ``valid`` is every check including the suite's own RMSD check, ``physics_valid``
    drops it, which is the number to quote beside an RMSD success rate so the same pose is not
    counted twice.
    """
    import pandas as pd
    try:
        from posebusters import PoseBusters
    except ImportError as e:
        raise ImportError("pose grading needs posebusters: pip install 'tt-bio[eval]'") from e

    df = PoseBusters(config=config).bust_table(pd.DataFrame(list(rows)))
    cols = [c for c in df.columns if df[c].dtype == bool]
    phys = [c for c in cols if "rmsd" not in c.lower()]
    valid = df[cols].all(axis=1).tolist()
    physics = df[phys].all(axis=1).tolist()
    if table_path is not None:
        Path(table_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(table_path)
    return {"valid": valid, "physics_valid": physics,
            "validity_rate": round(sum(valid) / len(valid), 4) if valid else None,
            "physics_validity_rate": round(sum(physics) / len(physics), 4) if physics else None,
            "n": len(valid), "table": str(table_path) if table_path else None}


# ------------------------------------------------------------------ benchmark driver


def _poses(complex_dir: Path, max_rank: int):
    """``rank<k>.sdf`` (or upstream's ``rank<k>_confidence*.sdf``) in confidence order."""
    from rdkit import Chem

    out = []
    for k in range(1, max_rank + 1):
        hit = next((p for p in [complex_dir / f"rank{k}.sdf",
                                *sorted(complex_dir.glob(f"rank{k}_confidence*.sdf"))]
                    if p.exists() and p.stat().st_size > 0), None)
        if hit is None:
            break
        m = next(Chem.SDMolSupplier(str(hit), sanitize=True, removeHs=True), None)
        if m is None:
            break
        out.append(m)
    return out


def _references(truth_dir: Path, cid: str):
    """Every crystal copy in ``<id>_ligands.sdf``, falling back to the single ``_ligand.sdf``."""
    from rdkit import Chem

    multi = truth_dir / cid / f"{cid}_ligands.sdf"
    one = truth_dir / cid / f"{cid}_ligand.sdf"
    if multi.exists():
        refs = [m for m in Chem.SDMolSupplier(str(multi), sanitize=True, removeHs=True)
                if m is not None]
        if refs:
            return refs
    m = next(Chem.SDMolSupplier(str(one), sanitize=True, removeHs=True), None) if one.exists() else None
    return [m] if m is not None else []


def grade_set(ids, poses_dir, truth_dir, max_rank=10, prefix="") -> dict:
    """Grade a whole held-out set of poses on disk against its crystal ligands.

    ``poses_dir/<id>/rank<k>.sdf`` against ``truth_dir/<id>/<id>_ligand(s).sdf``, which is the
    layout both PoseBusters and upstream's ``--out_dir`` use. Returns ``per_complex`` records
    and the :func:`success_rates` block over the complexes that produced poses.

    ``n_attempted`` and ``n_scored`` are reported separately and a complex with no pose stays
    in the list with ``scored: False``. A per-complex failure that quietly leaves the
    denominator is how a broken run reads as a good one.
    """
    poses_dir, truth_dir = Path(poses_dir), Path(truth_dir)
    ids = list(ids)
    recs, mat, cents, selfd = [], [], [], []
    for cid in ids:
        preds, refs = _poses(poses_dir / cid, max_rank), _references(truth_dir, cid)
        if not preds or not refs:
            recs.append({"id": cid, "scored": False,
                         "why": "no pose" if not preds else "no reference"})
            continue
        rmsd, naive, cen, msd = [], False, [], []
        for p in preds:
            v, used_naive = best_symmetry_rmsd(refs, p)
            naive |= used_naive
            rmsd.append(v)
            cen.append(min(centroid_distance(r, p) for r in refs))
            msd.append(min_self_distance(_heavy(p).GetConformer().GetPositions()))
        xyz = _heavy(preds[0]).GetConformer().GetPositions()
        recs.append({"id": cid, "scored": True, "rmsd": rmsd, "centroid": cen,
                     "min_self_distance": msd, "n_refs": len(refs), "naive_rmsd_used": naive,
                     "nonfinite_coords": not bool(np.isfinite(xyz).all())})
        mat.append(rmsd)
        cents.append(cen)
        selfd.append(msd)
    # The aggregate is rectangular, so a ragged set is graded at the narrowest complex's depth
    # and `samples_graded` says what that was. Padding the short ones instead would either
    # invent poses or move the pooled rate, and both are worse than saying the depth out loud.
    width = min((len(r) for r in mat), default=0)
    agg = success_rates([r[:width] for r in mat], [c[:width] for c in cents],
                        [s[:width] for s in selfd], prefix=prefix) if width else {}
    return {"per_complex": recs, "n_attempted": len(ids),
            "n_scored": sum(1 for r in recs if r["scored"]),
            "n_naive_rmsd": sum(1 for r in recs if r.get("naive_rmsd_used")),
            "nan_or_collapse": sum(1 for r in recs if r.get("nonfinite_coords")),
            "samples_graded": width, "metrics": agg}


# ------------------------------------------------------------------------- gnina


def gnina_available(binary="gnina") -> bool:
    return shutil.which(binary) is not None


def gnina_minimize(pose_path, receptor_path, out_path, binary="gnina", timeout=600):
    """Upstream's optional post-minimisation, which shells out to an external gnina binary.

    Off by default upstream and not part of the graded recipe. Without the binary this raises
    ``FileNotFoundError`` rather than silently returning the unminimised pose: a minimisation
    that did not happen must not be reported as one.
    """
    if not gnina_available(binary):
        raise FileNotFoundError(f"{binary} is not on PATH; gnina minimisation is optional and "
                                f"not part of the default recipe")
    cmd = [binary, "--minimize", "-r", str(receptor_path), "-l", str(pose_path),
           "--autobox_ligand", str(pose_path), "-o", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"gnina failed ({r.returncode}): {r.stderr.strip()[:400]}")
    score = None
    for line in r.stdout.splitlines():
        if line.startswith("Affinity:"):
            score = float(line.split()[1])
    return {"out": str(out_path), "affinity": score}
