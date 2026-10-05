"""Full-matrix confidence outputs, written the same way for every model.

Every fold that asks for ``--write_pae`` gets one ``{name}_pae.npz`` beside its structure and a
``{name}_pae.json`` sidecar naming each array in it. The arrays a model can hold:

``pae``            [N, N] float32, Å. Expected aligned error of token j when aligned on token i.
``pde``            [N, N] float32, Å. Expected distance error, where the head has one.
``contact_probs``  [N, N] float32, probability that the two tokens' representative atoms are
                   within ``contact_cutoff_A``. Symmetric, diagonal 1.
``contact_cutoff_A`` scalar float32, Å. The cutoff the probabilities are actually at.

N is the number of tokens in the written structure, in the structure's own order, padding
stripped. Every array is the best-ranked sample's, the one written as ``{name}.{fmt}``.

Contact probabilities come from a model's own distogram logits and nothing else: softmax over
the bins, summed over every bin whose upper edge is at most the requested cutoff. That is
upstream OpenDDE's ``compute_contact_prob`` rule. A distogram has a bin edge near 8 Å but rarely
on it, so the cutoff a user gets is the largest edge <= the one asked for, and that is the
number written beside the matrix. A model supplies its logits and its bin grid; the arithmetic
is here once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONTACT_CUTOFF_A = 8.0

#: Distogram bin grids, as (first break, last break, number of bins). Bin k covers
#: (break[k-1], break[k]]; the last bin is open above. Each grid is the model's training grid.
DISTOGRAM_GRID = {
    "boltz2": (2.0, 22.0, 64),          # boltz2 checkpoint min_dist/max_dist/num_bins
    "opendde": (2.25, 25.75, 96),       # opendde/config/model_base.py confidence.distogram
    "openfold3": (2.3125, 21.6875, 64),  # openfold3 train/losses.py
    "af2ig": (2.3125, 21.6875, 64),     # af2 distogram first_break/last_break/num_bins
    "esmfold2": (2.0, 22.0, 64),        # esmfold2 prepare_input distogram grid
    "rf3": (2.0, 22.0, 65),             # foundry rf3 af3_losses.py linspace(2, 22, 64) breaks
}
DISTOGRAM_GRID.update(openbind=DISTOGRAM_GRID["openfold3"], **{"opendde-abag": DISTOGRAM_GRID["opendde"]},
                      **{"esmfold2-fast": DISTOGRAM_GRID["esmfold2"]})

UNITS = {"pae": "angstrom", "pde": "angstrom", "contact_probs": "probability",
         "contact_cutoff_A": "angstrom"}


def bin_upper_edges(first: float, last: float, n_bins: int) -> np.ndarray:
    """Upper edge of each distogram bin, float64: ``n_bins - 1`` breaks, then +inf."""
    return np.append(np.linspace(first, last, n_bins - 1), np.inf)


def contact_probs(logits, upper_edges, cutoff: float = CONTACT_CUTOFF_A):
    """``(probs [N, N] float32, cutoff_A)`` from distogram logits ``[N, N, n_bins]``.

    Computed in float64. Symmetrised as the mean of P and its transpose, which changes nothing
    for a head whose logits are already symmetric; the diagonal is 1, a token's representative
    atom being 0 Å from itself, rather than whatever an untrained diagonal logit says.
    """
    x = _numpy(logits)
    edges = np.asarray(upper_edges, dtype=np.float64)
    if x.shape[-1] != edges.shape[0]:
        raise ValueError(f"distogram has {x.shape[-1]} bins but the grid has {edges.shape[0]}")
    keep = edges <= cutoff
    if not keep.any():
        raise ValueError(f"contact cutoff {cutoff} Å is below the first distogram bin edge "
                         f"{edges[0]:.3f} Å")
    p = np.empty(x.shape[:2])
    for i in range(0, x.shape[0], 128):      # float64 in row blocks: N*N*bins never at once
        e = np.asarray(x[i:i + 128], dtype=np.float64)
        e = np.exp(e - e.max(-1, keepdims=True))
        p[i:i + 128] = e[..., keep].sum(-1) / e.sum(-1)
    p = 0.5 * (p + p.T)
    np.fill_diagonal(p, 1.0)
    return p.astype(np.float32), float(edges[keep][-1])


def _numpy(t):
    if hasattr(t, "detach"):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def write(out_dir, name: str, model: str, *, real=None, pae=None, pde=None, distogram=None,
          cutoff: float = CONTACT_CUTOFF_A, absent: dict[str, str] | None = None) -> dict:
    """Write ``{name}_pae.npz`` and ``{name}_pae.json``; return the sidecar.

    ``pae``/``pde`` are [N, N] matrices in Å and ``distogram`` the [N, N, n_bins] logits, all
    for the best sample and all on the same token axis. ``real`` is a boolean [N] mask of the
    tokens that are in the written structure; every array is cropped to it. ``absent`` maps an
    array the model cannot produce to the reason, which goes into the sidecar instead of a
    zero-filled stand-in.
    """
    arrays: dict[str, np.ndarray] = {}
    for key, m in (("pae", pae), ("pde", pde)):
        if m is not None:
            arrays[key] = np.asarray(_numpy(m), dtype=np.float32)
    sidecar: dict = {"model": model}
    if distogram is not None:
        edges = bin_upper_edges(*DISTOGRAM_GRID[model])
        arrays["contact_probs"], edge = contact_probs(distogram, edges, cutoff)
        sidecar.update(contact_threshold_A=float(cutoff), contact_cutoff_A=round(edge, 4),
                       distogram_bin_upper_edges_A=[round(float(e), 4) for e in edges[:-1]])
    if real is not None:
        real = np.asarray(_numpy(real)).astype(bool)
        arrays = {k: v[np.ix_(real, real)] for k, v in arrays.items()}
    n = {v.shape[0] for v in arrays.values()}
    if len(n) > 1 or any(v.shape != (v.shape[0],) * 2 for v in arrays.values()):
        raise ValueError(f"{name}: confidence matrices disagree on the token axis: "
                         f"{ {k: v.shape for k, v in arrays.items()} }")
    if "contact_probs" in arrays:
        arrays["contact_cutoff_A"] = np.float32(sidecar["contact_cutoff_A"])
    sidecar["n_tokens"] = n.pop() if n else 0
    sidecar["token_order"] = "tokens of the written structure, in file order"
    sidecar["arrays"] = {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype),
                             "units": UNITS[k]} for k, v in arrays.items()}
    sidecar["absent"] = dict(absent or {})
    out_dir = Path(out_dir)
    np.savez_compressed(out_dir / f"{name}_pae.npz", **arrays)
    (out_dir / f"{name}_pae.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar
