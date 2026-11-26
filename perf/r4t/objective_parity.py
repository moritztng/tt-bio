"""Loss parity: tt_bio.train.rfd4_objective against upstream RFD4-Proteina's own code.

Runs upstream's `ProductSpaceFlowMatcher.corrupt_batch` + `compute_loss`,
`minimum_loss_attempt_indices` and `optimization_terms` exactly as `Proteina.training_step`
chains them (exploration_k = 4, t shared across a example's candidates), in float64 on CPU,
then our reference on the same batch with upstream's random draws replayed. Compared: x_t,
every term per physical example, the selected winners, the total, and the gradient of the
total with respect to every network output (upstream's autograd against our seeds).

Needs upstream's environment (rfproteina, lightning, atomworks, omegaconf), e.g.:

    ~/rfd4-cpu-venv/bin/python perf/r4t/objective_parity.py --n 8 --tokens 256 --out parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.train import rfd4_objective as R  # noqa: E402

N_CLASSES = 147          # nn.sequence_readout.num_classes
LATENT = 32              # local latent width of the release autoencoder


def upstream_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "product_flowmatcher": {
            "bb_ca": {"zero_com_noise": True, "guidance_enabled": True, "dim": 3,
                      "stochastic_centering_mode": "x_0", "stochastic_centering_scale": 0.2,
                      "stochastic_centering_scope": "per_group",
                      "stochastic_centering_group_keys": ["molecule_iid"]},
            "local_latents": {"zero_com_noise": False, "guidance_enabled": True, "dim": LATENT,
                              "stochastic_centering_mode": "none",
                              "stochastic_centering_scale": None},
        },
        "loss": {"signed_volume_weight": 2.5,
                 "sequence_prediction": {"enabled": True, "t_min": 0.0, "t_max": 1.0},
                 "t_distribution": {
                     m: {"name": "mix_unif_beta", "p1": p1, "p2": p2, "p3": p3}
                     for m, (p1, p2, p3) in R.T_DISTRIBUTION.items()} | {"shared_groups": None}},
    })


def make_batch(rng, n_ex, k, n):
    """One collated batch shaped like a crop-256 training batch: protein + ligand-like tokens."""
    b = n_ex * k
    rep = lambda a: np.repeat(a[::k], k, 0)
    length = rng.integers(int(0.6 * n), n + 1, n_ex)
    pad = np.zeros((b, n), bool)
    for i, L in enumerate(length):
        pad[i * k:(i + 1) * k, :L] = True
    ctx = rep(rng.random((b, n)) < 0.05) & pad
    sll = rep(rng.random((b, n)) < 0.9) & pad
    res = rep(rng.random((b, n)) < 0.95) & pad
    ccm = rep(rng.random((b, n)) < 0.05) & pad
    mol = rep(np.minimum(np.arange(n)[None] // max(n // 3, 1), 2).repeat(b, 0))
    ins = -np.ones((b, n), np.int64)
    tet = -np.ones((b, n, 4), np.int64)
    for i in range(n_ex):
        L = length[i]
        for q in rng.choice(np.arange(L // 2, L), 4, replace=False):        # unindexed tokens
            ins[i * k:(i + 1) * k, q] = rng.integers(0, L // 2)
        for c in rng.choice(np.arange(4, L - 4), 6, replace=False):         # chiral centres
            tet[i * k:(i + 1) * k, c] = [c - 2, c - 1, c + 1, c]
    seq = rep(np.where(rng.random((b, n)) < 0.1, rng.integers(1, 30, (b, n)), 0))
    lw = rep(np.where(rng.random((b, n)) < 0.1, 2.5, 1.0))
    # a CA trace in nm with ~0.38 nm steps, so the chiral volumes are real geometry
    ca = rep(np.cumsum(rng.normal(scale=0.22, size=(b, n, 3)), 1))
    lat = rep(rng.normal(size=(b, n, LATENT)))
    return {"padding_mask": pad, "is_in_context_conditioning": ctx, "supports_local_latent": sll,
            "is_resolved_mask": res, "coord_cond_mask": ccm, "molecule_iid": mol,
            "insertion_source_idx": ins, "tetrahedral_vertices": tet,
            "condition_sequence_1_residue": seq, "loss_weight": lw,
            "residue_type": rep(rng.integers(0, N_CLASSES, (b, n)))}, {"bb_ca": ca, "local_latents": lat}


def nn_outputs(rng, b, n, x1, xt, t):
    """Network-shaped outputs: a velocity that partly points at x_1, logits of realistic scale."""
    out = {}
    for m in R.MODES:
        tt = t[m][:, None, None]
        v_true = (x1[m] - xt[m]) / (1.0 - tt + 1e-5)
        out[f"v_{m}"] = v_true + rng.normal(scale=0.5, size=v_true.shape)
    out["sequence_logits"] = rng.normal(scale=2.0, size=(b, n, N_CLASSES))
    out["insertion_logits"] = rng.normal(scale=3.0, size=(b, n, n))
    return out


def fixture_batch(rng, k):
    """Upstream's own crops, from the committed fixture (`tt_bio/train/rfd4_data.py`)."""
    from tt_bio.train import rfd4_data
    b = rfd4_data.load()
    if int(b["exploration_k"]) != k:
        raise SystemExit(f"fixture was built at exploration_k={int(b['exploration_k'])}")
    batch = {key: b[key] for key in rfd4_data.FIXTURE_KEYS}
    batch["molecule_iid"] = batch["molecule_iid"].astype(np.int64)
    batch["insertion_source_idx"] = batch["insertion_source_idx"].astype(np.int64)
    batch["residue_type"] = batch["residue_type"].astype(np.int64)
    batch["condition_sequence_1_residue"] = batch["condition_sequence_1_residue"].astype(np.int64)
    batch.pop("coords"), batch.pop("atom37_mask"), batch.pop("token_type"), batch.pop("chain_iid")
    ca = np.asarray(b["coords"], np.float64)[:, :, 1, :] / 10.0
    # The AE is frozen and its latents are a function of its checkpoint; parity of the LOSS
    # does not need upstream's own latents, only the same ones on both sides.
    lat = np.repeat(rng.normal(size=(ca.shape[0] // k, ca.shape[1], LATENT))[:, None], k, 1)
    return batch, {"bb_ca": ca, "local_latents": lat.reshape(ca.shape[0], ca.shape[1], LATENT)}


def run_one(fm, seed, n_ex, k, n, broken=False, fixture=False):
    from rfproteina.utils.torch import minimum_loss_attempt_indices
    rng = np.random.default_rng(seed)
    batch_np, x1 = fixture_batch(rng, k) if fixture else make_batch(rng, n_ex, k, n)
    n = batch_np["padding_mask"].shape[1]
    n_ex = batch_np["padding_mask"].shape[0] // k
    t_ex = {m: R.sample_t(rng, (n_ex,), m) for m in R.MODES}
    t = {m: np.repeat(v, k) for m, v in t_ex.items()}
    b = n_ex * k

    # --- upstream: corrupt + loss + select + total, with autograd on the outputs
    T = lambda a: torch.as_tensor(a)
    ub = {key: T(v) for key, v in batch_np.items()}
    ub["x_1"] = {m: T(v) for m, v in x1.items()}
    torch.manual_seed(seed)
    ub = fm.corrupt_batch(ub, t={m: T(v) for m, v in t.items()})
    xt_up = {m: ub["x_t"][m].numpy() for m in R.MODES}
    outs = nn_outputs(rng, b, n, x1, xt_up, t)
    leaves = {key: T(v).requires_grad_(True) for key, v in outs.items()}
    nn_out = {"bb_ca": {"v": leaves["v_bb_ca"]}, "local_latents": {"v": leaves["v_local_latents"]},
              "sequence": {"logits": leaves["sequence_logits"]},
              "insertion": {"logits": leaves["insertion_logits"]}}
    losses = fm.compute_loss(batch=ub, nn_out=nn_out)
    win_up = minimum_loss_attempt_indices(losses=losses, loss_names=list(R.EXPLORATION_LOSSES),
                                          attempts_per_example=k)
    per_up = {name: losses[name].detach().numpy() for name in R.TERMS}
    sel = {name: loss[win_up] for name, loss in losses.items()}
    total_up = sum(v.mean() for v in fm.optimization_terms(sel).values())
    total_up.backward()
    grad_up = {key: leaf.grad.numpy() for key, leaf in leaves.items()}

    # --- ours: replay upstream's draws in its order (bb_ca noise, latent noise, one offset
    # per molecule in ascending id order), then the reference
    torch.manual_seed(seed)
    eps = {m: torch.randn((b, n, x1[m].shape[-1])).numpy() for m in R.MODES}
    mk = R.masks(batch_np)["emit"]["bb_ca"] & ~batch_np["coord_cond_mask"]
    offsets = {int(g): torch.randn(b, 3).numpy() for g in np.unique(batch_np["molecule_iid"][mk])}
    ours = {**batch_np, "exploration_k": k}
    if broken:          # the control: a featuriser that forgot the per-token loss weight
        ours.pop("loss_weight")
    ours.update(R.corrupt(batch_np, x1, t, eps, offsets))
    ours["t"] = t
    per_ours = {name: v[0] for name, v in R.terms(ours, outs).items()}
    total, br, seeds = R.rfd4_loss(ours, outs, R.WEIGHTS)

    rel = lambda a, b_: float(np.max(np.abs(a - b_)) / max(np.max(np.abs(b_)), 1e-300))
    return {
        "seed": seed, "tokens": n, "examples": n_ex, "k": k,
        "x_t_rel": max(rel(ours["x_t"][m], xt_up[m]) for m in R.MODES),
        "winners_equal": br["winners"] == win_up.tolist(),
        "terms_rel": {name: rel(per_ours[name], per_up[name]) for name in R.TERMS},
        "term_values": {name: br[name]["value"] for name in R.TERMS},
        "total_rel": abs(total - total_up.item()) / abs(total_up.item()),
        "total": total, "total_upstream": total_up.item(),
        "grad_rel": {key: rel(seeds[key], grad_up[key]) for key in grad_up},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=8, help="batches")
    ap.add_argument("--examples", type=int, default=2, help="examples per batch")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--tol", type=float, default=1.0e-4, help="PLAN.md G3 fp32-arm bar")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fixture", action="store_true",
                    help="use the committed fixture's real crops instead of synthetic batches")
    ap.add_argument("--broken", action="store_true",
                    help="control arm: drop loss_weight on our side; must FAIL")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
    fm = ProductSpaceFlowMatcher(upstream_cfg())
    rows = [run_one(fm, s, a.examples, R.EXPLORATION_K, a.tokens, a.broken, a.fixture) for s in range(a.n)]
    worst = max(max(r["terms_rel"].values()) for r in rows)
    worst = max(worst, max(r["total_rel"] for r in rows), max(r["x_t_rel"] for r in rows))
    gworst = max(max(r["grad_rel"].values()) for r in rows)
    ok = worst <= a.tol and gworst <= a.tol and all(r["winners_equal"] for r in rows)
    summary = {"worst_rel": worst, "worst_grad_rel": gworst, "tol": a.tol, "n": len(rows),
               "terms": len(R.TERMS), "pass": ok, "torch": torch.__version__, "rows": rows}
    text = json.dumps(summary, indent=1)
    if a.out:
        a.out.write_text(text)
    print(f"LOSS-PARITY: rel={worst:.2e} grad_rel={gworst:.2e} tol={a.tol:.1e} n={len(rows)} "
          f"terms={len(R.TERMS)} winners_equal={all(r['winners_equal'] for r in rows)} "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
