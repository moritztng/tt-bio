"""Has the pair representation stopped moving before the third recycle runs?

Boltz-2 folds at 3 recycles because upstream's CLI default is 3, which is AlphaFold2's number.
Nothing in this codebase has ever measured whether the trunk is still learning on the last pass.
This harness attaches `TrunkModule.recycle_probe` and records, after each of the 4 trunk
iterations:

  * `dz_rel`   -- mean |z_i - z_{i-1}| / mean |z_i|, the raw pair-representation movement;
  * `dD_mean_A`, `dD_max_A` -- the change in the EXPECTED pairwise distance the distogram head
    predicts, in Angstrom. This is the quantity AlphaFold2's `recycle_early_stop_tolerance`
    thresholds (0.5 A on the CA-CA matrix), so a criterion expressed in it is comparable to a
    number that already ships in another folder, not invented here.

The distogram head is the model's own read-out of `z`, one Linear over the pair tensor, so a
criterion built on it needs no new weights and no new tensor: it is already computed once per
fold, and an early exit would compute it once per recycle instead.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

import torch                                                          # noqa: E402
torch.set_grad_enabled(False)


def distogram_probs(distogram_module, z):
    """The trunk distogram head's per-pair distance distribution. [B, N, N, bins]."""
    return torch.softmax(distogram_module(z)[..., 0, :].float(), dim=-1)


def bin_centers(num_bins: int):
    """Boltz-2's trunk distogram spans 2-22 A in `num_bins` bins.

    Not a guess: `ConfidenceHeads.forward` calls the first 20 of 64 bins "contact"
    (boltz2.py, `contacts[:, :, :, :20] = 1.0`), and the contact definition in this model is
    8 A. (8 - 2) / (22 - 2) * 64 = 19.2 bins, which is that 20 and no other range.
    """
    edges = torch.linspace(2.0, 22.0, num_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=10)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from fold_ab_multi import patch_boltz2_cfg

    B.SAMPLING_STEPS = a.sampling_steps
    patch_boltz2_cfg()
    target = ROOT / "perf/size512/fixtures" / f"cdk2x2_{a.tokens}.yaml"
    one_fold, meta, state = B.build_fold(
        "boltz2", ROOT / f".msa_b2zwr_{a.tokens}", target, target.with_suffix(".a3m"),
        recycling_steps=a.recycles)
    model = state.model
    dg = model.distogram_module
    nb = dg.num_bins
    # Boltz-2's distogram bins: the model's own const, not a guess.
    centers = bin_centers(nb)

    trunk = model._tt_trunk_module()                                  # noqa: SLF001
    seen = []

    def probe(cyc, z):
        seen.append((cyc, z.clone()))

    trunk.recycle_probe = probe
    secs, m = one_fold()
    trunk.recycle_probe = None

    rows = []
    prev_z = prev_D = prev_P = None
    for cyc, z in seen:
        P = distogram_probs(dg, z.float())
        D = (P * centers).sum(-1)
        r = {"iteration": cyc, "n_tokens": int(z.shape[1])}
        if prev_z is not None:
            r["dz_rel"] = round(float((z - prev_z).abs().mean() / z.abs().mean()), 6)
            d = (D - prev_D).abs()
            r["dD_mean_A"] = round(float(d.mean()), 6)
            r["dD_max_A"] = round(float(d.max()), 6)
            # scale-free companion, so the criterion does not rest on the bin range
            r["tv_mean"] = round(float(0.5 * (P - prev_P).abs().sum(-1).mean()), 6)
        rows.append(r)
        prev_z, prev_D, prev_P = z, D, P

    import importlib.metadata as im
    import socket
    out = {"host": socket.gethostname(), "arch": T.arch_name(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"), "ttnn": im.version("ttnn"),
           "target": target.name, "tokens": a.tokens, "recycles": a.recycles,
           "sampling_steps": a.sampling_steps, "fold_s": round(secs, 3),
           "plddt": m.get("plddt"), "n_msa": m.get("n_msa"),
           "bin_centers_A": [2.0, 22.0, nb],
           "curve": rows}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["curve"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
