"""Why a conv layer's tensor-product output disagrees while the network's answer does not.

    scripts/diffdock_port/confidence_tp_layout.sh

`confidence_parity.py` reports that from conv layer 2 onward the `.tp` outputs disagree
with upstream's even as a multiset of rows, while the final `[N, 2]` matches to ~1e-5 and
the pose ranking is identical on every complex. Both cannot be describing the same kind of
difference, and an intermediate activation that is genuinely wrong cannot leave the output
right across 30 poses.

The hypothesis this script tests is **layout**: the port's `FullyConnectedTensorProduct`
emits the same numbers in a different channel order than e3nn's. Downstream code reads them
consistently, so nothing is wrong -- but a row-wise or row-sorted comparison of the raw
tensor sees total disagreement, because it compares column j against a different channel.

Three comparisons, cheapest first, on one module:

* rows sorted  -- same rows, any order?
* columns matched by their own statistics -- is there a permutation of channels that lines
  the two up?
* every scalar sorted -- same multiset of numbers at all?

If the last passes and the first fails, the difference is layout and nothing else. If the
last fails too, the values really do differ and the per-module grade has found something.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from tt_bio.diffdock import config as dfd_config
from tt_bio.diffdock.confidence_model import load_released

GOLD = Path.home() / ".coworker/state/dfd/goldens"
CKPT_DIR = Path.home() / "scratch/dfd/ckpt/extracted/confidence_model"


def pcc(a, b) -> float:
    x, y = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def match_columns(mine: np.ndarray, ref: np.ndarray):
    """Greedily pair each reference channel with the port channel it correlates best with.

    Returns `(permutation, worst paired correlation)`. A clean layout difference gives a
    permutation that is one-to-one with every pair near 1.0; genuinely different values
    give a best pair that is still poor.
    """
    a = np.asarray(mine, np.float64)
    b = np.asarray(ref, np.float64)
    a = (a - a.mean(0)) / (a.std(0) + 1e-12)
    b = (b - b.mean(0)) / (b.std(0) + 1e-12)
    corr = (b.T @ a) / len(a)                      # [ref channel, port channel]
    perm, worst, taken = [], 1.0, set()
    for j in range(corr.shape[0]):
        order = np.argsort(-np.abs(corr[j]))
        pick = next((k for k in order if k not in taken), int(order[0]))
        taken.add(pick)
        perm.append(int(pick))
        worst = min(worst, float(abs(corr[j, pick])))
    return perm, worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="8G43_ZU6")
    ap.add_argument("--module", action="append", dest="modules",
                    default=None, help="defaults to the three worst from the parity run")
    args = ap.parse_args()
    cid = args.id
    modules = args.modules or ["conv_layers.18.tp", "conv_layers.27.tp", "conv_layers.20.tp"]

    cfg = dfd_config.load(CKPT_DIR, old_model=True, name="confidence")
    state = torch.load(CKPT_DIR / "best_model_epoch75.pt", map_location="cpu",
                       weights_only=False)
    model = load_released(state, cfg)

    inp = {k: torch.from_numpy(v)
           for k, v in np.load(GOLD / f"{cid}__s0/confidence_inputs.npz").items()}
    poses = np.load(GOLD / f"{cid}__s0/poses.npz")
    lig_pos = torch.from_numpy(poses["pos"] - poses["center"]).float()

    # Hook only the modules under test: the full 295-module capture is minutes, this is not.
    acts: dict[str, list] = {m: [] for m in modules}
    by_name = dict(model.named_modules())
    for name in modules:
        by_name[name].register_forward_hook(
            lambda m, a, o, n=name: acts[n].append(o.detach().cpu().numpy()))

    with torch.no_grad():
        for pose in lig_pos:
            model(inp["lig_x"].float(), pose, inp["bond_index"].long(),
                  inp["bond_attr"].float(), inp["rec_x"].float(), inp["rec_pos"].float(),
                  inp["rec_edge_index"].long(), inp["atom_x"].float(),
                  inp["atom_pos"].float(), inp["atom_edge_index"].long(),
                  inp["atom_rec_edge_index"].long())

    for name in modules:
        mine = np.concatenate(acts[name], axis=0)
        gold = GOLD / f"{cid}__mod" / f"{cid}__conf__{name.replace('.', '__')}__c0.npz"
        if not gold.exists():
            print(f"{name}: no golden at {gold.name}")
            continue
        ref = np.load(gold)["out0"]
        rows = min(len(ref), len(mine))
        a, b = mine[:rows], ref[:rows]
        print(f"\n== {name}  port {mine.shape}  golden {ref.shape}")
        if a.shape[1:] != b.shape[1:]:
            print("   shape mismatch, nothing further to say")
            continue
        print(f"   row-wise PCC            {pcc(a, b):.6f}")
        sa = a.reshape(rows, -1)[np.lexsort(a.reshape(rows, -1).T[::-1])]
        sb = b.reshape(rows, -1)[np.lexsort(b.reshape(rows, -1).T[::-1])]
        print(f"   rows sorted PCC         {pcc(sa, sb):.6f}")
        print(f"   all scalars sorted PCC  {pcc(np.sort(a, axis=None), np.sort(b, axis=None)):.6f}")
        perm, worst = match_columns(a, b)
        identity = perm == list(range(len(perm)))
        print(f"   best channel pairing: worst |corr| {worst:.6f}  "
              f"permutation is identity: {identity}")
        if not identity:
            print(f"   permutation (ref channel -> port channel), first 24: {perm[:24]}")
            print(f"   PCC after applying it   {pcc(a[:, perm], b):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
