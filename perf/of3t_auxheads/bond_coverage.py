#!/usr/bin/env python3
"""PROTOCOL SS6: fire the `bond` loss term and measure its gradient contribution.

SS6 s coverage table stood at 7 of 8 terms with `bond` firing nowhere, and the campaign s
own record read that as a dataset gap. It is narrower than that. Two independent things
kept `bond` off, both visible in BUNDLE-MIN-043 s batch: `initial_training` sets
`loss_weights.bond = 0.0`, and 5nw3 s crop carries `token_bonds` all-zero, so even at
weight 4.0 there would be nothing to sum over. `bond_scan.py` shows the second is a
property of that batch and not of the corpus -- all eight targets of the training subset
carry inter-token bonds under `finetune_1`.

So this fires it: one (stage, dataset) pair, `finetune_1` / `weighted-pdb`, on a target
whose crop carries bonds, with `loss_weights.bond = 4.0`.

SS6 also says a term that fires with a zero gradient contribution has been skipped with
extra steps, so the reported quantity is the CONTRIBUTION, not the fact that it ran: one
forward, two losses differing only in the bond weight, two backwards, and the answer is
`||g(bond=4) - g(bond=0)||^2` as a share of `||g(bond=4)||^2`. A term that changes no
gradient reads exactly zero there, and that is the shape the check has to be able to fail.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "of3_port"))
sys.path.insert(0, str(HERE.parent / "of3t_reference"))

import batch_digest as BD  # noqa: E402
import bundle_min as BM  # noqa: E402


def collate1(x, ref=None):
    """A batch of one, with a known-good batch as the rank template.

    Unsqueezing every tensor is wrong: the pipeline already emits some features with the
    axis the model expects, and a blind unsqueeze gives them one too many -- which surfaces
    far downstream as an expand() of a [1,1,1] onto size [1,1]. So each feature is matched
    against BUNDLE-MIN-043s own batch, which upstreams real collator produced, and gains
    an axis only where its rank is one short.
    """
    if torch.is_tensor(x):
        if ref is None or not torch.is_tensor(ref):
            return x.unsqueeze(0)
        return x.unsqueeze(0) if x.dim() + 1 == ref.dim() else x
    if isinstance(x, dict):
        r = ref if isinstance(ref, dict) else {}
        # `ref_space_uid_to_perm` is a per-sample MAPPING {ref-space uid -> [n_perm, n_atom]}
        # and its batch axis is a plain python list, not a tensor dimension: upstream's
        # `expand_batch_to_per_sample` does `ref_space_uid_to_perm[i]` over the batch
        # (permutation_alignment.py:1693-1702). Recursing into it instead unsqueezes every
        # permutation tensor and hands `single_batch["ref_space_uid_to_perm"]` the entry for
        # uid 0 rather than the mapping, so the first uid above 0 raises IndexError, upstream
        # catches it and silently falls back to NAIVE alignment. Found on 4G5J, whose crop has
        # 199 ref spaces; a crop with one would never have shown it.
        return {k: ([v] if k == "ref_space_uid_to_perm" else collate1(v, r.get(k)))
                for k, v in x.items()}
    # Same rule for the non-tensor features.
    if isinstance(ref, list) and isinstance(x, list):
        return x
    return [x]


def snapshot(model):
    """Clone the gradients at the parameter's own dtype, not at float64.

    Two float64 snapshots of a 570 M-parameter model are 9.2 GB that carry no information the
    float32 gradients did not already have; the comparison below upcasts per tensor, so the
    accumulated sums are float64 either way. Measured on pc: with this and the checkpoint freed,
    the crop-384 run peaks around 13 GB instead of being OOM-killed at 22 GB.
    """
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--package", default="tt_bio._vendor.openfold3",
                    help="package the dataset classes come from")
    ap.add_argument("--cache-file", type=Path,
                    help="subset cache to use instead of the seeded 8-structure sample; "
                         "needed to reach a corpus built with build_of3_subset.py --ids")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--stage", default="finetune_1")
    ap.add_argument("--crop", type=int, default=384,
                    help="token budget. finetune_1 s own crop is 640; 384 is the largest "
                         "this campaign has ever run a taped cycle at and is what keeps "
                         "the probe affordable. The bond term does not depend on the crop "
                         "beyond the crop carrying a bond, which is asserted below.")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--rank-template", type=Path,
                    help="a batch upstreams own collator produced, used only to decide "
                         "which features already carry the batch axis")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()

    BD.seed_everything(a.seed)
    ds = BD.build_dataset(a.package, a.data_dir, 4,
                          token_budget=a.crop, split="train", stage=a.stage,
                          cache_file=a.cache_file)
    guard = BD.install_retry_guard(ds)
    dp = ds.datapoint_cache.iloc[a.index]
    sample = ds[a.index]
    if guard["retries"]:
        raise SystemExit(f"{guard['retries']} silent sample substitutions; this is not the "
                         f"target it claims to be")
    nnz = int((sample["token_bonds"] != 0).sum())
    w_bond = float(sample["loss_weights"]["bond"])
    # The term is polymer-ligand, so `token_bonds` alone does not say it can fire: the 8
    # corpus targets all carry inter-token bonds and none carries a polymer-ligand one,
    # which is why `bond_loss` read 0.0 on every one of them. Report the mask the loss
    # actually sums over, computed with its own expression (diffusion.py:205-210).
    is_polymer = sample["is_protein"] + sample["is_dna"] + sample["is_rna"]
    mask_nnz = int((sample["token_bonds"]
                    * (is_polymer[..., None, :] * sample["is_ligand"][..., None]) != 0).sum())
    print(f"target {a.index} ({dp['pdb_id']} {dp['preferred_chain_or_interface']}): "
          f"token_bonds nnz {nnz} ({nnz // 2} pairs), bond_mask nnz {mask_nnz}, "
          f"loss_weights.bond {w_bond}, crop {a.crop}", flush=True)
    if nnz == 0:
        raise SystemExit("this crop carries no inter-token bond; pick another index")
    if w_bond == 0.0:
        raise SystemExit("this stage zeroes the bond weight; the term cannot fire")

    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    BM.pin_deterministic_kernels(True)
    cfg, model, loss_fn, dropout = BM.build(dtype, a.seed, "cpu", 0)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    if inc.unexpected_keys:
        raise SystemExit(f"KEY GATE FAILED: {len(inc.unexpected_keys)} unexpected tensors")
    # The checkpoint and its recast copy are another 4.6 GB held for the whole run.
    del ck, sd

    tmpl = torch.load(a.rank_template, weights_only=False) if a.rank_template else None
    batch = BM.move(collate1(sample, tmpl), "cpu", dtype)
    pinned = BM.rng_state(model)
    BM.set_rng_state(pinned, model)
    import copy
    private = copy.deepcopy(batch)
    ctx = BM.no_autocast() if dtype is torch.float64 else BM._null()
    with ctx:
        b, out = model(private)
    print(f"[{time.time()-t0:.0f}s] forward done", flush=True)

    arms = {}
    for tag, w in (("bond4", 4.0), ("bond0", 0.0)):
        b["loss_weights"]["bond"] = torch.full_like(b["loss_weights"]["bond"], w)
        BM.set_rng_state(pinned, model)
        with ctx:
            loss, bd = loss_fn(b, out, _return_breakdown=True)
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=(tag == "bond4"))
        g = snapshot(model)
        arms[tag] = {
            "weight": w, "loss": float(loss),
            "breakdown": {k: float(v) for k, v in bd.items()
                          if torch.is_tensor(v) and v.numel() == 1},
            "grads": g,
        }
        print(f"[{time.time()-t0:.0f}s] {tag}: loss {float(loss):.9f}  "
              f"bond_loss {arms[tag]['breakdown'].get('bond_loss')}", flush=True)

    g4, g0 = arms["bond4"]["grads"], arms["bond0"]["grads"]
    tot4 = tot_d = 0.0
    per_section = {}
    for n, v4 in g4.items():
        v0 = g0.get(n)
        if v4 is None:
            continue
        v4 = v4.double()
        v0 = None if v0 is None else v0.double()
        s4 = float((v4 ** 2).sum())
        d = v4 if v0 is None else v4 - v0
        sd_ = float((d ** 2).sum())
        tot4 += s4
        tot_d += sd_
        sec = n.split(".")[0]
        e = per_section.setdefault(sec, {"sq": 0.0, "delta_sq": 0.0, "n": 0})
        e["sq"] += s4
        e["delta_sq"] += sd_
        e["n"] += 1
    for e in per_section.values():
        e["share_of_own"] = e["delta_sq"] / e["sq"] if e["sq"] else None

    report = {
        "instrument": "PROTOCOL SS6: bond term fires, with its gradient contribution measured",
        "stage": a.stage, "dataset": "weighted-pdb", "crop": a.crop, "index": a.index,
        "dtype": a.dtype, "seed": a.seed,
        "package": a.package, "cache_file": str(a.cache_file) if a.cache_file else None,
        "target": {"pdb_id": str(dp["pdb_id"]),
                   "datapoint": str(dp["preferred_chain_or_interface"]),
                   "token_bonds_nnz": nnz, "token_bonds_pairs": nnz // 2,
                   "bond_mask_nnz": mask_nnz, "loss_weight_bond": w_bond,
                   "n_tokens_real": int(sample["token_mask"].sum())},
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "grads"} for k, v in arms.items()},
        "bond_gradient_contribution": {
            "squared_norm_with_bond": tot4,
            "squared_norm_of_difference": tot_d,
            "share_of_squared_gradient_norm": tot_d / tot4 if tot4 else None,
            "n_params_moved": sum(1 for n, v4 in g4.items()
                                  if v4 is not None and g0.get(n) is not None
                                  and not torch.equal(v4, g0[n])),
            "n_params": sum(1 for v in g4.values() if v is not None),
        },
        "by_section": per_section,
        "dropout": dropout,
        "total_s": time.time() - t0,
    }
    c = report["bond_gradient_contribution"]
    print(f"\nbond gradient contribution: {c['share_of_squared_gradient_norm']:.6e} of the "
          f"squared gradient norm, moving {c['n_params_moved']} of {c['n_params']} tensors")
    for sec, e in sorted(per_section.items(), key=lambda kv: -kv[1]["delta_sq"]):
        print(f"  {sec:26s} delta_sq {e['delta_sq']:.6e}  share of its own {e['share_of_own']:.6e}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
