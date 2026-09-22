#!/usr/bin/env python3
"""D117: does `safe_multi_chain_permutation_alignment` complete, or does it fall back?

The campaign has carried "multi-chain permutation alignment is NOT COVERED" since
`of3t-reference`, on the grounds that upstream raises `KeyError 'ref_space_uid_to_perm'` and
takes its documented fallback to naive alignment. Two rows have now hit that key by two
different routes, both in HARNESS collation and copying rather than in the model, which is
evidence for a harness gap and not a capability one.

This probe calls upstream's own `safe_multi_chain_permutation_alignment` on a real batch and
reports whether the real alignment completed. Upstream's `permutation_alignment.py` is not
modified: the probe builds the batch, synthesises a prediction, and asks the guard
(`permalign_guard.py`) which path ran.

WHAT IS REAL AND WHAT IS NOT, before the numbers.
  REAL: the batch. Either `of3t-reference`'s frozen `batch_step003.pt`, which upstream's own
        collator produced, or a datapoint built by upstream's `WeightedPDBDataset` from
        upstream's own preprocessed corpus.
  REAL: the function. `openfold3.core.utils.permutation_alignment.
        safe_multi_chain_permutation_alignment`, unmodified, called the way `model.py:694`
        calls it including the sample-dimension expansion `model.py:669-672` does first.
  NOT REAL: the prediction. There is no rollout here, so `atom_positions_predicted` is the
        ground truth plus seeded Gaussian noise. That is enough for the question asked --
        whether the algorithm RUNS on these features -- and is not enough to say anything
        about which permutation a trained model would select. Stated, not implied.

The controls are the point. `--collate recurse` reproduces `of3t-bondcov`'s defect, and
`--drop-key` reproduces `of3t-reference`'s: both must make this probe REPORT a fallback, or
the probe cannot tell the two outcomes apart and has measured nothing.
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

from permalign_guard import AlignmentGuard  # noqa: E402


def collate1(x, ref=None, mode: str = "fixed"):
    """`of3t-auxheads`' batch-of-one collator, with the `ref_space_uid_to_perm` branch switchable.

    `mode="fixed"` is `perf/of3t_auxheads/bond_coverage.py:63` verbatim: the mapping gains a
    plain python list as its batch axis, which is what upstream's collator produces and what
    `permutation_alignment.py:1693-1702` and `multi_chain_permutation_alignment` index.
    `mode="recurse"` is the defect that row found -- recursing into the mapping unsqueezes every
    permutation tensor and hands `single_batch` the entry for uid 0 instead of the mapping.
    """
    if torch.is_tensor(x):
        if ref is None or not torch.is_tensor(ref):
            return x.unsqueeze(0)
        return x.unsqueeze(0) if x.dim() + 1 == ref.dim() else x
    if isinstance(x, dict):
        r = ref if isinstance(ref, dict) else {}
        return {k: ([v] if (k == "ref_space_uid_to_perm" and mode == "fixed")
                    else collate1(v, r.get(k), mode))
                for k, v in x.items()}
    if isinstance(ref, list) and isinstance(x, list):
        return x
    return [x]


def count_ref_spaces(batch) -> dict:
    r = batch.get("ref_space_uid_to_perm")
    if r is None:
        return {"present": False}
    out = {"present": True, "type": type(r).__name__}
    if isinstance(r, list):
        out["batch_axis_len"] = len(r)
        if r and isinstance(r[0], dict):
            out["n_ref_spaces_sample0"] = len(r[0])
            k = sorted(r[0])[0]
            out["example_uid"] = int(k)
            out["example_perm_shape"] = list(r[0][k].shape)
            out["max_uid"] = int(max(r[0]))
            # A ref space with one permutation is skipped outright
            # (`permutation_alignment.py:1177-1191`), so the symmetry the algorithm
            # actually has to resolve is the count of ref spaces above one.
            nper = [int(v.shape[0]) for v in r[0].values()]
            out["n_ref_spaces_with_alternatives"] = sum(1 for n in nper if n > 1)
            out["max_permutations_in_one_ref_space"] = max(nper)
    elif isinstance(r, dict):
        out["n_entries"] = len(r)
        ks = sorted(r)
        out["key_kind"] = type(ks[0]).__name__
        out["example_perm_shape"] = list(r[ks[0]].shape) if torch.is_tensor(r[ks[0]]) else None
    return out


def add_sample_dim(batch: dict, n_samples: int) -> dict:
    """Exactly what `model.py:669-672` does before it calls the alignment.

    The key is popped, every tensor gains a sampling dimension, and the key is put back onto
    the NEW dict. The pop is why a second forward over the same caller-held dict finds the key
    gone; here the batch is built fresh per call, so it is reproduced rather than inherited.
    """
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    perm = batch.pop("ref_space_uid_to_perm", None)
    out = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
    if perm is not None:
        out["ref_space_uid_to_perm"] = perm
    if n_samples > 1:
        # the rollout emits n_samples structures; the features stay at size 1 and are
        # expanded inside `reshape_per_sample_inputs`.
        pass
    return out


def flip_symmetric(batch: dict, pred: torch.Tensor, n_samples: int) -> dict:
    """Reorder the PREDICTION inside every ref space that has an alternative permutation.

    `moved 0 atoms` is only a result if the probe could have read a non-zero. Upstream picks,
    per ref space, the permutation of the ground truth closest to the prediction
    (`permutation_alignment.py:1197-1246`); with a prediction that is the ground truth plus
    isotropic noise, the identity IS the right answer everywhere and zero is the correct
    reading. So this arm hands it a prediction that is the ground truth with permutation 1
    applied inside each symmetric ref space. The algorithm now has to select something other
    than the identity, and the ground truth it returns has to differ from the naive one.
    """
    uid = batch["ref_space_uid"]
    uid = uid[:, 0] if (n_samples > 0 and uid.dim() == 3) else uid
    perms = batch["ref_space_uid_to_perm"]
    flipped = pred.clone()
    n_spaces = n_atoms = 0
    for b in range(uid.shape[0]):
        table = perms[b] if isinstance(perms, list) else perms
        for u, pm in table.items():
            if pm.shape[0] < 2:
                continue
            idx = torch.nonzero(uid[b] == int(u), as_tuple=True)[0]
            if idx.numel() != pm.shape[1]:
                continue
            src = idx[pm[1].long()]
            if torch.equal(src, idx):
                continue
            flipped[b, ..., idx, :] = pred[b, ..., src, :]
            n_spaces += 1
            n_atoms += int(idx.numel())
    return flipped, {"ref_spaces_flipped": n_spaces, "atoms_reordered": n_atoms}


def build_prediction(batch: dict, n_samples: int, noise: float, seed: int):
    gt = batch["ground_truth"]["atom_positions"]
    g = torch.Generator().manual_seed(seed)
    if n_samples == 0:
        base = gt if gt.dim() == 3 else gt.squeeze(1)
        return base + torch.randn(base.shape, generator=g, dtype=base.dtype) * noise
    base = gt if gt.dim() == 4 else gt.unsqueeze(1)
    base = base.expand(-1, n_samples, -1, -1)
    return base + torch.randn(base.shape, generator=g, dtype=base.dtype) * noise


def run_via_forward(a, batch, refspaces_raw, provenance, PA, t0):
    """The alignment as `model.py:694` reaches it, on a real rollout's own predicted positions."""
    import copy
    import json as _json
    sys.path.insert(0, str(HERE.parents[1] / "perf" / "of3t_reference"))
    import bundle_min as BM

    if not a.checkpoint:
        raise SystemExit("--via-forward needs --checkpoint")
    BM.pin_deterministic_kernels(True)
    cfg, model, loss_fn, dropout = BM.build(torch.float32, a.seed, "cpu", 0)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    if inc.unexpected_keys:
        raise SystemExit(f"KEY GATE FAILED: {len(inc.unexpected_keys)} unexpected tensors")
    del ck, sd

    batch = BM.move(batch, "cpu", torch.float32)
    pinned = BM.rng_state(model)
    # `--forward-twice` withholds the deep copy on purpose. `model.py:670` pops
    # `ref_space_uid_to_perm` off the dict it is handed and puts it back on a NEW dict, so the
    # caller's copy loses the key for good. A second forward over the same dict is therefore
    # the exact configuration `of3t-reference` was in before it added `copy.deepcopy`, and the
    # second pass here is the reproduction of its `KeyError`, not a simulation of it.
    n_passes = 2 if a.forward_twice else 1
    verdicts = []
    for i in range(n_passes):
        BM.set_rng_state(pinned, model)
        private = batch if a.forward_twice else copy.deepcopy(batch)
        if i == 0:
            gt_before = private["ground_truth"]["atom_positions"].clone()
        with torch.no_grad(), AlignmentGuard(PA.__name__) as guard:
            b, out = model(private)
        verdicts.append(guard.verdict())
    v = verdicts[-1]
    gt_after = b["ground_truth"]["atom_positions"]
    # The forward adds the sampling axis itself, so the returned ground truth is one rank
    # higher than the one that went in. Drop a size-1 sample axis before comparing.
    cmp_after = gt_after.squeeze(1) if (gt_after.dim() == gt_before.dim() + 1
                                        and gt_after.shape[1] == 1) else gt_after
    moved = (int((cmp_after != gt_before).any(dim=-1).sum())
             if cmp_after.shape == gt_before.shape else None)

    report = {
        "instrument": "D117: the alignment as the real forward reaches it",
        "tag": a.tag, "provenance": provenance, "ref_spaces": refspaces_raw,
        "call": {"via_forward": True, "checkpoint": str(a.checkpoint),
                 "pred_shape": list(out["atom_positions_predicted"].shape),
                 "seed": a.seed, "collate": a.collate, "drop_key": a.drop_key,
                 "forward_twice": a.forward_twice},
        "verdict": v,
        "verdict_per_pass": verdicts,
        "effect": {"gt_atoms_moved": moved,
                   "gt_shape_before": list(gt_before.shape),
                   "gt_shape_after": list(gt_after.shape),
                   "all_losses_zeroed": all(float(w.abs().sum()) == 0.0
                                            for w in b["loss_weights"].values())},
        "dropout": dropout,
        "openfold3": getattr(__import__("openfold3"), "__file__", None),
        "torch": torch.__version__, "total_s": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(_json.dumps(report, indent=1, default=str) + "\n")
    if len(verdicts) > 1:
        for i, vv in enumerate(verdicts):
            print(f"  pass {i + 1}: "
                  f"{'COMPLETED' if vv['completed'] else 'FELL BACK'}"
                  + (f"  {vv['errors'][0]['error']}" if vv["errors"] else ""))
    state = "COMPLETED" if v["completed"] else "FELL BACK"
    print(f"\n{a.tag}: alignment {state} inside the real forward  "
          f"(calls {v['n_alignment_calls']}, completed {v['n_completed']}, "
          f"naive invoked {v['naive_fallback_invoked']}, detectors agree {v['detectors_agree']})")
    print(f"  ref spaces: {refspaces_raw}")
    print(f"  prediction shape from the rollout: {list(out['atom_positions_predicted'].shape)}")
    print(f"  gt atoms moved by the alignment: {moved}")
    for e in v["errors"]:
        print(f"  ERROR {e['error']}")
    for r in v["log_records"]:
        print(f"  LOG   [{r['level']}] {r['message'].splitlines()[0]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_argument_group("batch source")
    src.add_argument("--batch", type=Path, help="a frozen batch upstream's collator produced")
    src.add_argument("--data-dir", type=Path)
    src.add_argument("--cache-file", type=Path)
    src.add_argument("--package", default="openfold3")
    src.add_argument("--stage", default="finetune_1")
    src.add_argument("--crop", type=int, default=256)
    src.add_argument("--index", type=int, default=12)
    src.add_argument("--rank-template", type=Path)
    ap.add_argument("--collate", default="fixed", choices=["fixed", "recurse"],
                    help="recurse reproduces of3t-bondcov's defect; a negative control")
    ap.add_argument("--drop-key", action="store_true",
                    help="delete ref_space_uid_to_perm before the call, which is what a second "
                         "forward over one dict leaves behind; of3t-reference's mechanism")
    ap.add_argument("--samples", type=int, default=1,
                    help="sampling dimension size; 0 calls the alignment with no sample axis")
    ap.add_argument("--noise", type=float, default=1.0, help="Angstrom sigma on the prediction")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--via-forward", action="store_true",
                    help="run the real checkpointed forward and let it call the alignment, "
                         "instead of calling the alignment on a synthetic prediction")
    ap.add_argument("--checkpoint", type=Path, help="required by --via-forward")
    ap.add_argument("--forward-twice", action="store_true",
                    help="two forwards over ONE dict, with no deep copy: of3t-reference's "
                         "configuration before its fix")
    ap.add_argument("--flip-symmetric", action="store_true",
                    help="apply permutation 1 inside each symmetric ref space of the "
                         "PREDICTION, so a completing alignment has to select a non-identity")
    ap.add_argument("--also-naive", action="store_true",
                    help="run naive_alignment on a copy and report how far the two differ")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()

    import batch_digest as BD  # noqa: E402
    BD.seed_everything(a.seed)

    provenance = {}
    if a.batch:
        batch = torch.load(a.batch, weights_only=False)
        provenance = {"source": "frozen batch", "path": str(a.batch),
                      "collated_by": "upstream's own collator"}
    else:
        if not a.data_dir:
            raise SystemExit("need --batch or --data-dir")
        ds = BD.build_dataset(a.package, a.data_dir, 4, token_budget=a.crop,
                              split="train", stage=a.stage, cache_file=a.cache_file)
        guard = BD.install_retry_guard(ds)
        dp = ds.datapoint_cache.iloc[a.index]
        sample = ds[a.index]
        if guard["retries"]:
            raise SystemExit(f"{guard['retries']} silent sample substitutions")
        tmpl = torch.load(a.rank_template, weights_only=False) if a.rank_template else None
        batch = collate1(sample, tmpl, a.collate)
        provenance = {"source": "WeightedPDBDataset", "data_dir": str(a.data_dir),
                      "cache_file": str(a.cache_file) if a.cache_file else None,
                      "stage": a.stage, "crop": a.crop, "index": a.index,
                      "pdb_id": str(dp["pdb_id"]),
                      "datapoint": str(dp["preferred_chain_or_interface"]),
                      "n_tokens_real": int(sample["token_mask"].sum()),
                      "collate": a.collate}

    refspaces_raw = count_ref_spaces(batch)
    if a.drop_key:
        batch.pop("ref_space_uid_to_perm", None)

    from openfold3.core.utils import permutation_alignment as PA

    if a.via_forward:
        # The faithful arm: no synthetic prediction at all. The checkpointed model runs its own
        # rollout and calls the alignment from `model.py:694` with the positions it predicted,
        # having done its own `model.py:669-672` sample-dimension expansion. Everything below
        # that reads `pred` is skipped, so `atom_positions_predicted` is never ours here.
        return run_via_forward(a, batch, refspaces_raw, provenance, PA, t0)

    if a.samples > 0:
        batch = add_sample_dim(batch, a.samples)
    pred = build_prediction(batch, a.samples, a.noise, a.seed)
    flip_info = None
    if a.flip_symmetric:
        pred, flip_info = flip_symmetric(batch, pred, a.samples)

    gt_before = batch["ground_truth"]["atom_positions"].clone()

    # The same batch through the fallback alone, so "completed" can be reported beside what
    # completing was worth: if the full path and the naive path leave identical ground truth,
    # taking the fallback silently cost nothing on this batch, and that is worth knowing too.
    naive_gt = None
    naive_err = None
    if a.also_naive:
        import copy as _copy
        nb = _copy.deepcopy(batch)
        try:
            nper, npred = PA.reshape_per_sample_inputs(
                batch=nb, atom_positions_predicted=pred)
            nf = PA.naive_alignment(batch=nper, atom_positions_predicted=npred)
            nf = PA.format_output_gt_features(ground_truth_features=nf,
                                              batch_dims=pred.shape[:-2])
            naive_gt = nf["atom_positions"].clone()
        except Exception as e:  # noqa: BLE001
            naive_err = f"{type(e).__name__}: {e}"

    with AlignmentGuard(PA.__name__) as guard:
        PA.safe_multi_chain_permutation_alignment(batch=batch, atom_positions_predicted=pred)
    v = guard.verdict()

    gt_after = batch["ground_truth"]["atom_positions"]
    moved = None
    if gt_after.shape == gt_before.shape:
        moved = int((gt_after != gt_before).any(dim=-1).sum())
    vs_naive = None
    if a.also_naive:
        if naive_gt is None:
            vs_naive = {"error": naive_err}
        elif naive_gt.shape != gt_after.shape:
            vs_naive = {"shape_mismatch": [list(naive_gt.shape), list(gt_after.shape)]}
        else:
            d = (naive_gt != gt_after).any(dim=-1)
            vs_naive = {"atoms_differing_from_naive": int(d.sum()),
                        "max_abs_coord_diff_A": float((naive_gt - gt_after).abs().max())}

    losses_zeroed = all(float(w.abs().sum()) == 0.0
                        for w in batch.get("loss_weights", {}).values()) \
        if batch.get("loss_weights") else None

    report = {
        "instrument": "D117: did upstream's multi-chain permutation alignment complete?",
        "tag": a.tag,
        "provenance": provenance,
        "ref_spaces": refspaces_raw,
        "call": {"samples": a.samples, "pred_shape": list(pred.shape),
                 "noise_sigma_A": a.noise, "seed": a.seed,
                 "drop_key": a.drop_key, "collate": a.collate,
                 "flip_symmetric": a.flip_symmetric},
        "verdict": v,
        "flip_symmetric": flip_info,
        "vs_naive": vs_naive,
        "effect": {"gt_atoms_moved": moved,
                   "gt_shape_before": list(gt_before.shape),
                   "gt_shape_after": list(gt_after.shape),
                   "all_losses_zeroed": losses_zeroed},
        "openfold3": getattr(__import__("openfold3"), "__file__", None),
        "torch": torch.__version__,
        "total_s": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")

    state = "COMPLETED" if v["completed"] else "FELL BACK"
    print(f"\n{a.tag}: alignment {state}  "
          f"(calls {v['n_alignment_calls']}, completed {v['n_completed']}, "
          f"naive invoked {v['naive_fallback_invoked']}, detectors agree {v['detectors_agree']})")
    print(f"  ref spaces: {refspaces_raw}")
    print(f"  gt atoms moved by the alignment: {moved}")
    if flip_info is not None:
        print(f"  prediction flipped in {flip_info['ref_spaces_flipped']} ref spaces "
              f"({flip_info['atoms_reordered']} atoms)")
    if vs_naive is not None:
        print(f"  vs naive alignment: {vs_naive}")
    for e in v["errors"]:
        print(f"  ERROR {e['error']}")
    for r in v["log_records"]:
        print(f"  LOG   [{r['level']}] {r['message'].splitlines()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
