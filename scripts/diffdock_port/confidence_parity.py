"""Grade `tt_bio.diffdock.confidence_model` against upstream's captured activations.

    scripts/diffdock_port/confidence_parity.sh [--id 8G43_ZU6] [--json OUT]

Needs `confidence_goldens.py` to have run first: it writes the per-module `conf__*` goldens,
the `confidence_inputs.npz` graph and the rebuilt `[N, 2]`.

**Both sides get the identical graph.** The port is fed upstream's own dataset tensors out of
`confidence_inputs.npz` rather than rebuilding the receptor k-NN and atom graphs itself, so a
disagreement here is the network's. Graph construction is `dfd-input`'s to grade and mixing
the two would hide either one: an edge-ordering difference would read as a conv-layer error,
and a conv-layer error would read as an edge-ordering difference.

**How a per-pose port is compared against a batched capture.** Upstream collates the 10 poses
into one disjoint graph, so every captured tensor is the 10 per-pose tensors concatenated in
pose order. The port runs one pose at a time, so its activations are concatenated the same way
before comparing. That is exact for node-indexed and edge-indexed tensors alike, because PyG
concatenates graph by graph. Where `GoldenStore` truncated a large tensor to its first rows,
only those rows are compared and the row count is reported -- 128 rows of a cross-edge tensor
still carries plenty of statistical power for a PCC, and nothing is quietly skipped.

The headline is **rank agreement**: how often the port's `argmax` of column 0 picks the pose
upstream picked. PCC and max relative error are the diagnostics that say *why* when it drops.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tt_bio.diffdock import config as dfd_config
from tt_bio.diffdock.confidence_model import load_released

GOLD = Path.home() / ".coworker/state/dfd/goldens"
CKPT_DIR = Path.home() / "scratch/dfd/ckpt/extracted/confidence_model"
DEFAULT_IDS = ("8G43_ZU6", "8AEU_M0L", "7A9E_R4W")


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the flattened tensors, in float64.

    A constant tensor has no correlation to report; both sides being identically constant is
    agreement, so that returns 1.0 rather than NaN.
    """
    x, y = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if x.std() == 0 and y.std() == 0:
        return 1.0 if np.allclose(x, y) else 0.0
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def max_rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    """Max |got - ref| / max(|ref|, scale), with *scale* the reference's own RMS.

    Dividing by |ref| alone makes the metric a report on the reference's smallest element:
    one near-zero activation in a tensor of thousands produces a huge ratio that says nothing
    about the error. The RMS floor keeps it a statement about the tensor.
    """
    g, r = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    scale = max(float(np.sqrt((r ** 2).mean())), 1e-12)
    return float(np.max(np.abs(g - r) / np.maximum(np.abs(r), scale)))


def load_model():
    cfg = dfd_config.load(CKPT_DIR, old_model=True, name="confidence")
    state = torch.load(CKPT_DIR / "best_model_epoch75.pt", map_location="cpu",
                       weights_only=False)
    return load_released(state, cfg), cfg


def run_port(cid: str, model, cfg, *, capture: bool = True):
    """Run the port on every captured pose.

    Returns ``([N, 2], {module: {call index: concatenated output}})``.

    **Which port call answers which golden.** Upstream runs one forward over the collated
    10-pose graph, so its `__c{j}` is that module's j-th call in a single pass. The port
    runs one forward per pose, so the same module's j-th call happens ten times. A module
    used once per pass (every conv layer) therefore has 10 port calls against 1 golden; one
    used twice (`lig_distance_expansion`, which upstream applies to ligand edges *and* to
    receptor-atom edges, and `cross_distance_expansion`) has 20 against 2. Flattening all
    port calls into one array and comparing it to golden call 0 interleaves two different
    edge sets and reports a correlation near zero on a port that is exactly right -- which
    is what the first run of this script did. Slicing `calls[j::per_pose]` keeps each
    golden against the calls that mean the same thing.
    """
    gold = GOLD / f"{cid}__s0"
    inp = {k: torch.from_numpy(v) for k, v in np.load(gold / "confidence_inputs.npz").items()}
    poses = np.load(gold / "poses.npz")
    lig_pos = torch.from_numpy(poses["pos"] - poses["center"]).float()

    acts: dict[str, list] = defaultdict(list)
    handles = []
    if capture:
        for name, mod in model.named_modules():
            if name and not list(mod.children()):
                handles.append(mod.register_forward_hook(
                    lambda m, a, o, n=name: acts[n].append(
                        o.detach().cpu().numpy() if torch.is_tensor(o) else None)))

    outs = []
    with torch.no_grad():
        for pose in lig_pos:
            outs.append(model(
                inp["lig_x"].float(), pose, inp["bond_index"].long(),
                inp["bond_attr"].float(), inp["rec_x"].float(), inp["rec_pos"].float(),
                inp["rec_edge_index"].long(), inp["atom_x"].float(),
                inp["atom_pos"].float(), inp["atom_edge_index"].long(),
                inp["atom_rec_edge_index"].long()))
    for h in handles:
        h.remove()

    n_poses = len(lig_pos)
    merged: dict[str, dict[int, np.ndarray]] = {}
    for name, calls in acts.items():
        if not calls or any(c is None for c in calls) or len(calls) % n_poses:
            continue
        per_pose = len(calls) // n_poses
        merged[name] = {j: np.concatenate(calls[j::per_pose], axis=0)
                        for j in range(per_pose)}
    return torch.cat(outs).numpy(), merged


def sorted_rows(a: np.ndarray) -> np.ndarray:
    """The tensor's rows in a canonical order, for comparing two orderings of one set.

    An edge-indexed tensor's row order is whatever order the graph builder emitted its
    edges in. Upstream builds the ligand graph once over the collated batch, so its rows
    run *all* poses' bond edges and then all poses' radius edges; the port builds one pose
    at a time. Same edges, same values, different row order -- and a row-wise correlation
    reads that as total disagreement. Sorting both sides turns the comparison into "is this
    the same multiset of rows", which is the question worth asking of an edge tensor. It is
    weaker than a row-wise match (it cannot tell which edge a row belongs to) and is
    reported beside it, never instead of it.
    """
    flat = np.asarray(a, np.float64).reshape(len(a), -1)
    return flat[np.lexsort(flat.T[::-1])]


def golden_modules(cid: str) -> dict[tuple[str, int], Path]:
    """``{(dotted module name, call index): npz}`` for the confidence goldens of *cid*.

    A missing directory is reported, not returned as "nothing to grade". `glob` on a path
    that does not exist yields nothing, which reads identically to a complex whose modules
    all agree -- and that is exactly how two complexes came back with a clean-looking
    `modules 0` after their `__mod` directories went away.
    """
    directory = GOLD / f"{cid}__mod"
    if not directory.is_dir():
        print(f"   {cid}: NO module goldens ({directory} does not exist); "
              f"per-module grading skipped, final-output numbers unaffected")
        return {}
    out = {}
    for p in sorted(directory.glob(f"{cid}__conf__*.npz")):
        m = re.fullmatch(rf"{re.escape(cid)}__conf__(.+)__c(\d+)", p.stem)
        if not m:
            continue
        out[(m.group(1).replace("__", "."), int(m.group(2)))] = p
    return out


def grade(cid: str, model, cfg) -> dict:
    got, acts = run_port(cid, model, cfg)
    ref_raw = np.load(GOLD / f"{cid}__s0" / "poses.npz")["confidence_raw"]
    rebuild_path = GOLD / f"{cid}__s0" / "confidence.npz"
    rebuild = None
    if rebuild_path.exists():
        z = np.load(rebuild_path)
        key = "confidence_raw" if "confidence_raw" in z else z.files[0]
        rebuild = z[key]

    modules = []
    for (name, call), path in golden_modules(cid).items():
        mine = acts.get(name, {}).get(call)
        if mine is None:
            continue
        z = np.load(path)
        if "out0" not in z:
            continue
        ref = z["out0"]
        rows = min(len(ref), len(mine))
        truncated = len(ref) < len(mine)
        a, b = mine[:rows], ref[:rows]
        if a.shape[1:] != b.shape[1:]:
            modules.append({"module": name, "call": call,
                            "shape_mismatch": [list(a.shape), list(b.shape)]})
            continue
        entry = {"module": name, "call": call, "rows": int(rows),
                 "truncated": bool(truncated), "pcc": pcc(a, b),
                 "max_rel_err": max_rel_err(a, b)}
        if entry["pcc"] < 0.999 and not truncated and a.ndim == 2:
            # Only worth the extra work when the row-wise number already failed.
            sa, sb = sorted_rows(a), sorted_rows(b)
            entry["pcc_sorted"] = pcc(sa, sb)
            entry["max_rel_err_sorted"] = max_rel_err(sa, sb)
            # Sorting rows cannot rescue a tensor whose *channels* are also ordered
            # differently: the lexicographic key is then a different key on each side.
            # The port's tensor product emits e3nn's output irreps in its own channel
            # order, so for `.tp` outputs both axes are permuted. Sorting every scalar is
            # invariant to both and answers the only question that matters of an
            # activation whose layout is known to differ: are these the same numbers?
            # It cannot localise an error and is never the headline -- the final output
            # and the ranking are.
            entry["pcc_all_sorted"] = pcc(np.sort(a, axis=None), np.sort(b, axis=None))
        modules.append(entry)

    ok = [m for m in modules if "pcc" in m]

    def _final(ref):
        port_order = np.argsort(-got[:, 0], kind="stable")
        ref_order = np.argsort(-ref[:, 0], kind="stable")
        return {
            "pcc": pcc(got, ref),
            "max_rel_err": max_rel_err(got, ref),
            "max_abs_err": float(np.max(np.abs(got.astype(np.float64) - ref))),
            "col0_pcc": pcc(got[:, 0], ref[:, 0]),
            "top1_same": bool(port_order[0] == ref_order[0]),
            "full_order_same": bool(np.array_equal(port_order, ref_order)),
            "port_top1": int(port_order[0]), "ref_top1": int(ref_order[0]),
        }

    def _best(m):
        return max(m.get("pcc", 0.0), m.get("pcc_sorted", 0.0),
                   m.get("pcc_all_sorted", 0.0))

    return {
        "complex": cid, "n_poses": int(got.shape[0]),
        # The rollout capture carries a float32 reference floor of up to 2.5e-01 that is
        # not the port's (see CONFIDENCE.md); the rebuild saw exactly the port's inputs.
        "final_vs_rollout": _final(ref_raw),
        "final_vs_rebuild": _final(rebuild) if rebuild is not None else None,
        "modules": {
            "n_graded": len(ok),
            "min_pcc": min((m["pcc"] for m in ok), default=None),
            "min_pcc_after_sort": min((_best(m) for m in ok), default=None),
            "layout_only": sorted(m["module"] for m in ok
                                  if m.get("pcc", 1.0) < 0.999
                                  and m.get("pcc_all_sorted", 0.0) >= 0.999),
            "max_rel_err": max((m["max_rel_err"] for m in ok), default=None),
            "row_order_only": sorted(m["module"] for m in ok
                                     if m.get("pcc", 1.0) < 0.999
                                     and m.get("pcc_sorted", 0.0) >= 0.999),
            "worst": sorted(ok, key=_best)[:6],
            "shape_mismatches": [m for m in modules if "shape_mismatch" in m],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", action="append", dest="ids", default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    model, cfg = load_model()
    results = []
    for cid in args.ids or list(DEFAULT_IDS):
        if not (GOLD / f"{cid}__s0" / "confidence_inputs.npz").exists():
            print(f"{cid}: no confidence_inputs.npz, run confidence_goldens.py first")
            continue
        r = grade(cid, model, cfg)
        results.append(r)
        m = r["modules"]
        for label in ("final_vs_rollout", "final_vs_rebuild"):
            f = r[label]
            if f is None:
                continue
            print(f"{r['complex']} {label}: PCC {f['pcc']:.8f}  max rel {f['max_rel_err']:.3e}"
                  f"  max abs {f['max_abs_err']:.3e}  top-1 {f['top1_same']}"
                  f"  full order {f['full_order_same']}", flush=True)
        worst_pcc = "n/a" if m["min_pcc"] is None else f"{m['min_pcc']:.6f}"
        sorted_pcc = ("n/a" if m["min_pcc_after_sort"] is None
                      else f"{m['min_pcc_after_sort']:.6f}")
        print(f"   modules {m['n_graded']}  min PCC {worst_pcc}  "
              f"min PCC allowing a reordering {sorted_pcc}  "
              f"row-order-only {len(m['row_order_only'])}  "
              f"layout-only {len(m['layout_only'])}", flush=True)

    if results:
        n = len(results)
        for label in ("final_vs_rollout", "final_vs_rebuild"):
            have = [r for r in results if r[label] is not None]
            if not have:
                continue
            print(f"\n{label}: top-1 {sum(r[label]['top1_same'] for r in have)}/{len(have)}"
                  f"  full order {sum(r[label]['full_order_same'] for r in have)}/{len(have)}"
                  f"  worst PCC {min(r[label]['pcc'] for r in have):.8f}"
                  f"  worst max rel {max(r[label]['max_rel_err'] for r in have):.3e}")
        graded = [r["modules"]["min_pcc_after_sort"] for r in results
                  if r["modules"]["min_pcc_after_sort"] is not None]
        if graded:
            print(f"worst module PCC allowing a reordering: {min(graded):.6f}")
        if len(graded) < len(results):
            print(f"WARNING: {len(results) - len(graded)} of {len(results)} complexes "
                  f"matched no module golden at all -- that is the grader, not the port, "
                  f"and the final-output numbers above are unaffected")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
