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
    """Run the port on every captured pose. Returns `([N, 2], {module: concatenated out})`."""
    gold = GOLD / f"{cid}__s0"
    inp = {k: torch.from_numpy(v) for k, v in np.load(gold / "confidence_inputs.npz").items()}
    poses = np.load(gold / "poses.npz")
    lig_pos = torch.from_numpy(poses["pos"] - poses["center"]).float()

    acts: dict[str, list[np.ndarray]] = defaultdict(list)
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

    # One hook fires once per conv layer per pose, so the per-pose lists line up with
    # upstream's pose-ordered concatenation.
    merged = {n: np.concatenate(v, axis=0) for n, v in acts.items()
              if all(x is not None for x in v) and v}
    return torch.cat(outs).numpy(), merged


def golden_modules(cid: str) -> dict[str, Path]:
    """`{dotted module name: npz}` for the confidence goldens of *cid*, first call only."""
    out = {}
    for p in sorted((GOLD / f"{cid}__mod").glob(f"{cid}__conf__*.npz")):
        m = re.fullmatch(rf"{re.escape(cid)}__conf__(.+)__c(\d+)", p.stem)
        if not m or m.group(2) != "0":
            continue
        out[m.group(1).replace("__", ".")] = p
    return out


def grade(cid: str, model, cfg) -> dict:
    got, acts = run_port(cid, model, cfg)
    ref_raw = np.load(GOLD / f"{cid}__s0" / "poses.npz")["confidence_raw"]

    modules = []
    for name, path in golden_modules(cid).items():
        mine = acts.get(name)
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
            modules.append({"module": name, "shape_mismatch": [list(a.shape), list(b.shape)]})
            continue
        modules.append({"module": name, "rows": int(rows), "truncated": bool(truncated),
                        "pcc": pcc(a, b), "max_rel_err": max_rel_err(a, b)})

    ok = [m for m in modules if "pcc" in m]
    final = {
        "pcc": pcc(got, ref_raw),
        "max_rel_err": max_rel_err(got, ref_raw),
        "max_abs_err": float(np.max(np.abs(got.astype(np.float64) - ref_raw))),
        "col0_pcc": pcc(got[:, 0], ref_raw[:, 0]),
    }
    port_order = np.argsort(-got[:, 0], kind="stable")
    ref_order = np.argsort(-ref_raw[:, 0], kind="stable")
    return {
        "complex": cid, "n_poses": int(got.shape[0]),
        "final": final,
        "rank": {
            "top1_same": bool(port_order[0] == ref_order[0]),
            "full_order_same": bool(np.array_equal(port_order, ref_order)),
            "port_top1": int(port_order[0]), "upstream_top1": int(ref_order[0]),
        },
        "modules": {
            "n_graded": len(ok),
            "min_pcc": min((m["pcc"] for m in ok), default=None),
            "max_rel_err": max((m["max_rel_err"] for m in ok), default=None),
            "worst": sorted(ok, key=lambda m: m["pcc"])[:5],
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
        f, m, k = r["final"], r["modules"], r["rank"]
        print(f"{r['complex']}: final PCC {f['pcc']:.8f}  max rel {f['max_rel_err']:.3e}  "
              f"max abs {f['max_abs_err']:.3e}", flush=True)
        worst_pcc = "n/a" if m["min_pcc"] is None else f"{m['min_pcc']:.8f}"
        worst_rel = "n/a" if m["max_rel_err"] is None else f"{m['max_rel_err']:.3e}"
        print(f"   modules {m['n_graded']}  min PCC {worst_pcc}  max rel {worst_rel}")
        print(f"   top-1 same: {k['top1_same']}   full order same: {k['full_order_same']}")

    if results:
        n = len(results)
        agree = sum(r["rank"]["top1_same"] for r in results)
        print(f"\nrank agreement (top-1): {agree}/{n}")
        print(f"full-order agreement  : {sum(r['rank']['full_order_same'] for r in results)}/{n}")
        print(f"worst final PCC       : {min(r['final']['pcc'] for r in results):.8f}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
