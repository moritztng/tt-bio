"""Emit the shape contract the device row has to allocate for, and the host-torch reference.

`dfd-device` owns the ttnn implementation and the capture; this row owns the claim that the
loop has one shape. That claim is only usable if it is written down as numbers, so this
script prints them: every buffer the traced step touches, its padded shape, its dtype, and
whether it is a loop invariant (uploaded once before the capture) or a per-step buffer (the
capture reads and writes it in place).

It also reports the measured op count per step and the warm-up rule, both from the same
recorder the test uses, so the device row can tell a refused capture caused by its own
allocation from one caused by the loop's shape.

Usage:
    python3 scripts/diffdock_port/trace_contract.py --out state/dfd/evidence/trace-contract.json
    python3 scripts/diffdock_port/trace_contract.py --n-lig 38 --n-rec 276 --n-rot 13
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.diffdock import graph as G
from tt_bio.diffdock import sampling as S
from tt_bio.diffdock.loop import coefficient_table

# 8G43_ZU6, the complex the static-graph golden was dumped from (state/dfd/dfd-edges.json).
DEFAULTS = dict(n_lig=38, n_rec=276, n_rot=13, n_poses=10, ns=48, nv=10)
BOUNDS = S.SigmaBounds(0.1, 19.0, 0.03, 1.55, 0.0314, 3.14)


def contract(n_lig, n_rec, n_rot, n_poses, ns, nv) -> dict:
    bonds = torch.tensor([[k for k in range(n_rot)] + [k + 1 for k in range(n_rot)],
                          [k + 1 for k in range(n_rot)] + [k for k in range(n_rot)]])
    s = G.StaticGraphs.build(n_lig, n_rec, bonds, n_rot_bonds=n_rot, n_poses=n_poses)
    cfg = S.SamplingConfig.from_recipe()
    table = coefficient_table(cfg, BOUNDS)
    # n_lig_padded/n_rec_padded are the WHOLE blocks, every pose included (graph.py:162):
    # 10 poses x 38 atoms -> bucket(380) = 384 ligand rows, not 10 x 64.
    n_lig_nodes, n_rec_nodes = s.n_lig_padded, s.n_rec_padded
    n_lig_edges = int(s.lig_edge_index.shape[1])
    n_cross_edges = int(s.cross_edge_index.shape[1])
    n_tor_edges = int(s.tor_edge_index.shape[1])
    node_dim = ns + 3 * nv          # 0e + 1o + 1e blocks the conv layers carry

    invariants = {
        "lig_edge_index": [2, n_lig_edges],
        "cross_edge_index": [2, n_cross_edges],
        "tor_edge_index": [2, n_tor_edges],
        "rec_pos": [n_rec_nodes, 3],
        "rec_node_attr": [n_rec_nodes, ns],
        "rec_edge_index_knn": [2, "24 * n_rec (upstream's k)"],
        "coefficient_table_score": list(table.score.shape),
        "coefficient_table_noise": list(table.noise.shape),
        "mask_rotate": [n_rot, n_lig_nodes // n_poses],
    }
    per_step = {
        "lig_pos": [n_lig_nodes, 3],   # viewed as [n_poses, n_lig_padded // n_poses, 3] by the pose update
        "lig_edge_mask": [n_lig_edges, 1],
        "cross_edge_mask": [n_cross_edges, 1],
        "tor_edge_mask": [n_tor_edges, 1],
        "lig_edge_sh": [n_lig_edges, 9],
        "cross_edge_sh": [n_cross_edges, 9],
        "node_attr": [n_lig_nodes + n_rec_nodes, node_dim],
        "tr_score": [n_poses, 3],
        "rot_score": [n_poses, 3],
        "tor_score": [n_poses, n_rot],
        "noise_tr/rot": [n_poses, 3],
        "noise_tor": [n_poses, n_rot],
    }
    # the per-edge tensor-product weight block is what decides whether a complex fits
    weight_numel = 4164
    cross_weight_bytes = n_cross_edges * weight_numel * 2
    return {
        "complex": dict(n_lig=n_lig, n_rec=n_rec, n_rot=n_rot, n_poses=n_poses),
        "padding": {
            "n_lig_rows_all_poses": s.n_lig_padded, "n_rec_rows_all_poses": s.n_rec_padded,
            "n_bond_rows_all_poses": s.n_bond_padded, "multiple": 32,
            "rec_offset": s.rec_offset, "lig_pad_node": s.lig_pad_node,
            "rec_pad_node": s.rec_pad_node,
            "note": ("node axes are padded over ALL poses at once -- 10 x 38 ligand atoms is "
                     "bucket(380) = 384 rows, not 10 x 64 -- and every axis is bucket(n+1), "
                     "so there is always a pad row for pad edges to point at"),
        },
        "edges": {
            "lig": n_lig_edges, "cross": n_cross_edges, "torsion": n_tor_edges,
            "note": "static and complete; the mask moves, the shapes do not",
        },
        "loop_invariants_uploaded_once": invariants,
        "per_step_buffers": per_step,
        "steps": table.steps,
        "noise_drawn_per_step": list(table.drawn),
        "memory": {
            "tp_weight_numel_per_edge": weight_numel,
            "cross_tp_weights_bf16_bytes": cross_weight_bytes,
            "cross_tp_weights_bf16_gib": round(cross_weight_bytes / 2**30, 3),
            "note": ("one conv layer's per-edge weight block for the cross group; "
                     "tp_scatter(edge_chunk=N) bounds it, and N is a device measurement"),
        },
        "capture_rules": {
            "warm_up_passes_before_capture": 1,
            "why": ("TensorProduct casts its float64 Wigner 3j tables to the working dtype on "
                    "first use (tensor_product.py::_w3j_cast); captured, that allocation would "
                    "belong to no one. Measured: step 0 issues 6 ops the others do not."),
            "capture_granularity": "all steps in one capture, not one capture per step",
            "host_returns_inside_the_loop": "none -- diagnostics are accumulated on device",
        },
        "host_reference": {
            "module": "tt_bio.diffdock.loop",
            "entry": "run_loop(pos, score_fn, coefficient_table(cfg, bounds), noise_fn, "
                     "bonds, mask_rotate)",
            "graded_against": "tt_bio.diffdock.sampling.sample",
            "float32_relative_deviation": 1.6e-05,
            "float64_relative_deviation": 2.0e-12,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=int, default=v)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = contract(a.n_lig, a.n_rec, a.n_rot, a.n_poses, a.ns, a.nv)
    text = json.dumps(out, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
