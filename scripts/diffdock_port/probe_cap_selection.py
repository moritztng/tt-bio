"""What subset does torch_cluster's ``max_num_neighbors`` cap actually keep?

`probe_radius_truncation.py` answers *whether* the cap binds on real ligands (it does, on 1
of 428). This answers *what it keeps when it binds*, which is the question that decides
whether the port can reproduce it at all.

Four properties, each a yes/no the port's design depends on:

* is the kept subset the first ``cap`` by **index order**?
* is it the **nearest** ``cap``?
* is it stable under a **relabelling** of the same points?
* is it stable under a **rotation** of the same points?

Run in the reference env, which is the only one with torch_cluster:

    ~/scratch/dfd/venv-ref/bin/python -I scripts/diffdock_port/probe_cap_selection.py \
        --out state/dfd/evidence/cap-selection.json
"""
import argparse
import json

import torch
from scipy.spatial.transform import Rotation
from torch_cluster import radius_graph

CUTOFF = 5.0
CAP = 32


def kept_sets(pos, cutoff=CUTOFF, cap=CAP):
    """``{node: [neighbour, ...]}`` as torch_cluster returns them, in its own row order."""
    ei = radius_graph(pos, cutoff, max_num_neighbors=cap)
    out = {}
    for src, dst in zip(ei[0].tolist(), ei[1].tolist()):
        out.setdefault(dst, []).append(src)
    return out, ei


def true_neighbours(pos, cutoff=CUTOFF):
    d = torch.cdist(pos.double(), pos.double())
    inside = d < cutoff
    inside.fill_diagonal_(False)
    return [inside[i].nonzero().flatten().tolist() for i in range(len(pos))]


def probe(n=60, box=2.0, seed=0, cutoff=CUTOFF, cap=CAP):
    """A cloud small enough that every node's true degree is n-1, so the cap always binds."""
    torch.manual_seed(seed)
    pos = (torch.rand(n, 3) * box).float()
    truth = true_neighbours(pos, cutoff)
    kept, ei = kept_sets(pos, cutoff, cap)
    d = torch.cdist(pos.double(), pos.double())

    by_index = sum(1 for i in range(n) if sorted(kept.get(i, [])) == sorted(truth[i])[:cap])
    by_distance = 0
    for i in range(n):
        nearest = sorted(truth[i], key=lambda j: d[i, j].item())[:cap]
        if sorted(kept.get(i, [])) == sorted(nearest):
            by_distance += 1

    perm = torch.randperm(n)
    permuted, _ = kept_sets(pos[perm], cutoff, cap)
    relabelled = {perm[p].item(): sorted(perm[s].item() for s in v)
                  for p, v in permuted.items()}
    stable_perm = sum(1 for i in range(n) if relabelled.get(i, []) == sorted(kept.get(i, [])))
    moved_edges = sum(len(set(relabelled.get(i, [])) ^ set(kept.get(i, []))) for i in range(n))

    rot = torch.tensor(Rotation.random(1, random_state=3).as_matrix()[0], dtype=torch.float32)
    rotated, _ = kept_sets(pos @ rot.T, cutoff, cap)
    stable_rot = sum(1 for i in range(n) if sorted(rotated.get(i, [])) == sorted(kept.get(i, [])))

    shifted, _ = kept_sets(pos + 100.0, cutoff, cap)
    stable_shift = sum(1 for i in range(n)
                       if sorted(shifted.get(i, [])) == sorted(kept.get(i, [])))

    repeat, ei2 = kept_sets(pos, cutoff, cap)

    degrees = [len(v) for v in truth]
    return {
        "n_nodes": n,
        "cutoff_a": cutoff,
        "cap": cap,
        "true_degree_min": min(degrees),
        "true_degree_max": max(degrees),
        "kept_degree_max": max(len(v) for v in kept.values()),
        "kept_edges_total": int(ei.shape[1]),
        "nodes_keeping_first_cap_by_index": by_index,
        "nodes_keeping_nearest_cap": by_distance,
        "nodes_stable_under_relabelling": stable_perm,
        "edges_moved_by_relabelling": moved_edges,
        "nodes_stable_under_rotation": stable_rot,
        "nodes_stable_under_translation_100a": stable_shift,
        "deterministic_on_repeat_call": bool(torch.equal(ei, ei2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nodes", type=int, default=60)
    args = ap.parse_args()
    result = probe(n=args.nodes)
    result["torch_cluster"] = __import__("torch_cluster").__version__
    result["torch"] = torch.__version__
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
