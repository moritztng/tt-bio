#!/usr/bin/env python3
"""Phase 4 label: conformational basin clustering + basin occupancy.

Clusters a fold's N diffusion samples into conformational basins by structural
similarity, using the pairwise DockQ matrix from scripts/abag_xm_pairwise_matrix.py
as the similarity. Distance = 1 - DockQ (symmetrised). DBSCAN (scipy) on the precomputed
distance matrix; eps and min_samples are tunable (defaults eps=0.1, min_samples=2,
i.e. samples within DockQ>=0.9 of each other form a basin). Reports per-sample cluster
labels (-1 = noise) and basin occupancy = fraction of samples in the largest non-noise
cluster (a scalar summarising how concentrated the ensemble is in one basin; a high
occupancy means the samples collapsed to a single conformational basin, a low occupancy
means multiple basins / diverse ensemble).

Usage:
    python3 scripts/abag_xm_basin_clust.py <pairwise_matrix.json> [--eps 0.1] [--min_samples 2] [--out json]
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_json")
    ap.add_argument("--eps", type=float, default=0.1,
                     help="DBSCAN eps on distance = 1 - DockQ (default 0.1 => DockQ>=0.9 same basin)")
    ap.add_argument("--min_samples", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    doc = json.loads(Path(a.matrix_json).read_text())
    n = doc["n_samples"]
    rows = doc["matrix"]
    # build symmetric distance matrix
    D = [[0.0] * n for _ in range(n)]
    for r in rows:
        i, j, dq = r["i"], r["j"], r.get("dockq")
        if dq is None:
            dq = 0.0
        d = 1.0 - dq
        D[i][j] = d
        D[j][i] = d

    try:
        from sklearn.cluster import DBSCAN
        import numpy as np
        labels = DBSCAN(eps=a.eps, min_samples=a.min_samples,
                         metric="precomputed").fit_predict(np.asarray(D))
        labels = labels.tolist()
    except Exception as e:
        # fallback: simple connected-components clustering at the eps threshold
        labels = [-1] * n
        visited = [False] * n
        cid = 0
        for i in range(n):
            if visited[i]:
                continue
            comp = []
            stack = [i]
            while stack:
                u = stack.pop()
                if visited[u]:
                    continue
                visited[u] = True
                comp.append(u)
                for v in range(n):
                    if not visited[v] and D[u][v] <= a.eps:
                        stack.append(v)
            if len(comp) >= a.min_samples:
                for u in comp:
                    labels[u] = cid
                cid += 1
            # else leave as -1 (noise)

    # basin occupancy = fraction in the largest non-noise cluster
    from collections import Counter
    cnt = Counter(l for l in labels if l >= 0)
    occ = (max(cnt.values()) / n) if cnt else 0.0
    n_clusters = len(cnt)
    out = {"target": doc.get("target"), "n_samples": n,
           "eps": a.eps, "min_samples": a.min_samples,
           "n_clusters": n_clusters,
           "basin_occupancy": round(occ, 6),
           "labels": labels,
           "cluster_sizes": {str(k): v for k, v in sorted(cnt.items())}}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
