"""pfm-ttfast accuracy: every lever against exact, sample by sample at the same seed, beside the seed floor.

Same seed = same initial noise and the same sampler draws, so sample k of a lever arm and sample k of exact
differ only by the lever. The floor is exact against exact at a different seed, which is the variation a user
already accepts by picking a seed. Two numbers per pair, both Kabsch-aligned:
  all  : all real atoms, superposed on all of them
  bind : chain B (the 150 aa binder) after superposing on chain A (the target), the interface view
usage: compare.py RUN_DIR [RUN_DIR ...]   (reads coords_*.pt from every c*/ under each)
"""
import glob, itertools, json, sys
from pathlib import Path
import torch

def kabsch(P, Q):
    """R, t minimising |(P - cP) R + cQ - Q| (rows are points)."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=P.dtype))
    R = U @ D @ Vt
    return R, cP, cQ

def rmsd_aligned(P, Q, fit, score):
    R, cP, cQ = kabsch(P[fit], Q[fit])
    Pa = (P - cP) @ R + cQ
    return float(((Pa[score] - Q[score]) ** 2).sum(-1).mean().sqrt())

runs = {}
for rd in sys.argv[1:]:
    for f in glob.glob(f"{rd}/c*/coords_*.pt"):
        name = Path(f).stem[len("coords_"):]           # arm_sSEED_kind
        arm, rest = name.rsplit("_s", 1); seed, kind = rest.split("_", 1)
        if kind == "census":
            continue
        d = torch.load(f, weights_only=False)
        runs.setdefault((arm, int(seed)), []).append((f, d))

def masks(d):
    f = d["feats"]; n = d["coords"].shape[-2]
    a2t = next((f[k] for k in ("atom_to_token_idx", "atom_to_token") if k in f), None)
    asym = f.get("asym_id")
    a2t = a2t.reshape(-1)[:n].long()
    if a2t.dim() == 2: a2t = a2t.argmax(-1)
    chain = asym.reshape(-1)[a2t].long()
    real = torch.ones(n, dtype=torch.bool)
    for k in ("atom_mask", "ref_mask"):
        if k in f and f[k].numel() >= n:
            real &= f[k].reshape(-1)[:n].bool()
    ids = sorted(set(chain[real].tolist()))
    return real, real & (chain == ids[0]), real & (chain == ids[-1])

def pair(da, db):
    A, B = da["coords"].double(), db["coords"].double()
    real, cA, cB = masks(da)
    out = []
    for k in range(A.shape[0]):
        out.append(dict(all=rmsd_aligned(A[k], B[k], real, real), bind=rmsd_aligned(A[k], B[k], cA, cB)))
    return out

def summ(rows):
    import statistics as st
    return {m: dict(mean=round(st.mean(r[m] for r in rows), 3), max=round(max(r[m] for r in rows), 3))
            for m in ("all", "bind")}

res = {"n_runs": {f"{a}_s{s}": len(v) for (a, s), v in runs.items()}}
# determinism: two runs of the same arm and seed (cold vs warm, or two chips)
det = {}
for key, v in runs.items():
    for (fa, da), (fb, db) in itertools.combinations(v, 2):
        det[f"{key[0]}_s{key[1]} {Path(fa).parent.name}/{Path(fa).stem[-4:]} vs {Path(fb).parent.name}/{Path(fb).stem[-4:]}"] = \
            float((da["coords"] - db["coords"]).abs().max())
res["determinism_maxabs_A"] = det
ex = {s: v[0][1] for (a, s), v in runs.items() if a == "exact"}
res["floor"] = {f"s{a} vs s{b}": summ(pair(ex[a], ex[b])) for a, b in itertools.combinations(sorted(ex), 2)}
res["levers"] = {}
for (arm, s), v in sorted(runs.items()):
    if arm == "exact" or s not in ex:
        continue
    res["levers"][f"{arm}_s{s}"] = summ(pair(v[0][1], ex[s]))
res["conf"] = {f"{a}_s{s}": v[0][1]["conf"] for (a, s), v in sorted(runs.items())}
print(json.dumps(res, indent=1))
