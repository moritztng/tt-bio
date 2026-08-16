import sys, torch, numpy as np, pathlib
base = pathlib.Path("/home/moritz/.coworker/wt/protenix-v2-nondeterminism-rootcause/perf/nondet/out")

def load(run, key):
    p = base / run / "dump" / f"cond_{key}.pt"
    return torch.load(p, weights_only=True).float() if p.exists() else None

runs = ["dump256_a", "dump256_b", "dump256_c"]
pz = {r: load(r, "pair_z") for r in runs}
for r in runs:
    print(r, "pair_z", None if pz[r] is None else tuple(pz[r].shape))

def wrong_cells(a, b, thr=1e-3):
    d = (a - b).abs().amax(-1)  # (N,N)
    rows, cols = torch.nonzero(d > thr, as_tuple=True)
    return d, set(rows.tolist()), set(cols.tolist())

pats = {}
for x, y in [("dump256_a","dump256_b"), ("dump256_a","dump256_c"), ("dump256_b","dump256_c")]:
    if pz[x] is None or pz[y] is None: continue
    d, rows, cols = wrong_cells(pz[x], pz[y])
    pats[(x,y)] = (rows, cols)
    rmod = sorted({r % 32 for r in rows})
    print(f"{x} vs {y}: cells>{1e-3}: rows={len(rows)} cols={len(cols)} max|d|={d.max():.4f} rows%32={rmod}")

# pattern repeat: intersection of wrong-row sets across pairs
if len(pats) == 3:
    r_ab, _ = pats[("dump256_a","dump256_b")]
    r_ac, _ = pats[("dump256_a","dump256_c")]
    r_bc, _ = pats[("dump256_b","dump256_c")]
    print("wrong-row overlap ab&ac:", len(r_ab & r_ac), " ab&bc:", len(r_ab & r_bc), " ac&bc:", len(r_ac & r_bc))
    print("rows ab:", sorted(r_ab)); print("rows ac:", sorted(r_ac)); print("rows bc:", sorted(r_bc))

# which other cond keys differ per pair
import glob
keys = sorted({p.name[5:-3] for p in (base/"dump256_a"/"dump").glob("cond_*.pt")})
for k in keys:
    ts = {r: load(r, k) for r in runs}
    if any(t is None for t in ts.values()): continue
    same_ab = torch.equal(ts["dump256_a"], ts["dump256_b"])
    same_ac = torch.equal(ts["dump256_a"], ts["dump256_c"])
    same_bc = torch.equal(ts["dump256_b"], ts["dump256_c"])
    print(f"cond[{k}]: a==b {same_ab}  a==c {same_ac}  b==c {same_bc}")
