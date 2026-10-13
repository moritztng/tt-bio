#!/usr/bin/env python3
"""Featurise the 512 page fixture repeatedly in one process; time it and dump features."""
import argparse, json, os, random, sys, tempfile, time
import numpy as np, torch, yaml
ap = argparse.ArgumentParser()
ap.add_argument("--fixture", default="perf/size512/fixtures/cdk2x2_512")
ap.add_argument("--reps", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dump", default=None)
a = ap.parse_args()
root = os.environ["PYTHONPATH"].split(":")[0]
fx = os.path.join(root, a.fixture)
seq = yaml.safe_load(open(fx + ".yaml"))["sequences"][0]["protein"]["sequence"]
from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet
from tt_bio.openfold3_data import build_openfold3_features, normalize_openfold3_msa_paths
from tt_bio._vendor.openfold3.core.data.primitives.structure import query as Q
msa_dir = tempfile.mkdtemp(prefix="of3conf-msa-")
q = {"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq,
                 "main_msa_file_paths": [fx + ".a3m"]}]}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump({"queries": {"q": q}}, fh)
pool_t = []
orig = Q._pooled_reference_molecules
def timed(jobs):
    s = time.perf_counter(); r = orig(jobs); pool_t.append(time.perf_counter() - s); return r
Q._pooled_reference_molecules = timed
tot = []
for rep in range(a.reps):
    query = next(iter(InferenceQuerySet.from_json(fh.name).queries.values()))
    query = normalize_openfold3_msa_paths(query, msa_dir)
    torch.manual_seed(0); np.random.seed(0); random.seed(a.seed)
    s = time.perf_counter()
    feats = build_openfold3_features(query)
    tot.append(time.perf_counter() - s)
    if rep == 0 and a.dump:
        torch.save({k: v for k, v in feats.items() if torch.is_tensor(v)}, a.dump)
    if rep == 0:
        first = {k: v.clone() for k, v in feats.items() if torch.is_tensor(v)}
    else:
        for k, v in first.items():
            assert torch.equal(v, feats[k]), f"rep {rep} differs on {k}"
w = slice(1, None)
print(f"impl={getattr(Q,'CONFORMER_IMPL','thread')} workers={Q.CONFORMER_THREADS} "
      f"build_openfold3_features warm median {np.median(tot[w])*1e3:.0f} ms (all {[round(x*1e3) for x in tot]}) | "
      f"pooled conformers warm median {np.median(pool_t[w])*1e3:.0f} ms (all {[round(x*1e3) for x in pool_t]})", flush=True)
