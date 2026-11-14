"""Host-only profile of OpenFold3's featurisation (build_openfold3_features) on a fixture, no device.

python perf/of3_host/featprof.py perf/size512/fixtures/cdk2x2_512.yaml [--openbind] [--reps 3] [--prof out.prof] [--dump f.pt]
Prints wall per rep; the last rep's cProfile top-30 cumulative.
"""
import argparse, cProfile, json, pstats, random, tempfile, time
from pathlib import Path

import numpy as np
import torch

from tt_bio.main import _read_bio_chains
from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet
from tt_bio.openfold3_data import build_openfold3_features, normalize_openfold3_msa_paths, resolve_openfold3_msas

ap = argparse.ArgumentParser()
ap.add_argument("yaml"); ap.add_argument("--openbind", action="store_true")
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--prof")
ap.add_argument("--dump", help="torch.save the last rep's tensor features here (exactness checks)")
a = ap.parse_args()
path = Path(a.yaml).resolve()
msa_dir = Path(tempfile.mkdtemp())
chains = []
for cid, cseq, spec, mt, mods in _read_bio_chains(path):
    p = spec if spec and Path(spec).exists() else (str(path.with_suffix(".a3m")) if path.with_suffix(".a3m").exists() else None)
    chains.append({"molecule_type": mt.upper(), "chain_ids": [cid], "non_canonical_residues": None,
                   "paired_msa_file_paths": None, "template_alignment_file_path": None,
                   "template_entry_chain_ids": None, "sdf_file_path": None, "sequence": cseq, "smiles": None,
                   "ccd_codes": None, "main_msa_file_paths": [p] if p else None})
query = {"query_name": path.stem, "use_msas": True, "use_paired_msas": False, "use_main_msas": True,
         "covalent_bonds": None, "chains": chains}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump({"queries": {path.stem: query}}, fh)
for r in range(a.reps):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    q = next(iter(InferenceQuerySet.from_json(fh.name).queries.values()))
    q = normalize_openfold3_msa_paths(q, msa_dir, openbind=a.openbind)
    q = resolve_openfold3_msas(q, msa_dir, target_id=path.stem, fetch=False)
    pr = cProfile.Profile() if r == a.reps - 1 else None
    t = time.perf_counter()
    if pr: pr.enable()
    f = build_openfold3_features(q, openbind=a.openbind)
    if pr: pr.disable()
    print(f"rep {r}: {time.perf_counter() - t:.3f} s, {len(f)} features, restype {tuple(f['restype'].shape)}", flush=True)
st = pstats.Stats(pr).sort_stats("cumulative"); st.print_stats(30)
if a.prof: pr.dump_stats(a.prof)
if a.dump: torch.save({k: v for k, v in f.items() if isinstance(v, torch.Tensor)}, a.dump)
