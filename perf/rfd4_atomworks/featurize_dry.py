"""CPU dry run of RF3 serving-path featurization (the spec _predict_rf3_one builds), for one input.

Usage: PYTHONPATH=<tree> python featurize_dry.py <input.yaml> <msa_dir> <out.pt>
Saves the flat feature dict so the two atomworks trees can be diffed key by key."""
import json, sys, tempfile
from pathlib import Path
import torch
from tt_bio.main import _read_bio_chains, _read_bio_constraints, _resolve_a3m_path
from tt_bio.worker import _rf3_bonds, _rf3_sequence
from tt_bio.rf3.featurize import featurize

path, msa_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
comps = []
for cid, cseq, spec, mt, mods in _read_bio_chains(path):
    if mt == "ligand":
        comps.append({"ccd_code": cseq[4:]} if cseq.startswith("CCD_") else {"smiles": cseq})
        comps[-1]["chain_id"] = cid
        continue
    c = {"seq": _rf3_sequence(cseq, mods), "chain_id": cid}
    a3m = _resolve_a3m_path(spec, cseq, msa_dir)
    if a3m:
        c["msa_path"] = str(a3m.resolve())
    comps.append(c)
with tempfile.TemporaryDirectory() as td:
    sp = Path(td) / f"{path.stem}.json"
    sp.write_text(json.dumps([{"name": path.stem, "components": comps,
                               "bonds": _rf3_bonds(comps, _read_bio_constraints(path))}]))
    o = featurize(sp, n_recycles=10, diffusion_batch_size=5, seed=0)[0]
flat = {}
def walk(x, p=""):
    if isinstance(x, torch.Tensor): flat[p.rstrip("/")] = x
    elif isinstance(x, dict):
        for k, v in x.items(): walk(v, f"{p}{k}/")
walk({k: v for k, v in o.items() if k != "atom_array"})
aa = o["atom_array"]
flat["_meta"] = {k: list(map(str, getattr(aa, k))) for k in ("chain_id", "res_name", "res_id", "atom_name")}
flat["_meta"]["charge"] = [int(c) for c in aa.charge]
torch.save(flat, out)
print("OK", path.name, "atoms", len(aa), "keys", len(flat))
