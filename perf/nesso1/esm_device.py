"""ESM-2 650M on the card vs the CPU fp32 reference Nesso-1 conditioned on until now.

    TT_VISIBLE_DEVICES=N python perf/nesso1/esm_device.py --out esm_device.json

Per length: rel Frobenius error, min per-token cosine, max |delta|, and the warm device forward
(median of 5) beside the CPU forward (median of 3), both on the page fixture's CDK2 and crops of it.
"""
import argparse
import json
import statistics
import time

import torch
import yaml

from tt_bio.main import ensure_p300_mesh_descriptor

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--fixture", default="perf/nesso1/inputs/ladder/aa512/cdk2_512.yaml")
ap.add_argument("--lengths", default="64,300,512")
a = ap.parse_args()
ensure_p300_mesh_descriptor()
import tt_bio.nesso1_input as NI
from tt_bio._vendor.nesso.data.esm import extract_esm_embedding, setup_esm_model

y = yaml.safe_load(open(a.fixture))
full = next(s["protein"]["sequence"] for s in y["sequences"] if "protein" in s)
cpu, tok = setup_esm_model(NI.ESM2_MODEL, torch.device("cpu"))
t = time.perf_counter()
dev = NI.device_esm()
res = {"load_s": round(time.perf_counter() - t, 3), "rows": []}
for n in (int(x) for x in a.lengths.split(",")):
    seq = full[:n]
    ref = extract_esm_embedding(seq, cpu, tok)[0].double()
    tc = []
    for _ in range(3):
        t = time.perf_counter(); extract_esm_embedding(seq, cpu, tok); tc.append(time.perf_counter() - t)
    td = []
    for _ in range(6):
        t = time.perf_counter(); e = dev(seq); td.append(time.perf_counter() - t)
    e = e[0].double()
    assert e.shape == ref.shape, (e.shape, ref.shape)
    cos = torch.nn.functional.cosine_similarity(e, ref, dim=-1)
    row = {"len": n, "rel_fro": float((e - ref).norm() / ref.norm()), "min_cos": float(cos.min()),
           "max_abs": float((e - ref).abs().max()), "ref_absmax": float(ref.abs().max()),
           "device_cold_s": round(td[0], 3), "device_s": round(statistics.median(td[1:]), 4),
           "cpu_s": round(statistics.median(tc), 3), "cpu_threads": torch.get_num_threads()}
    res["rows"].append(row)
    print(json.dumps(row), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
