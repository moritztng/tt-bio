"""ESMC-300M cold start the way `tt-bio embed --model esmc-300m` runs it on one card: load, then embed twice.

    python perf/startup/esmc_run.py INPUT.yaml OUT.jsonl

Embeds every protein chain of INPUT (the SPD c730 complex: 580 + 150 aa). The first embed call is the first
prediction, the second the steady state. Phase accounting comes from perf/startup/trace (ST_OUT).
"""
import json, sys, time
import yaml

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
from tt_bio import esmc  # noqa: E402

doc = yaml.safe_load(open(sys.argv[1]))
seqs = {s["protein"]["id"]: s["protein"]["sequence"] for s in doc["sequences"] if "protein" in s}
log = open(sys.argv[2], "a")
m = esmc.load_esmc("esmc-300m", fast=False, trace=esmc.trace_pays(seqs))
for k in range(2):
    t = time.perf_counter()
    r = esmc.embed_sequences(m, seqs)
    log.write(json.dumps(dict(ev="rep", k=k, s=time.perf_counter() - t, n=len(r),
                              dim=int(getattr(r[0], "embedding", r[0]).shape[-1]) if r else None)) + "\n")
log.close()
