#!/usr/bin/env python3
"""The pairformer_stack section alone, per D174 arm, in the model bundle's denominator.

`score.sh` puts the trunk into the whole-model union, which is the right artifact and reads
three references end to end. This answers the one number AMENDMENT 2 predicts on
(`pairformer_stack` 2.159527121735274 against float64, upstream's own bf16 at 0.3147698293887927)
by loading only the 2,736 tensors of that section, so the answer does not wait on the assembly.

Same scorer -- `of3t-trajectory`'s `agreement.py`, imported not reimplemented -- same denominator
10.279642678524981, same pinned references verified by sha256 before loading.
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path("/home/ttuser/.coworker/wt/of3t-ditmodel")
sys.path.insert(0, str(REPO / "perf" / "of3t_trajectory"))
import agreement  # noqa: E402

MODEL_TOTAL_SQ = 10.279642678524981
agreement.MODEL_TOTAL_SQ = MODEL_TOTAL_SQ
PIN = Path("/home/ttuser/of3t_refprec/pinned_p175")
F64 = Path("/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt")
BF16 = PIN / "arm4_bf16_autocast/grads_f64.pt"
F32 = PIN / "arm2_f32_upstream/grads_f64.pt"
EXPECT = {
    str(F64): "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4",
    str(BF16): "ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb",
    str(F32): "09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548",
}
SECTIONS = json.loads((REPO / "perf/of3t_orchestrator/SECTION_MASS_MEASURED.json").read_text())
SECTIONS = list(SECTIONS["sections_pct_of_model"])
O = Path("/tmp/of3t/of3t-ditmodel")

inputs = {}
for p in (F64, BF16, F32):
    h = agreement.sha256(p)
    if h != EXPECT[str(p)]:
        raise SystemExit(f"STOP: {p} is {h}, the pin says {EXPECT[str(p)]}")
    inputs[p.name if p is F64 else str(p)] = {"path": str(p), "sha256": h, "matches_pin": True}
    print(f"verified {p}", flush=True)

arms = {}
for tag in ("MASKOFF", "MASKON", "MASKONES"):
    f = O / f"dev_{tag}_n384.pt"
    if not f.exists():
        print(f"skipping {tag}: no {f}", flush=True)
        continue
    d = torch.load(f, map_location="cpu", weights_only=False)
    g = d["grads"]
    arms[tag] = {k: v.to(torch.float64).reshape(-1) for k, v in g.items() if v is not None}
    print(f"{tag}: {len(arms[tag])} tensors", flush=True)
    del d, g

names = sorted(set().union(*[set(a) for a in arms.values()]))
print(f"union {len(names)} tensors", flush=True)

print("loading float64 subset", flush=True)
f64 = agreement.load_subset(F64, names)
refs = {"FLOAT64": f64,
        "UPSTREAM_BF16": agreement.load_subset(BF16, names),
        "UPSTREAM_F32": agreement.load_subset(F32, names)}
arms["ZERO_MODEL"] = {}

out = {"what": __doc__.strip().splitlines()[0], "host": "tt-quietbox2",
       "denominator_model_total_sq": MODEL_TOTAL_SQ, "inputs": inputs,
       "n_tensors": len(names), "stats": {}}
for tag in list(arms):
    for rk in ("FLOAT64", "UPSTREAM_BF16"):
        rows = agreement.pair_rows(refs[rk], arms[tag], names, f64, SECTIONS)
        out["stats"][f"{tag}_vs_{rk}"] = agreement.stat(rows, f"{tag}_vs_{rk}")
rows = agreement.pair_rows(f64, refs["UPSTREAM_BF16"], names, f64, SECTIONS)
out["stats"]["UPSTREAM_BF16_vs_FLOAT64"] = agreement.stat(rows, "UPSTREAM_BF16_vs_FLOAT64")
rows = agreement.pair_rows(f64, refs["UPSTREAM_F32"], names, f64, SECTIONS)
out["stats"]["UPSTREAM_F32_vs_FLOAT64"] = agreement.stat(rows, "UPSTREAM_F32_vs_FLOAT64")

# the break control is a BIT question, not a statistic
if "MASKOFF" in arms and "MASKONES" in arms:
    off, ones = arms["MASKOFF"], arms["MASKONES"]
    diff = {k: float(torch.linalg.vector_norm(ones[k] - off[k]))
            for k in off if k in ones}
    out["ones_control"] = {
        "n_compared": len(diff),
        "n_bit_identical": sum(1 for k in off if k in ones and torch.equal(ones[k], off[k])),
        "max_absdiff": max(diff.values()) if diff else None,
        "worst_tensor": max(diff, key=diff.get) if diff else None,
    }
if "MASKOFF" in arms and "MASKON" in arms:
    off, on = arms["MASKOFF"], arms["MASKON"]
    diff = {k: float(torch.linalg.vector_norm(on[k] - off[k])) for k in off if k in on}
    out["maskon_vs_maskoff"] = {
        "n_compared": len(diff),
        "n_bit_identical": sum(1 for k in off if k in on and torch.equal(on[k], off[k])),
        "max_absdiff": max(diff.values()) if diff else None,
        "worst_tensor": max(diff, key=diff.get) if diff else None,
        "our_grad_norm_MASKOFF": float(torch.linalg.vector_norm(
            torch.cat([off[k] for k in sorted(off)]))),
        "our_grad_norm_MASKON": float(torch.linalg.vector_norm(
            torch.cat([on[k] for k in sorted(on)]))),
    }

json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out["stats"].items():
    print(f"{k:34s} rel {v['mass_weighted_rel_l2']!r:24s} r {v['mass_weighted_norm_ratio']!r:22s} "
          f"cos {v['mass_weighted_cos']} worst {v['worst_rel_l2']} {v['worst_tensor']}")
for k in ("ones_control", "maskon_vs_maskoff"):
    if k in out:
        print(k, json.dumps(out[k]))
