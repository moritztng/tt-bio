#!/usr/bin/env python3
"""of3t-padshape: the width sweep scored -- gradient against the pinned float64, forward against
upstream 0.4.3's own float64, both in the campaign's own denominators, at every width.

One scorer for both halves so the two curves are read off one process and one reference set.
`agreement.py` is imported, not reimplemented, so the gradient side is the scorer that produced
2.159527121735274.
"""
import json
import sys
from pathlib import Path

import torch

W = Path("/home/ttuser/.coworker/wt/of3t-padshape")
sys.path.insert(0, str(W / "perf" / "of3t_trajectory"))
import agreement  # noqa: E402

MODEL_TOTAL_SQ = 10.279642678524981
agreement.MODEL_TOTAL_SQ = MODEL_TOTAL_SQ
F64 = Path("/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt")
EXPECT = "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4"
SECTIONS = list(json.loads(
    (W / "perf/of3t_orchestrator/SECTION_MASS_MEASURED.json").read_text())["sections_pct_of_model"])
O = Path("/tmp/of3t/of3t-padshape")
REF043 = Path("/home/ttuser/of3t_trunk043ref/ref_043_f64_c64.pt")
REAL = 56

h = agreement.sha256(F64)
if h != EXPECT:
    raise SystemExit(f"STOP: {F64} is {h}, the pin says {EXPECT}")
print(f"verified {F64} sha256 {h}", flush=True)

# The arms. `banked` entries are of3t-modelboundary's own tensors, reused rather than re-run;
# every other entry is this row's.
ARMS = [
    ("CTRL_w64_A", 64, 0, O / "dev_CTRL_w64_A.pt"),
    ("CTRL_w64_B", 64, 0, O / "dev_CTRL_w64_B.pt"),
    ("CTRL_w128", 128, 0, O / "dev_CTRL_w128.pt"),
    ("CTRL_w192", 192, 0, O / "dev_CTRL_w192.pt"),
    ("CTRL_w256", 256, 0, O / "dev_CTRL_w256.pt"),
    ("CTRL_w384_banked", 384, 0,
     Path("/tmp/of3t/of3t-modelboundary/dev_CTRL_n384_nocaptures.pt")),
    ("RENORM_w64", 64, 1, O / "dev_RENORM_w64.pt"),
    ("RENORM_w256", 256, 1, O / "dev_RENORM_w256.pt"),
    ("RENORM_w384_banked", 384, 1,
     Path("/tmp/of3t/of3t-modelboundary/dev_RENORM_n384_nocaptures.pt")),
]

ref = torch.load(REF043, map_location="cpu", weights_only=False)
fs = ref["s"][:, :REAL].to(torch.float64).reshape(-1)
fz = ref["z"][:, :REAL, :REAL].to(torch.float64).reshape(-1)
nfs, nfz = float(torch.linalg.vector_norm(fs)), float(torch.linalg.vector_norm(fz))
print(f"0.4.3 float64 forward reference, real block: s {nfs!r} z {nfz!r}", flush=True)

loaded, fwd = {}, {}
for tag, wid, ren, p in ARMS:
    if not p.exists():
        print(f"MISSING {tag}: {p}", flush=True)
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = {k: v.to(torch.float64).reshape(-1) for k, v in d["grads"].items() if v is not None}
    loaded[tag] = (wid, ren, g)
    s, z = d["s"], d["z"]
    a_s = s[:, :REAL].to(torch.float64).reshape(-1)
    a_z = z[:, :REAL, :REAL].to(torch.float64).reshape(-1)

    def m(a, b, nb):
        na = float(torch.linalg.vector_norm(a))
        return {"rel_l2": float(torch.linalg.vector_norm(a - b) / nb), "norm_ratio": na / nb,
                "cos": float((a @ b) / (na * nb)), "ours_norm": na, "ref_norm": nb}
    fwd[tag] = {"width": wid, "renorm": ren,
                "s_real": m(a_s, fs, nfs), "z_real": m(a_z, fz, nfz),
                "s_full_norm": float(s.to(torch.float64).norm()),
                "z_full_norm": float(z.to(torch.float64).norm())}
    print(f"{tag:20s} forward real-block  s rel {fwd[tag]['s_real']['rel_l2']!r}  "
          f"z rel {fwd[tag]['z_real']['rel_l2']!r}", flush=True)
    del d

names = sorted(set().union(*[set(g) for _, _, g in loaded.values()]))
print(f"union {len(names)} tensors; loading the float64 subset", flush=True)
f64 = agreement.load_subset(F64, names)

out = {"what": __doc__.strip().splitlines()[0],
       "host": "tt-quietbox2 (qb2), card 0, p300c", "denominator_model_total_sq": MODEL_TOTAL_SQ,
       "float64_reference": {"path": str(F64), "sha256": h, "matches_pin": True},
       "forward_reference": {"path": str(REF043), "sha256": agreement.sha256(REF043),
                             "policy": ref["policy"], "tree": ref["tree"],
                             "real_block_only": f"{REAL} of the padded width"},
       "n_tensors": len(names), "forward": fwd, "gradient": {}}

for tag, (wid, ren, g) in loaded.items():
    cat = torch.cat([g[k] for k in sorted(g)])
    rows = agreement.pair_rows(f64, g, names, f64, SECTIONS)
    st = agreement.stat(rows, tag)
    out["gradient"][tag] = {
        "width": wid, "renorm": ren, "n": len(g),
        "grad_norm": float(torch.linalg.vector_norm(cat)),
        "mass_weighted_rel_l2": st["mass_weighted_rel_l2"],
        "mass_weighted_norm_ratio": st["mass_weighted_norm_ratio"],
        "mass_weighted_cos": st["mass_weighted_cos"],
        "worst_rel_l2": st["worst_rel_l2"], "worst_tensor": st["worst_tensor"],
        "over_bar_5e-2": sum(1 for r in rows if (r.get("rel_l2") or 0) > 5.0e-2),
        "scored": sum(1 for r in rows if r.get("rel_l2") is not None)}
    print(f"{tag:20s} w={wid:3d} renorm={ren} ||g||={out['gradient'][tag]['grad_norm']!r} "
          f"rel={st['mass_weighted_rel_l2']!r} worst={st['worst_rel_l2']} {st['worst_tensor']}",
          flush=True)

# A16: the zero model, scored on this exact set rather than assumed to be 1
rows = agreement.pair_rows(f64, {}, names, f64, SECTIONS)
st = agreement.stat(rows, "ZERO_MODEL_A16")
out["A16_zero_model"] = {"mass_weighted_rel_l2": st["mass_weighted_rel_l2"],
                         "mass_weighted_norm_ratio": st["mass_weighted_norm_ratio"],
                         "grad_norm": 0.0, "scored": sum(
                             1 for r in rows if r.get("rel_l2") is not None)}
print(f"A16 zero model rel {st['mass_weighted_rel_l2']!r} r {st['mass_weighted_norm_ratio']!r}",
      flush=True)

# A/A: two runs of one arm, same card, same process shape
if "CTRL_w64_A" in loaded and "CTRL_w64_B" in loaded:
    a, b = loaded["CTRL_w64_A"][2], loaded["CTRL_w64_B"][2]
    ident = sum(1 for k in a if k in b and torch.equal(a[k], b[k]))
    worst = max(((float(torch.linalg.vector_norm(a[k] - b[k])), k) for k in a if k in b),
                default=(0.0, None))
    out["A_over_A"] = {"arms": "CTRL_w64_A against CTRL_w64_B, same card, same process shape",
                       "n_compared": len(a), "n_bit_identical": ident,
                       "max_absdiff": worst[0], "worst_tensor": worst[1],
                       "grad_norm_A": out["gradient"]["CTRL_w64_A"]["grad_norm"],
                       "grad_norm_B": out["gradient"]["CTRL_w64_B"]["grad_norm"],
                       "forward_s_rel_A": fwd["CTRL_w64_A"]["s_real"]["rel_l2"],
                       "forward_s_rel_B": fwd["CTRL_w64_B"]["s_real"]["rel_l2"]}
    print("A/A " + json.dumps(out["A_over_A"]), flush=True)

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("wrote " + sys.argv[1])
