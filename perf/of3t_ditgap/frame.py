"""of3t-ditgap: is the DiT leg in-frame? Two float64 arms of the same function must agree.

Numerator arm: diffcap043's own in-frame float64 gradient (diffusion_boundary.pt["grad_f64"]).
Denominator arm: the model-scope float64 bundle the SECTION_ATTRIBUTION ratio divides by.
Both are float64. Any disagreement above the instrument floor is FRAME, not precision.
"""
import json, torch, sys

CAP = "/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt"
MODEL_F64 = "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt"
FLOOR = 7.4177415630308e-05
PREFIX = "diffusion_module.diffusion_transformer."

cap = torch.load(CAP, map_location="cpu", weights_only=False)["grad_f64"]
mdl = torch.load(MODEL_F64, map_location="cpu", weights_only=False)
if not isinstance(mdl, dict):
    print("model bundle is", type(mdl)); sys.exit(2)
# the bundle may wrap the grads
if "grads" in mdl and isinstance(mdl["grads"], dict):
    mdl = mdl["grads"]

print("cap keys", len(cap), "sample:", list(cap)[:2])
print("mdl keys", len(mdl), "sample:", list(mdl)[:2])

def norm(t): return float(torch.linalg.vector_norm(t.double()))

rows = []
missing = []
for k in mdl:
    if not k.startswith(PREFIX):
        continue
    ck = k[len("diffusion_module."):]          # capture strips the module prefix
    if ck not in cap:
        missing.append(k); continue
    a = cap[ck].double().flatten()             # capture-frame float64
    b = mdl[k].double().flatten()              # model-frame float64
    if a.shape != b.shape:
        missing.append(k + f" shape {tuple(a.shape)} vs {tuple(b.shape)}"); continue
    d = a - b
    nb = norm(b); na = norm(a); nd = norm(d)
    dot = float(torch.dot(a, b))
    rows.append(dict(param=k, ref_norm=nb, arm_norm=na, diff_norm=nd,
                     rel_l2=(nd / nb if nb > 0 else float("nan")),
                     norm_ratio=(na / nb if nb > 0 else float("nan")),
                     cos=(dot / (na * nb) if na > 0 and nb > 0 else float("nan"))))

sq_ref = sum(r["ref_norm"] ** 2 for r in rows)
sq_dif = sum(r["diff_norm"] ** 2 for r in rows)
mw = (sq_dif ** 0.5) / (sq_ref ** 0.5)
rows.sort(key=lambda r: -r["diff_norm"])
out = dict(
    what="the DiT leg's capture-frame float64 against the model-scope float64 the 2.019x divides by",
    padded_width=384, real_tokens=56, host="tt-quietbox2 (qb2), CPU only, no card",
    capture=CAP, model_reference=MODEL_F64,
    n_tensors=len(rows), n_missing=len(missing), missing=missing[:20],
    instrument_floor_upstream_f32_vs_float64=FLOOR,
    mass_weighted_rel_l2_f64_vs_f64=mw,
    ratio_to_instrument_floor=mw / FLOOR,
    median_rel_l2=sorted(r["rel_l2"] for r in rows)[len(rows) // 2] if rows else None,
    worst_rel_l2=max((r["rel_l2"], r["param"]) for r in rows) if rows else None,
    top10_by_absolute_difference=rows[:10],
)
print(json.dumps(out, indent=1)[:4000])
json.dump(out, open("perf/of3t_ditgap/FRAME_f64_vs_f64.json", "w"), indent=1)
print("\nSUMMARY mw_rel_l2 =", mw, " floor =", FLOOR, " ratio =", mw / FLOOR)
