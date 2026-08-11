"""Re-derive every published N=512 headline number from the final_*.json artifacts
and diff against the values printed in the state doc. No number is taken on trust."""
import json, sys

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "final"

MODELS = ["boltz2", "esmfold2", "opendde-abag", "protenix-v2"]

# Values transcribed from the CAMPAIGN RESULTS section of state/abag-xm-deepn-n512.md
DOC = {
    "boltz2": dict(n=160, gap64=0.1308, gap256=0.1726, gap512=0.1962,
                   d2=(0.0057, 0.0232, 0.0432), sd=(-0.0479, -0.0188, 0.0131),
                   orc=[(0.0090, 0.0144, 0.0218), (0.0135, 0.0228, 0.0346), (0.0114, 0.0202, 0.0326)],
                   floor=0.0135, marg=(0.008001, 0.003559), cardh=1.577, basis=158,
                   reach=(96, 75, 29)),
    "esmfold2": dict(n=160, gap64=0.0944, gap256=0.1288, gap512=0.1567,
                     d2=(0.0157, 0.0277, 0.0405), sd=(-0.0273, -0.0066, 0.0138),
                     orc=[(0.0183, 0.0276, 0.0406), (0.0152, 0.0251, 0.0376), (0.0238, 0.0363, 0.0525)],
                     floor=0.0156, marg=(0.011875, 0.008594), cardh=1.173, basis=160,
                     reach=(100, 65, 36)),
    "opendde-abag": dict(n=152, gap64=0.0827, gap256=0.1124, gap512=0.1223,
                         d2=(0.0048, 0.0098, 0.0157), sd=(-0.0333, -0.0195, -0.0077),
                         orc=[(0.0080, 0.0155, 0.0266), (0.0070, 0.0109, 0.0158), (0.0068, 0.0105, 0.0156)],
                         floor=0.0062, marg=(0.002605, 0.001256), cardh=2.316, basis=152,
                         reach=(128, 110, 74)),
    "protenix-v2": dict(n=158, gap64=0.1519, gap256=0.2098, gap512=0.2456,
                        d2=(0.0250, 0.0355, 0.0478), sd=(-0.0461, -0.0221, 0.0007),
                        orc=[(0.0172, 0.0266, 0.0385), (0.0250, 0.0382, 0.0542), (0.0222, 0.0321, 0.0443)],
                        floor=0.0172, marg=(0.008724, 0.003676), cardh=2.429, basis=158,
                        reach=(131, 106, 41)),
}

fails, checks = [], 0

def cmp(model, label, got, want, tol):
    global checks
    checks += 1
    if got is None:
        fails.append(f"{model} {label}: MISSING in artifact")
        return
    if abs(got - want) > tol:
        fails.append(f"{model} {label}: artifact {got!r} vs doc {want!r} (tol {tol})")

for m in MODELS:
    doc = DOC[m]
    p3 = json.load(open(f"{PREFIX}_{m}_panel3_512.json"))
    blk = p3["models"][m]
    deep = json.load(open(f"{PREFIX}_{m}.json"))[f"{m}__deep"]
    pw = json.load(open(f"{PREFIX}_{m}.json"))[f"{m}__pairwise_gain_ci"]

    # panel size
    checks += 1
    if len(p3["panel"]) != doc["n"]:
        fails.append(f"{m} panel size: artifact {len(p3['panel'])} vs doc {doc['n']}")
    # rung identity of the 3-rung panel
    checks += 1
    if (p3["lo"], p3["mid"], p3["hi"]) != (64, 256, 512):
        fails.append(f"{m} panel3 rungs are {(p3['lo'],p3['mid'],p3['hi'])}, expected (64,256,512)")

    for r in ("64", "256", "512"):
        cmp(m, f"gap@{r}", blk["gap"][r], doc[f"gap{r}"], 5e-5)
    for i, k in enumerate(("lo", "med", "hi")):
        cmp(m, f"d2_ci.{k}", blk["d2_ci"][i], doc["d2"][i], 5e-5)
        cmp(m, f"second_diff_ci.{k}", blk["second_diff_ci"][i], doc["sd"][i], 5e-5)

    # oracle gain ladder
    for j, step in enumerate(("64->128", "128->256", "256->512")):
        got = pw[step]["gain_ci"]["oracle"]
        for i, k in enumerate(("lo", "med", "hi")):
            cmp(m, f"oracle {step}.{k}", got[i], doc["orc"][j][i], 5e-5)

    # seed-noise floor at 256 must be the FIXED-panel floor, not the free-panel one
    fixed = deep["within_fold_common"]["floor_med"].get("256")
    free = deep["seed_noise_floor_med"].get("256")
    cmp(m, "floor[256] (fixed panel)", fixed, doc["floor"], 5e-5)
    checks += 1
    if free is not None and abs(free - doc["floor"]) <= 5e-5 and abs(free - fixed) > 5e-5:
        fails.append(f"{m} floor[256]: doc matches FREE-panel {free} not fixed-panel {fixed}")

    # marginal oracle per 1000 card-seconds, cost basis
    cmp(m, "marg 128->256", pw["128->256"].get("marginal_oracle_per_1000cs"), doc["marg"][0], 5e-7)
    cmp(m, "marg 256->512", pw["256->512"].get("marginal_oracle_per_1000cs"), doc["marg"][1], 5e-7)
    cmp(m, "cost_h_per_target 256->512", pw["256->512"].get("cost_h_per_target"), doc["cardh"], 5e-4)
    checks += 1
    if pw["256->512"].get("cost_basis_targets") != doc["basis"]:
        fails.append(f"{m} cost_basis_targets: artifact {pw['256->512'].get('cost_basis_targets')} vs doc {doc['basis']}")

    # interface reach counts
    for thr, want in zip(("0.23", "0.49", "0.8"), doc["reach"]):
        checks += 1
        got = deep["solvable_at_top"].get(thr)
        if got != want:
            fails.append(f"{m} reach>={thr}: artifact {got} vs doc {want}")

    # stop-rule verdict: does the point estimate clear the fixed-panel floor?
    o = pw["256->512"]["gain_ci"]["oracle"]
    print(f"{m:14s} panel {len(p3['panel']):3d}  oracle256->512 {o[1]:+.4f} "
          f"[{o[0]:+.4f},{o[2]:+.4f}]  floor {fixed:.4f}  ratio {o[1]/fixed:.2f}x  "
          f"point{'ABOVE' if o[1] > fixed else 'AT/BELOW'}  "
          f"CIlo{'above' if o[0] > fixed else 'BELOW'}")

# four-model panel
fm = json.load(open(f"{PREFIX}_fourmodel_panel3_512.json"))
checks += 1
if len(fm["panel"]) != 150:
    fails.append(f"four-model panel: {len(fm['panel'])} vs doc 150")
print(f"\nfour-model panel: {len(fm['panel'])} targets, models {sorted(fm['models'])}")
for m in MODELS:
    b = fm["models"][m]
    print(f"  {m:14s} gap 64/256/512 {b['gap']['64']:.4f} {b['gap']['256']:.4f} {b['gap']['512']:.4f}"
          f"  sd [{b['second_diff_ci'][0]:+.4f},{b['second_diff_ci'][1]:+.4f},{b['second_diff_ci'][2]:+.4f}]"
          f"  {b['verdict_level']}/{b['verdict_rate']}")

print(f"\n{checks} checks run")
if fails:
    print(f"MISMATCHES: {len(fails)}")
    for f in fails:
        print("  FAIL", f)
else:
    print("ALL PUBLISHED NUMBERS REPRODUCE FROM ARTIFACTS")
