"""Where does the captured cotangent's mass sit, real positions or padded ones?

Block 47's backward hands back a pair-track cotangent 0.588x the reference with cosine 0.995,
a near-scalar shrink, while blocks 0 and 23 hand back 1.000. (56/64)^4 = 0.5862 is within
0.3 % of that, so the candidate is a pair mask applied once more on our side than on theirs.
The candidate only survives if the captured cotangent actually carries mass on the padded
positions at block 47 and not at 0 and 23; otherwise the match is a coincidence and it is
said so.
"""
import json, torch
out = {"what": __doc__.strip().splitlines()[0], "crop": 64, "blocks": {}}
for k in (0, 23, 47):
    cap = torch.load(f"/home/ttuser/of3t_gradients/cap/block{k}_boundary.pt",
                     map_location="cpu", weights_only=False)
    sm = cap["kwargs"]["single_mask"][:, :64].double()
    cs = cap["cot"][0][:, :64].double()
    cz = cap["cot"][1][:, :64, :64].double()
    s_in = cap["args"][0][:, :64].double()
    z_in = cap["args"][1][:, :64, :64].double()
    m1 = sm.reshape(1, 64, 1)
    m2 = (sm.reshape(1, 64, 1) * sm.reshape(1, 1, 64)).reshape(1, 64, 64, 1)
    f = lambda t, m: float((t * m).pow(2).sum()) / (float(t.pow(2).sum()) or 1.0)
    out["blocks"][k] = {
        "real_tokens": int(sm.sum()),
        "cot_s_norm": float(cs.norm()), "cot_z_norm": float(cz.norm()),
        "cot_s_mass_on_real": f(cs, m1), "cot_z_mass_on_real": f(cz, m2),
        "s_in_mass_on_real": f(s_in, m1), "z_in_mass_on_real": f(z_in, m2),
        "s_in_norm": float(s_in.norm()), "z_in_norm": float(z_in.norm())}
    b = out["blocks"][k]
    print(f"block {k:2d}: cot_s {b['cot_s_mass_on_real']*100:8.4f} % of mass on real rows, "
          f"cot_z {b['cot_z_mass_on_real']*100:8.4f} % | s_in {b['s_in_mass_on_real']*100:8.4f} % "
          f"z_in {b['z_in_mass_on_real']*100:8.4f} % | |cot_s| {b['cot_s_norm']:.4e} "
          f"|cot_z| {b['cot_z_norm']:.4e}")
print(f"\n(56/64)^2 = {(56/64)**2:.6f}   (56/64)^4 = {(56/64)**4:.6f}   "
      f"measured block-47 dz_in norm ratio 0.588304, weight-gradient norm ratio 0.591026")
open("/home/ttuser/.coworker/wt/of3t-trunkback/perf/of3t_trunkback/PAD_MASS.json","w").write(
    json.dumps(out, indent=1) + "\n")
