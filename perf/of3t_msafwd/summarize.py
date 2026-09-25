#!/usr/bin/env python3
"""BISECT tables for of3t-msafwd from the UP_*/OURS_* reports and the float64 trace.

Per site, teacher-forced: ours, upstream bf16, upstream fp32 and the site's bf16 storage floor
(its float64 update rounded to bf16), as local rel_l2 and as error mass over |z_out|. Then the
quadrature closure per side, then the chain arms."""
import json
import sys
from pathlib import Path

import torch

P = Path(__file__).resolve().parent
T = torch.load(sys.argv[1] if len(sys.argv) > 1 else "/home/ttuser/of3t_msafwd/trace_f64_crop64.pt",
               map_location="cpu", weights_only=False)
B = torch.load(T["boundary"], map_location="cpu", weights_only=False)
n = int(T["z_out"].shape[-2])
tokm = torch.diagonal(B["inputs"]["kwargs"]["pair_mask"].reshape(n, n)) > 0
rz = lambda x: x.reshape(n, n, -1)[tokm][:, tokm]
rm = lambda x: x.reshape(x.shape[-3], n, -1)[:, tokm]
zo = float(rz(T["z_out"]).norm())
L = lambda f: json.loads((P / f).read_text())["sites"]
ours, ub, uf = L("OURS_tf.json"), L("UP_bf16_tf.json"), L("UP_f32_tf.json")

fam, out = {}, {"sites": {}}
print(f"{'site':22s} {'ours rel':>10s} {'up bf16':>10s} {'up fp32':>10s} {'bf16 flr':>10s} "
      f"{'ours/up':>8s} | {'ours inj':>10s} {'up inj':>10s}")
for k, t in T["sites"].items():
    trk = ours[k]["track"]
    R = rm if trk == "m" else rz
    u = R(t["upd"])
    flr = float((u.to(torch.bfloat16).double() - u).norm() / u.norm())
    o, b, f = ours[k]["upd_rel"], ub[k]["upd_rel"], uf[k]["upd_rel"]
    oi, bi = ours[k]["upd_err_over_zout"], ub[k]["upd_err_over_zout"]
    print(f"{k:22s} {o:10.3e} {b:10.3e} {f:10.3e} {flr:10.3e} {o/b:8.2f} | "
          + (f"{oi:10.3e} {bi:10.3e}" if oi is not None else f"{'(m track)':>21s}"))
    out["sites"][k] = {"ours": o, "up_bf16": b, "up_fp32": f, "bf16_floor": flr,
                       "ours_inj": oi, "up_inj": bi}
    s = k.split(".")[1]
    F = fam.setdefault(s, {"o2": 0.0, "b2": 0.0, "or": [], "br": []})
    F["or"].append(o); F["br"].append(b)
    if oi is not None:
        F["o2"] += oi ** 2; F["b2"] += bi ** 2

print("\nper family (z-track injected error in quadrature over blocks, /|z_out|; m-track: mean local rel)")
tot_o = tot_b = 0.0
for s, F in fam.items():
    if F["o2"]:
        tot_o += F["o2"]; tot_b += F["b2"]
        print(f"  {s:16s} ours {F['o2']**.5:.3e}  up bf16 {F['b2']**.5:.3e}  ratio {(F['o2']/F['b2'])**.5:5.2f}"
              f"  excess^2 {(F['o2']-F['b2'])*1e6:+.3f}e-06")
    else:
        mo, mb = sum(F["or"]) / len(F["or"]), sum(F["br"]) / len(F["br"])
        print(f"  {s:16s} local rel ours {mo:.3e}  up bf16 {mb:.3e}  ratio {mo/mb:5.2f}")
    out.setdefault("families", {})[s] = F
print(f"  op-error sum in quadrature: ours {tot_o**.5:.3e}  up bf16 {tot_b**.5:.3e}")

# Residual storage injected by bf16 RNE rounding of every state that is written, from the trace.
res2 = 0.0
keys = list(T["sites"])
for i, k in enumerate(keys):
    if T["sites"][k] is None or k.split(".")[1] in ("pwa", "msa_transition"):
        continue
    after = T["sites"][keys[i + 1]]["z"] if i + 1 < len(keys) else T["z_out"]
    a = rz(after)
    res2 += float((a.to(torch.bfloat16).double() - a).norm()) ** 2 / zo ** 2
zin = rz(T["sites"][keys[0]]["z"])
inp2 = float((zin.to(torch.bfloat16).double() - zin).norm()) ** 2 / zo ** 2
print(f"  bf16 RNE residual storage, 24 z writes: {res2**.5:.3e}; bf16 z input: {inp2**.5:.3e}")
print(f"  closure: ours predicted {(tot_o + res2 + inp2)**.5:.3e}; up bf16 predicted {tot_b**.5:.3e}")
out["closure"] = {"ours_op_quad": tot_o ** .5, "up_op_quad": tot_b ** .5,
                  "rne_residual_quad": res2 ** .5, "input_rounding": inp2 ** .5}

print("\nchain arms (z_out real block vs float64)")
for f in sorted(P.glob("OURS_*.json")) + sorted(P.glob("UP_*.json")):
    j = json.loads(f.read_text())
    if "z_out_vs_capture_rel" in j and j.get("n_tokens") == n:
        print(f"  {f.stem:22s} {j['z_out_vs_capture_rel']:.6e}")
        out.setdefault("chain", {})[f.stem] = j["z_out_vs_capture_rel"]
(P / "BISECT.json").write_text(json.dumps(out, indent=1) + "\n")
