"""Extend CROPS.json with the diffusion capture, rather than starting a second table (D193)."""
import json, torch, pathlib
P = pathlib.Path("perf/of3t_orchestrator/crops/CROPS.json")
d = json.loads(P.read_text())
f = "/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt"
b = torch.load(f, map_location="cpu", weights_only=False)
kw = b["kwargs"]
e = dict(token_mask=list(kw["token_mask"].shape), atom_mask=list(kw["atom_mask"].shape),
         zij_trunk=list(kw["zij_trunk"].shape),
         padded_width=int(kw["token_mask"].shape[-1]),
         real_tokens=int(kw["token_mask"].sum()),
         no_samples=int(b["no_samples"]), loss=float(b["loss"]),
         mask_key_read="token_mask (this capture has no `single_mask`; the diffusion module takes "
                       "`token_mask`, so the rule's key is module-dependent and the key read is "
                       "recorded beside the number)",
         used_by="the `diffusion=` leg of every model-scope arm in "
                 "perf/of3t_modelboundary/MODEL_withtrunk_n384.json, so every diffusion_module.* "
                 "section figure this campaign publishes -- 43.89 % of the model's squared "
                 "gradient norm -- is a padded-width-384 reading",
         measured_on="tt-quietbox2 (qb2), of3t-ditgap, torch load + shape read, no device")
if f in d["captures"] and d["captures"][f] == e:
    print("already present and identical"); raise SystemExit(0)
d["captures"][f] = e
d["consequences"].append(
    "of3t-ditgap added the DIFFUSION capture `diffcap043/diffusion_boundary.pt`: padded width "
    "384, 56 real tokens, 48 diffusion samples. It is the same width and the same real-token "
    "count as the three CAP block boundaries, so the campaign's diffusion-side and trunk-side "
    "figures are at one width. Its mask key is `token_mask`, not `single_mask` -- the rule's key "
    "is module-dependent, and a row that reads `single_mask` on this file gets a KeyError rather "
    "than a wrong number, which is the safe direction.")
d["not_covered"] = ("the n64 and n384 boundaries built by of3t-frame384 and of3t-trunk043ref are "
                    "not in this table; a row that adds one should extend it rather than start a "
                    "second table.")
P.write_text(json.dumps(d, indent=2) + "\n")
print(json.dumps(e, indent=1))
