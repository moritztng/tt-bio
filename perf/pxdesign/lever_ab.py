"""One arm of a PXDesign lever A/B: the page cell's designs at a lever set, timed, with the designs kept.

    python perf/pxdesign/lever_ab.py OUT --levers normal [--bf16] [--trace] [--seeds 8] [--fixture laczc_512_nomsa]

Same work as perf/bmk/design_cell.py's pxdesign cell (featurise + design at 400 steps + write, one backbone per
call, checkpoint load outside), on seeds 0..N-1 after one dropped cold design at seed 0. Writes OUT/arm.json (per
design wall, fit residual) and OUT/designs.pt (each seed's binder in the input frame, the target's distogram atoms,
binder CA mask), which perf/pxdesign/lever_grade.py pairs by seed across arms.
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--levers", default="normal", help="parse_levers spec; `none` is the bare module")
ap.add_argument("--bf16", action="store_true", help="bf16 diffusion (PROTENIX_DIFFUSION_FP32_DEVICE=0)")
ap.add_argument("--trace", action="store_true", help="replay a captured trace of each denoise step")
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--fixture", default="laczc_512_nomsa")
a = ap.parse_args()
if a.bf16:
    os.environ["PROTENIX_DIFFUSION_FP32_DEVICE"] = "0"
os.environ["TT_BIO_LEVERS"] = a.levers   # model_levers reads it; the model takes no levers= from a checkpoint load
TREE = Path.cwd()
sys.path.insert(0, str(TREE))

import torch, yaml  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor, ensure_pxdesign_weights  # noqa: E402
from tt_bio.pxdesign.inputs import design_inputs_from_yaml  # noqa: E402
from tt_bio.pxdesign.model import ProtenixDesign  # noqa: E402
from tt_bio.pxdesign.write import BINDER_RESTYPE, rigid_transform, write_design_cifs  # noqa: E402

torch.set_grad_enabled(False)
a.out.mkdir(parents=True, exist_ok=True)
spec = yaml.safe_load((TREE / f"perf/pxdesign/targets/{a.fixture}.yaml").read_text())
spec["target"]["file"] = str(TREE / "perf/pxdesign/targets" / Path(spec["target"]["file"]).name)
y = a.out / f"{a.fixture}.yaml"
y.write_text(yaml.safe_dump(spec))
ensure_p300_mesh_descriptor()
T.get_device(trace="protenix" if a.trace else None)
t = time.perf_counter()
model = ProtenixDesign.load_from_checkpoint(str(ensure_pxdesign_weights(Path.home() / ".boltz")))
res = {"levers": sorted(model._levers), "bf16": a.bf16, "trace": a.trace, "fixture": a.fixture, "arch": str(model.dev.arch()),
       "model_init_s": round(time.perf_counter() - t, 3), "designs": []}
keep = {}
for i, seed in enumerate([0] + list(range(a.seeds))):
    t0 = time.perf_counter()
    feats = design_inputs_from_yaml(y)
    feats = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v) for k, v in feats.items()}
    coords = model.design(feats, n_step=400, n_sample=1, seed=seed, trace=a.trace)
    rows = write_design_cifs(coords, feats, a.out / f"r{i}", stem=a.fixture)
    wall = time.perf_counter() - t0
    d = {"rep": i, "seed": seed, "cold": i == 0, "wall_s": round(wall, 3), "fit_rmsd": rows[0]["fit_rmsd"],
         "finite": bool(torch.isfinite(coords).all())}
    res["designs"].append(d)
    print(json.dumps(d), flush=True)
    if i == 0:
        continue
    cond = feats["condition"]
    conditioned = torch.tensor([n != "xpb" for n in cond["res_name"]]) & cond["is_resolved"].bool()
    disto = feats["distogram_rep_atom_mask"].bool()
    a2t = feats["atom_to_token_idx"].long()
    at_binder = (feats["restype"].argmax(-1) == BINDER_RESTYPE)[a2t]
    ref = cond["coord"][conditioned].double()
    rep = coords[0][disto].double()
    r, ca, cb = rigid_transform(rep[conditioned], ref)
    chars = feats["ref_atom_name_chars"][at_binder]
    is_ca = torch.tensor(["".join(chr(int(c.argmax()) + 32) for c in row).strip() == "CA" for row in chars])
    keep[seed] = {"binder": (coords[0][at_binder].double() - ca) @ r + cb, "binder_ca": is_ca, "target": ref}
torch.save(keep, a.out / "designs.pt")
warm = sorted(d["wall_s"] for d in res["designs"] if not d["cold"])
res["warm_median_s"] = warm[len(warm) // 2] if len(warm) % 2 else (warm[len(warm) // 2 - 1] + warm[len(warm) // 2]) / 2
(a.out / "arm.json").write_text(json.dumps(res, indent=1))
print("WARM_MEDIAN", res["warm_median_s"], "levers", ",".join(res["levers"]) or "none", "bf16", a.bf16, "trace", a.trace)
T.cleanup()
