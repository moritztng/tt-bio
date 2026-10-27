"""Where a PXDesign sampler step goes: atom encoder, token DiT, atom decoder, rest (host + glue), synchronised.

    python perf/pxdesign/stage_split.py OUT.json [--levers normal] [--bf16] [--fixture laczc_512_nomsa]

One cold design (dropped), one warm design unsynchronised (the wall), one warm design with ttnn.synchronize_device
around the three stages of every step (adds a sync per stage, so its total reads a little over the wall). Seed 0.
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--levers", default="normal")
ap.add_argument("--bf16", action="store_true")
ap.add_argument("--fixture", default="laczc_512_nomsa")
a = ap.parse_args()
if a.bf16:
    os.environ["PROTENIX_DIFFUSION_FP32_DEVICE"] = "0"
TREE = Path.cwd()
sys.path.insert(0, str(TREE))

import torch, ttnn, yaml  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor, ensure_pxdesign_weights  # noqa: E402
from tt_bio.pxdesign.inputs import design_inputs_from_yaml  # noqa: E402
from tt_bio.pxdesign.model import ProtenixDesign  # noqa: E402

torch.set_grad_enabled(False)
spec = yaml.safe_load((TREE / f"perf/pxdesign/targets/{a.fixture}.yaml").read_text())
spec["target"]["file"] = str(TREE / "perf/pxdesign/targets" / Path(spec["target"]["file"]).name)
y = a.out.with_suffix(".yaml")
a.out.parent.mkdir(parents=True, exist_ok=True)
y.write_text(yaml.safe_dump(spec))
ensure_p300_mesh_descriptor()
model = ProtenixDesign.load_from_checkpoint(str(ensure_pxdesign_weights(Path.home() / ".boltz")))
D = model.diffusion
acc = {}
on = [False]


def timed(name, fn):
    def run(*x, **k):
        if not on[0]:
            return fn(*x, **k)
        ttnn.synchronize_device(model.dev)
        t = time.perf_counter()
        r = fn(*x, **k)
        ttnn.synchronize_device(model.dev)
        acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
        return r
    return run


class Timed:
    """The module, with its call timed; every other attribute (bias caches, weights) reads through."""
    def __init__(self, name, m):
        self._m, self._call = m, timed(name, m)

    def __call__(self, *x, **k):
        return self._call(*x, **k)

    def __getattr__(self, k):
        return getattr(self._m, k)


D.atxE = Timed("atom_encoder", D.atxE)
D.atxD = Timed("atom_decoder", D.atxD)
D._token_dit_device = timed("token_dit", D._token_dit_device)
D._denoise_device = timed("denoise_total", D._denoise_device)


def design():
    feats = design_inputs_from_yaml(y)
    feats = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v) for k, v in feats.items()}
    t = time.perf_counter()
    model.design(feats, n_step=400, n_sample=1, seed=0)
    return time.perf_counter() - t, int(feats["restype"].shape[0]), int(feats["ref_pos"].shape[0])


cold, nt, na = design()
wall, _, _ = design()
on[0] = True
synced, _, _ = design()
res = {"levers": sorted(model._levers), "bf16": a.bf16, "arch": str(model.dev.arch()), "tokens": nt, "atoms": na,
       "cold_s": round(cold, 3), "warm_wall_s": round(wall, 3), "synced_wall_s": round(synced, 3),
       "stages_s": {k: round(v, 3) for k, v in acc.items()}}
res["stages_s"]["outside_denoise"] = round(synced - acc.get("denoise_total", 0.0), 3)
a.out.write_text(json.dumps(res, indent=1))
print(json.dumps(res))
T.cleanup()
