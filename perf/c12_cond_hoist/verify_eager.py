#!/usr/bin/env python3
"""Prove `_cond_weights()` is now paid at MODEL LOAD, not inside the first fold.

The lazy build cost 0.5687 s once per process and landed in whichever fold first took the hoisted
path, so a caller that folds exactly once was 0.31-0.36 s WORSE off with the lever on than off.
That was one of the two things the flip was gated on. This checks the fix rather than asserting it:
it records WHEN each build happened against the load/fold boundary, and it runs both arms so a
"passes because nothing built" false pass is impossible.

    base    lever off at import -> 0 builds, ever
    hoist   lever on at import  -> every token-level stack builds DURING load_model, 0 during folds

The lever is read at import here on purpose. The interleaved A/B pins nothing and flips the module
attribute per fold, which still reaches the lazy path; that path is retained and is what this
script's `base` arm leaves untouched.
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

ARM = sys.argv[1] if len(sys.argv) > 1 else "hoist"
SIZE = sys.argv[2] if len(sys.argv) > 2 else "298"
assert ARM in ("base", "hoist")
# set BEFORE tt_bio is imported: the module-level flag is bound at import
os.environ["TT_BIO_DIT_COND_HOIST"] = "1" if ARM == "hoist" else "0"

import ab_flag_levers as AB  # noqa: E402
import torch  # noqa: E402
torch.set_grad_enabled(False)
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.worker import _WorkerState, _ensure_local_artifacts  # noqa: E402
from tt_bio import esmfold2 as _E  # noqa: E402
_E.set_progress(lambda *a, **k: None)
import tt_bio as _TB  # noqa: E402
assert Path(_TB.__file__).resolve().is_relative_to(REPO), f"wrong tt_bio: {_TB.__file__}"
assert bool(T._B2_DIT_COND_HOIST) == (ARM == "hoist"), "the env var did not reach the module flag"

PHASE = ["import"]
builds = []
_cw = T.DiffusionTransformer._cond_weights


def traced(self):
    if self._cond_w is not None:              # cache hit, not a build
        return _cw(self)
    ttnn.synchronize_device(get_device())
    t = time.perf_counter()
    r = _cw(self)
    ttnn.synchronize_device(get_device())
    builds.append({"phase": PHASE[0], "atom_level": self.atom_level,
                   "s": round(time.perf_counter() - t, 4)})
    return r


T.DiffusionTransformer._cond_weights = traced

work = Path(tempfile.mkdtemp(prefix="c12-eager-"))
struct_dir = work / "out"; struct_dir.mkdir(parents=True)
msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
AB._seed_msa(AB.FIX / f"cdk2x2_{SIZE}.yaml", (AB.FIX / f"cdk2x2_{SIZE}.a3m").read_text(), msa_dir)
cfg = AB.build_cfg(msa_dir, struct_dir)
_ensure_local_artifacts(cfg)

dev = get_device()
PHASE[0] = "load"
t = time.perf_counter()
state = _WorkerState("tenstorrent")
state.load_model(cfg)
state.bind_run("c12-eager", cfg)
ttnn.synchronize_device(dev)
load_s = round(time.perf_counter() - t, 3)
built_at_load = list(builds)

folds = []
PHASE[0] = "fold"
for i in range(2):
    for p in struct_dir.glob("*"):
        p.unlink() if p.is_file() else shutil.rmtree(p)
    cfg["seed"] = 0
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    state.predict_one(AB.FIX / f"cdk2x2_{SIZE}.yaml", cfg)
    ttnn.synchronize_device(dev)
    folds.append(round(time.perf_counter() - t, 3))

built_in_fold = [b for b in builds if b["phase"] == "fold"]
res = {"arm": ARM, "size": SIZE, "flag_at_import": bool(T._B2_DIT_COND_HOIST),
       "model_load_s": load_s, "fold_s": folds,
       "builds_at_load": built_at_load, "builds_in_fold": built_in_fold,
       "n_builds_at_load": len(built_at_load), "n_builds_in_fold": len(built_in_fold)}
print(json.dumps(res, indent=1), flush=True)

if ARM == "hoist":
    assert built_at_load, "hoist arm built NOTHING at load: the eager build did not fire"
    assert all(not b["atom_level"] for b in built_at_load), \
        "an atom-level stack built conditioning blocks; the lever does not apply there"
    assert not built_in_fold, f"still building inside the fold: {built_in_fold}"
else:
    assert not builds, f"base arm built conditioning blocks it never uses: {builds}"
print(f"PASS {ARM}: {len(built_at_load)} build(s) at load "
      f"({sum(b['s'] for b in built_at_load):.4f}s), {len(built_in_fold)} in fold", flush=True)
