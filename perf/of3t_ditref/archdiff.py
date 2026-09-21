#!/usr/bin/env python3
"""Which modules does the 0.4.3 -> 0.5.0 boundary actually move?

D58 pairs the diffusion module's 19.6x with `msa_module`'s 19.8x. The repaired denominator is a
diffusion-scope repair, and "the trunk is out of reach of it" is exactly the kind of claim this
campaign has had to withdraw before, so it is measured instead of argued: build the WHOLE model
under each tree and diff the parameter names by top-level module.

AMENDMENT 2, and read this before quoting the output: a `named_parameters()` diff is blind
to every change that adds no parameter, and the campaign holds the counter-example on a
module this script calls identical. `pairformer_stack` is 2736 on both trees with identical
names, and `of3t-trunk043ref` measured 5.647x between the two references on it, because
`transpose_bias` is a 0.5.0 convention rather than a 0.5.0 parameter. So "module X is
identical" here means its parameter NAMES are, which is a weaker sentence than "the boundary
does not reach module X". The functional boundary is enumerated at source in
`perf/of3t_orchestrator/PARAMETER_NAME_IDENTITY_IS_NOT_FUNCTION_IDENTITY.json`; use a
version-only arm, not this diff, before saying a module is out of reach.

One tree per process -- the two packages cannot both be `openfold3` at once.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import refpath  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

refpath.install(a.tree)
tree = refpath.assert_resolved(a.tree)
print(f"REF_TREE resolved: {tree}", flush=True)

from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry  # noqa: E402
from openfold3.projects.of3_all_atom.model import OpenFold3  # noqa: E402

cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
cfg.architecture.shared.use_confidence_emb_prob = 0.8
cfg.architecture.shared.diffusion.use_conditioning_prob = 0.8
m = OpenFold3(cfg)
names = sorted(n for n, _ in m.named_parameters())
by_mod = {}
for n in names:
    by_mod.setdefault(n.split(".")[0], []).append(n)
json.dump({"REF_TREE_resolved": tree, "n_parameters": len(names),
           "by_module": {k: len(v) for k, v in sorted(by_mod.items())},
           "names": names},
          open(a.out, "w"), indent=1, sort_keys=True)
print(f"{len(names)} parameters, {len(by_mod)} top-level modules -> {a.out}", flush=True)
for k, v in sorted(by_mod.items()):
    print(f"  {k:34s} {len(v)}")
