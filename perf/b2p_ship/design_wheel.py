"""A BindCraft 2 design from a CLEAN INSTALL of the built wheel, through the user-facing entry.

Nothing here imports from the tt-bio source tree: the interpreter is `~/b2pship_venv312`, which
holds only the wheel, its `[tenstorrent]` extra and BindCraft 2 itself. It prints where `tt_bio`
was imported from so the record cannot be mistaken for a tree run.

One trajectory, the smallest binder length BindCraft 2 draws, and the campaign's own budget at 1,
because what is being proved is that the installed path runs a design, not a rate.
"""
import json
import os
import pathlib
import sys
import time

import tt_bio
from tt_bio import bindcraft2, duotraj
from bindcraft.settings import parse_setting_overrides, read_settings
from bindcraft.preflight import cleaned_campaign_settings

BC2 = os.environ["BCX_BC2"]
PARAMS = os.environ["AF2_PARAMS"]
OUT = os.environ["DESIGN_OUT"]
BINDER = int(os.environ.get("BINDER", "60"))

pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
overrides = [f"campaign_seed=100", "max_trajectories=1", f"project_folder={OUT}",
             f"binder_lengths=[{BINDER}]", "compile_next_length=0"]
settings = cleaned_campaign_settings(
    read_settings(os.path.join(BC2, "examples", "pdl1.json"),
                  parse_setting_overrides(overrides)))

tokens = bindcraft2.design_tokens(settings)
stamp = {"tt_bio_from": tt_bio.__file__, "python": sys.version.split()[0],
         "card": os.environ.get("TT_VISIBLE_DEVICES"), "binder": BINDER,
         "design_tokens": tokens, "auto_would_choose": list(duotraj.auto_trajectories(tokens)),
         "out": OUT, "started_utc": time.strftime("%FT%TZ", time.gmtime())}
print("DESIGN_STAMP " + json.dumps(stamp), flush=True)

t0 = time.time()
n = bindcraft2.run_campaign(settings, OUT, af2_weights=PARAMS)
print("DESIGN_RESULT " + json.dumps(
    {"trajectories": n, "wall_seconds": round(time.time() - t0, 1),
     "finished_utc": time.strftime("%FT%TZ", time.gmtime())}), flush=True)

for sub in ("Trajectory", "Accepted", "MPNN"):
    d = pathlib.Path(OUT) / sub
    if d.exists():
        print(f"DESIGN_FILES {sub} " + str(len(list(d.rglob('*')))), flush=True)
