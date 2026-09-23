#!/usr/bin/env python3
"""Write plan.json: the models the references cover and the settings each is folded at.

The model list is PREDICT_MODELS and the recycles and sampling steps are tt-bio's own shipped
defaults for each model, read from tt_bio.main rather than retyped, so a model added there shows
up here and the GPU folds at the settings the TT fold uses. Needs an environment that can import
tt_bio.main (it opens no device); the GPU box reads the json because it cannot.

    TT_VISIBLE_DEVICES= ~/tt-bio/env/bin/python perf/mgx/ref/make_plan.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from tt_bio.main import (PREDICT_MODELS, _resolve_recycling_steps,  # noqa: E402
                         _resolve_sampling_steps)

plan = {"source": "tt_bio.main PREDICT_MODELS, _resolve_recycling_steps, _resolve_sampling_steps",
        "seeds": [0, 1],
        "models": {m: {"recycles": _resolve_recycling_steps(None, m),
                       "steps": _resolve_sampling_steps(None, m)} for m in PREDICT_MODELS}}
(HERE / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
print(json.dumps(plan["models"]))
