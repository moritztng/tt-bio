"""What the reader and the capability check say about each (model, input), without a device.

This is the claim the device run is checked against: the same two functions every predict path
calls before it opens a chip. The model list comes from tt_bio.main, never retyped.
"""
import json
import sys
from pathlib import Path

import click

from tt_bio.capabilities import check_capabilities
from tt_bio.main import PREDICT_MODELS, _read_bio_chains

inputs = sorted(Path(sys.argv[1]).glob("*.*"))
out = {}
for model in PREDICT_MODELS:
    for p in inputs:
        notes = []
        try:
            chains = _read_bio_chains(p)
            found = check_capabilities(p, chains, model, echo=notes.append)
            v = {"verdict": "noted" if notes else "accepted", "found": found, "notes": notes,
                 "chains": [(c[0], c[3], len(c[1])) for c in chains]}
        except (RuntimeError, click.ClickException, ValueError, KeyError) as e:
            msg = e.format_message() if isinstance(e, click.ClickException) else str(e)
            v = {"verdict": "refused", "error": type(e).__name__, "message": msg}
        out.setdefault(model, {})[p.name] = v
json.dump(out, sys.stdout, indent=1)
