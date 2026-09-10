#!/usr/bin/env python3
"""`<verdict> <wall_s> <class>` for the newest row in one ladder's evidence file, or `NONE 0 -`.

`ladder.sh` reads this to decide whether the card is still worth handing the next rung, because a
chain that does not look back turns one freeze into a queue of false ceilings (twice on
2026-09-10: ten rungs, then three).

`class` is read from that rung's OWN log, not guessed from its wall clock:

  DEVOPEN  the card refused to come up -- `risc_firmware_initializer` / "failed to initialize FW"
           (a chip left dirty by a kill or a freeze), or DeviceInUseError / exit 75 (a co-tenant).
           Nothing about the model was measured and nothing after it will be either.
  RAN      the rung got a device and its outcome is about the model.

A wall-clock threshold would be the wrong instrument: 16 s is proof of a broken card at 1024
tokens and a plausible small fold at 128.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_paths  # noqa: E402

DEVOPEN = ("risc_firmware_initializer", "failed to initialize FW", "DeviceInUseError",
           "is in use by pid", "Refusing to open it concurrently")

tag = ladder_paths.tag_from_env(sys.argv[1] if len(sys.argv) > 1 else "")
jl = ladder_paths.results_path(tag)
rows = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()] if jl.is_file() else []
if not rows:
    print("NONE 0 -")
    raise SystemExit(0)
r = rows[-1]
label = f"{r['model']}_{r['size']}" + (f"_{r['tag']}" if r.get("tag") else "")
log = ladder_paths.runs_dir(tag) / label / "fold.log"
text = (log.read_text(errors="replace") if log.is_file() else "") + (r.get("tail") or "")
klass = "DEVOPEN" if any(p in text for p in DEVOPEN) else "RAN"
print(f"{r['verdict']} {int(r.get('wall_s') or 0)} {klass}")
