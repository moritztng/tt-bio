#!/usr/bin/env python3
"""What is on main that no user can reach?

`ExtraMsaOnDevice` sat on main for days as a class with no caller: the port was complete, the
measured lever was the campaign's second largest, and it was worth a verified 0.0000 s to anyone
using the library. That is this row's charge stated as a defect -- a win users never receive did
not happen -- and it had never been swept for.

For every top-level class and public function defined under `tt_bio/` (skipping `_vendor/`), ask
where else its name appears:

    engine      another tt_bio module references it
    harness     only tests/ perf/ scripts/ do
    unreferenced nobody does

`harness` and `unreferenced` are where a shipped-zero hides. It is a NAME grep, so a symbol
reached by `getattr`, a registry string or a re-export will read as unreachable when it is not;
every hit needs its own look. That is why this prints the evidence rather than a verdict.
"""
from __future__ import annotations

import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
DEF = re.compile(r"^(?:class|def)\s+([A-Za-z]\w*)")

engine_files, harness_files = [], []
for p in sorted((ROOT / "tt_bio").rglob("*.py")):
    if "_vendor" in p.parts:
        continue
    engine_files.append(p)
for d in ("tests", "perf", "scripts"):
    harness_files += sorted((ROOT / d).rglob("*.py"))

# A DECORATED definition is registered, not referenced: `@click.command` puts `design_cmd` on
# the CLI without anyone naming it again, and the first run of this sweep flagged three of the
# most user-facing functions in the repo because of it. Skip anything with a decorator directly
# above it.
defs = defaultdict(list)
for p in engine_files:
    lines = p.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines, 1):
        m = DEF.match(line)
        if not m or m.group(1).startswith("_"):
            continue
        above = next((l for l in reversed(lines[:i - 1]) if l.strip()), "")
        # A multi-line decorator stack ends on its own closing paren, so the line above a
        # registered `def` is often `)` or `... help="...")` rather than `@...`. Missing that
        # is how `preflight_cmd` -- an actual `@cli.command` -- survived the first exclusion.
        if above.lstrip().startswith("@") or above.rstrip().endswith((",", "(", ")")):
            continue                  # registered by a decorator, or inside its argument list
        defs[m.group(1)].append((p, i))

# Tokenise each file ONCE. Searching every name against every file is O(names x files) and on
# this repo that is minutes; a token set per file answers the same question in seconds.
TOKEN = re.compile(r"[A-Za-z_]\w*")
tokens = {p: TOKEN.findall(p.read_text(errors="replace"))
          for p in engine_files + harness_files}
tokset = {p: set(t) for p, t in tokens.items()}
rows = []
for name, sites in defs.items():
    if len(sites) != 1:
        continue                      # defined twice: an override or a shim, not this question
    home, line = sites[0]
    in_engine = sum(1 for p in engine_files if p != home and name in tokset[p])
    in_harness = sum(1 for p in harness_files if name in tokset[p])
    own = tokens[home].count(name) - 1
    if in_engine:
        continue
    rows.append({"name": name, "home": str(home.relative_to(ROOT)), "line": line,
                 "own": own, "harness": in_harness,
                 "kind": "unreferenced" if not in_harness and own <= 0 else
                         ("harness-only" if in_harness else "own-file-only")})

rows.sort(key=lambda r: (r["kind"], -r["harness"], r["name"]))
for kind in ("unreferenced", "harness-only", "own-file-only"):
    group = [r for r in rows if r["kind"] == kind]
    print(f"\n=== {kind}: {len(group)} ===")
    for r in group[:18]:
        print(f"  {r['name']:34s} {r['home']}:{r['line']:<6} own_uses={r['own']:<3} "
              f"harness_files={r['harness']}")
print(f"\ntotal public top-level defs with no other tt_bio module referencing them: {len(rows)}")
