"""Write the golden INDEX (markdown) from the MANIFEST.json files a capture left under GOLDEN_ROOT.

    python3 make_index.py GOLDEN_ROOT POINTER > INDEX.md

POINTER is how a reader reaches GOLDEN_ROOT (e.g. ttuser@tt-quietbox2:~/mol1-goldens/n4). Needs only the stdlib.
"""
import json
import re
import sys
from pathlib import Path

root, pointer = Path(sys.argv[1]), sys.argv[2]
print(f"# Mol-1 upstream goldens\n\nRoot: `{pointer}` (`<group>/goldens/<case>/seed_<seed>/`; predictions in "
      "`<group>/pred/`). Format: perf/mol1_ref/capture.py docstring on wk/mol1-ref.\n")
for run_log in sorted(root.glob("*/run.log")):
    grp = run_log.parent
    log = run_log.read_text(errors="replace")
    cfg = re.search(r"cycle=(\d+), step=(\d+)", log)
    print(f"## {grp.name}\n\nupstream log: cycles {cfg.group(1) if cfg else '?'}, steps {cfg.group(2) if cfg else '?'}\n")
    for man in sorted(grp.glob("goldens/*/seed_*/MANIFEST.json")):
        d = man.parent
        case, seed = d.parent.name, d.name.split("_", 1)[1]
        tok = re.search(rf"{re.escape(case)} \[seed:{seed}\]: N_asym (\d+), N_token (\d+), N_atom (\d+)", log)
        ok = re.search(rf"{re.escape(case)} \[seed:{seed}\] succeeded. Model forward time: ([\d.]+)s", log)
        cifs = sorted(p.name for p in (grp / "pred" / case / f"seed_{seed}" / "predictions").glob("*.cif"))
        print(f"### {case} seed {seed}\n")
        print(f"N_asym {tok.group(1)}, N_token {tok.group(2)}, N_atom {tok.group(3)}; " if tok else "", end="")
        print(f"forward {ok.group(1)} s; " if ok else "forward FAILED or missing; ", end="")
        print(f"{len(cifs)} CIF sample(s)\n")
        print("| file | bytes | sha256 | contents |\n|---|---|---|---|")
        for fn, m in sorted(json.loads(man.read_text()).items()):
            s = json.dumps(m["summary"])
            print(f"| {fn} | {m['bytes']} | `{m['sha256'][:16]}` | {s[:160] + ('...' if len(s) > 160 else '')} |")
        print()
