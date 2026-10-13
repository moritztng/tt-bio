#!/usr/bin/env python3
"""Run release_gate.py's rfd3 or rfd3-fusion leg in process at chosen seeds, one JSON row each.

    gap_gate_seeds.py OUT.jsonl rfd3 42,43,44,45
    gap_gate_seeds.py OUT.jsonl rfd3-fusion

The rfd3 leg's floors were set on one seed (42); this measures how its worst-of-4 counts spread
over seeds for the tree and environment it runs in, scored by the gate's own code.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_gate as R                                                  # noqa: E402

out, leg = Path(sys.argv[1]), sys.argv[2]
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [R.RFD3_SEED]
for seed in seeds:
    R.RFD3_SEED = seed
    row = R.run_rfd3_fusion(True) if leg == "rfd3-fusion" else R.run_rfd3(True)
    row["seed"] = seed
    with out.open("a") as fp:
        fp.write(json.dumps(row, default=str) + "\n")
    print(f"[gate_seeds] {leg} seed {seed}: gate {row.get('gate')} clashes {row.get('clashes')} "
          f"breaks {row.get('breaks')} clean {row.get('clean_rate')} error {row.get('error')} "
          f"findings {row.get('findings')}", flush=True)
