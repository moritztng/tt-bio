#!/usr/bin/env python3
"""Which combination of the trunk byte levers moves the trpcage structure?

Each flag on its own reproduces the all-off CIF to the byte; all three together do not. So the
effect is an interaction, not one lever's arithmetic. Fold the three pairs against the same
all-off reference to locate it.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEG_ID = "boltz2-trpcage-nomsa"
QKVG, QKVGB, GOUT = ("TT_BIO_TRIATT_FUSED_QKVG", "TT_BIO_TRIATT_FUSED_QKVGB",
                     "TT_BIO_TRIMUL_FUSED_GOUT")
FLAGS = (QKVG, QKVGB, GOUT)
PAIRS = {"qkvg+qkvgb": (QKVG, QKVGB), "qkvg+gout": (QKVG, GOUT), "qkvgb+gout": (QKVGB, GOUT)}


def load_gate():
    path = ROOT / "scripts" / "full_parity_gate.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("full_parity_gate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    card = os.environ.get("CARD", "3")
    gate = load_gate()
    leg = gate.LEGS_BY_ID[LEG_ID]
    work = ROOT / "perf" / "b2z2_trunk_ship" / "trpcage_pairs_work"
    work.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, on in PAIRS.items():
        out_dir = work / name.replace("+", "_")
        subprocess.run(["rm", "-rf", str(out_dir)], check=True)
        cmd = gate.device_cmd(leg, 0, out_dir, work)
        env = dict(os.environ)
        env.update({f: ("1" if f in on else "0") for f in FLAGS})
        env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                    "TT_BIO_LEASE_HOLDER": "worker:b2z2-trunk-byte-round2-ship",
                    "PYTHONPATH": str(ROOT)})
        log = work / (name.replace("+", "_") + ".log")
        print("[%s]" % name, flush=True)
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit("[%s] fold failed rc=%d, see %s" % (name, rc, log))
        cifs = sorted(out_dir.rglob("*.cif"))
        assert len(cifs) == 1, cifs
        results[name] = hashlib.sha256(cifs[0].read_bytes()).hexdigest()
        print("[%s] sha256 %s" % (name, results[name]), flush=True)

    ref = json.load(open(ROOT / "perf/b2z2_trunk_ship/trpcage_bitexact_ab.json"))
    off = ref["arms"]["off"]["sha256"]
    allon = ref["arms"]["on-1"]["sha256"]
    out = {"leg": LEG_ID, "card": card, "all_off_sha256": off, "all_on_sha256": allon,
           "pairs": results,
           "bit_exact_vs_off": {n: (h == off) for n, h in results.items()},
           "equals_all_on": {n: (h == allon) for n, h in results.items()}}
    (ROOT / "perf/b2z2_trunk_ship/trpcage_pairs.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"bit_exact_vs_off": out["bit_exact_vs_off"],
                      "equals_all_on": out["equals_all_on"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
