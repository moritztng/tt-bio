#!/usr/bin/env python3
"""Is boltz2-trpcage-nomsa bit-exact under the three trunk byte levers?

The 44-leg control found this leg PASS in both runs but with the envelope numerator moving
0.0778 -> 0.0808 against the current-main baseline, while every other scored leg reproduced to
the digit. Either the levers are not bit-exact at this shape, or this fold is not deterministic
run to run. Those need different answers, so measure both:

  on-1   flags at this branch's defaults (all three ON)
  off    all three forced OFF -- this is main's numerics
  on-2   flags ON again, a same-code repeat that separates lever effect from run-to-run drift

Bit-exact means sha256(CIF) equal. The fold command is built by the gate's OWN device_cmd() for
the same Leg, so this reproduces the gate's fold rather than an approximation of it.
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
FLAGS = ("TT_BIO_TRIATT_FUSED_QKVG", "TT_BIO_TRIATT_FUSED_QKVGB", "TT_BIO_TRIMUL_FUSED_GOUT")


def load_gate():
    path = ROOT / "scripts" / "full_parity_gate.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("full_parity_gate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_cif(out_dir):
    cifs = sorted(out_dir.rglob("*.cif"))
    if len(cifs) != 1:
        raise SystemExit("expected exactly one CIF under %s, found %d" % (out_dir, len(cifs)))
    return cifs[0]


def main():
    card = os.environ.get("CARD", "3")
    gate = load_gate()
    leg = gate.LEGS_BY_ID[LEG_ID]
    work = ROOT / "perf" / "b2z2_trunk_ship" / "trpcage_ab_work"
    work.mkdir(parents=True, exist_ok=True)

    arms = [("on-1", "1"), ("off", "0"), ("on-2", "1")]
    results = {}
    for name, val in arms:
        out_dir = work / name
        subprocess.run(["rm", "-rf", str(out_dir)], check=True)
        cmd = gate.device_cmd(leg, 0, out_dir, work)
        env = dict(os.environ)
        env.update({f: val for f in FLAGS})
        env.update({
            "TT_VISIBLE_DEVICES": card,
            "TT_BIO_LEASE_CARDS": card,
            "TT_BIO_LEASE_HOLDER": "worker:b2z2-trunk-byte-round2-ship",
            "PYTHONPATH": str(ROOT),
        })
        log = work / ("%s.log" % name)
        print("[%s] flags=%s  %s" % (name, val, " ".join(cmd)), flush=True)
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit("[%s] fold failed rc=%d, see %s" % (name, rc, log))
        cif = find_cif(out_dir)
        results[name] = {"flags": val, "cif": str(cif.relative_to(ROOT)), "sha256": digest(cif)}
        print("[%s] sha256 %s" % (name, results[name]["sha256"]), flush=True)

    gate_cif = (ROOT / "perf/b2z2_gate/trunkship_work" / LEG_ID / "seed0"
                / "boltz2_results_trpcage_no_msa/structures/trpcage_no_msa.cif")
    if gate_cif.exists():
        results["gate-run"] = {"flags": "1 (branch default)",
                               "cif": str(gate_cif.relative_to(ROOT)),
                               "sha256": digest(gate_cif)}

    verdict = {
        "lever_bit_exact": results["on-1"]["sha256"] == results["off"]["sha256"],
        "run_to_run_deterministic": results["on-1"]["sha256"] == results["on-2"]["sha256"],
    }
    out = {"leg": LEG_ID, "card": card, "arms": results, "verdict": verdict}
    (ROOT / "perf/b2z2_trunk_ship/trpcage_bitexact_ab.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
