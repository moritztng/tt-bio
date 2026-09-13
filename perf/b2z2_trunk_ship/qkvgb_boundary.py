#!/usr/bin/env python3
"""Where between 20 and 117 residues does TT_BIO_TRIATT_FUSED_QKVGB stop being bit-exact?

trpcage (20 aa) is not bit-exact under this flag; prot (117 aa) and hsa (585 aa) are, as is the
512-1536 ladder. Sweep synthetic chains across that gap and fold each one twice, with QKVGB the
only thing that differs between the arms (QKVG is on in both because QKVGB is inert without it,
GOUT is off in both so it cannot contribute). Bit-exact means sha256(CIF) equal.

The sequence content does not matter here -- the arms fold the SAME sequence and the question is
only whether the two structures agree to the byte -- so the chains are the trpcage sequence tiled
and cut to length, which keeps them valid residues.

Lengths are chosen around the 32-residue tile: 32 fills one tile of the pair tensor, 48 and 64 two,
96 three, 112 four. If the boundary sits on a tile count, that is where it shows.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "NLYIQWLKDGGPSSGRPPPS"
LENGTHS = [32, 48, 64, 96, 112]
QKVG, QKVGB, GOUT = ("TT_BIO_TRIATT_FUSED_QKVG", "TT_BIO_TRIATT_FUSED_QKVGB",
                     "TT_BIO_TRIMUL_FUSED_GOUT")


def seq(n):
    return (BASE * (n // len(BASE) + 1))[:n]


def main():
    card = os.environ.get("CARD", "3")
    py = sys.executable
    work = ROOT / "perf" / "b2z2_trunk_ship" / "qkvgb_boundary_work"
    work.mkdir(parents=True, exist_ok=True)

    rows = {}
    for n in LENGTHS:
        yaml = work / ("chain%d.yaml" % n)
        yaml.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n"
                        "      sequence: %s\n      msa: empty\n" % seq(n))
        digests = {}
        for arm, qkvgb in (("qkvgb-off", "0"), ("qkvgb-on", "1")):
            out_dir = work / ("n%d" % n) / arm
            subprocess.run(["rm", "-rf", str(out_dir)], check=True)
            cmd = [py, "-m", "tt_bio.main", "predict", str(yaml.relative_to(ROOT)),
                   "--model", "boltz2", "--out_dir", str(out_dir), "--override", "--seed", "0",
                   "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1"]
            env = dict(os.environ)
            env.update({QKVG: "1", QKVGB: qkvgb, GOUT: "0"})
            env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                        "TT_BIO_LEASE_HOLDER": "worker:b2z2-trunk-byte-round2-ship",
                        "PYTHONPATH": str(ROOT)})
            log = work / ("n%d_%s.log" % (n, arm))
            with open(log, "w") as lf:
                rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf,
                                    stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise SystemExit("n=%d %s failed rc=%d, see %s" % (n, arm, rc, log))
            cifs = sorted(out_dir.rglob("*.cif"))
            assert len(cifs) == 1, (n, arm, cifs)
            digests[arm] = hashlib.sha256(cifs[0].read_bytes()).hexdigest()
        rows[n] = {"sha256": digests, "bit_exact": digests["qkvgb-off"] == digests["qkvgb-on"]}
        print("n=%-4d bit_exact=%s" % (n, rows[n]["bit_exact"]), flush=True)

    out = {"flag": QKVGB, "card": card, "known_break": 20, "known_clean": [117, 585],
           "lengths": rows,
           "bit_exact": {str(n): rows[n]["bit_exact"] for n in LENGTHS}}
    (ROOT / "perf/b2z2_trunk_ship/qkvgb_boundary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["bit_exact"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
