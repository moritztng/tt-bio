#!/usr/bin/env python3
"""Does the single-tile guard restore bit-exactness for the whole three-flag stack?

`qkvgb_boundary.json` put the QKVGB difference at chains of at most one 32-residue tile on the
token axis. The guard declines exactly those. This folds the short chains that were broken (20,
32) and the first two that were already clean (48, 64) with all three flags on against all three
off, so a pass means the stack writes the shipped structure at every length, not just above the
tile.
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "NLYIQWLKDGGPSSGRPPPS"
LENGTHS = [20, 32, 48, 64]
FLAGS = ("TT_BIO_TRIATT_FUSED_QKVG", "TT_BIO_TRIATT_FUSED_QKVGB", "TT_BIO_TRIMUL_FUSED_GOUT")


def main():
    card = os.environ.get("CARD", "3")
    work = ROOT / "perf/b2z2_trunk_ship/qkvgb_guard_work"
    work.mkdir(parents=True, exist_ok=True)
    rows = {}
    for n in LENGTHS:
        yaml = work / ("chain%d.yaml" % n)
        yaml.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n"
                        "      sequence: %s\n      msa: empty\n"
                        % (BASE * (n // len(BASE) + 1))[:n])
        digests = {}
        for arm, v in (("all-off", "0"), ("all-on", "1")):
            out_dir = work / ("n%d" % n) / arm
            subprocess.run(["rm", "-rf", str(out_dir)], check=True)
            env = dict(os.environ)
            env.update({f: v for f in FLAGS})
            env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                        "TT_BIO_LEASE_HOLDER": "worker:b2z2-trunk-byte-round2-ship",
                        "PYTHONPATH": str(ROOT)})
            log = work / ("n%d_%s.log" % (n, arm))
            with open(log, "w") as lf:
                rc = subprocess.run(
                    [sys.executable, "-m", "tt_bio.main", "predict", str(yaml.relative_to(ROOT)),
                     "--model", "boltz2", "--out_dir", str(out_dir), "--override", "--seed", "0",
                     "--recycling_steps", "3", "--sampling_steps", "200",
                     "--diffusion_samples", "1"],
                    cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise SystemExit("n=%d %s rc=%d, see %s" % (n, arm, rc, log))
            cifs = sorted(out_dir.rglob("*.cif"))
            assert len(cifs) == 1, (n, arm, cifs)
            digests[arm] = hashlib.sha256(cifs[0].read_bytes()).hexdigest()
        rows[str(n)] = {"sha256": digests, "bit_exact": digests["all-off"] == digests["all-on"]}
        print("n=%-4d bit_exact=%s" % (n, rows[str(n)]["bit_exact"]), flush=True)
    out = {"guard": "single_tile_axis", "card": card, "arms": "all three flags on vs all off",
           "lengths": rows,
           "bit_exact": {k: v["bit_exact"] for k, v in rows.items()}}
    (ROOT / "perf/b2z2_trunk_ship/qkvgb_guard_verify.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["bit_exact"], indent=1))
    return 0 if all(out["bit_exact"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
