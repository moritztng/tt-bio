"""Does the default-ON openfold3.trunk HiFi route still fold OpenFold3 at its top size?

The release gate's size-ladder asks this too, but it folds at 6 diffusion steps, and its own
docstring says that is not a fold-quality verdict (boltz2 at 256 aa: 100 %% backbone breaks at
6 steps, 0 %% at 25). This arm asks it at 25 steps on the one model whose code path the
candidate moves, so the merge does not wait on eight untouched models' rungs.

NOT a perf cell. Each leg records its DURING-sampled AICLK because the rule is a clock beside
every runtime, but the board-pair sibling (card 1) is busy and the host is loaded, so the wall
clocks here price nothing. What the legs decide: the fold completes, the CIF has every residue,
and the geometry is no worse than the incumbent route's at the same size and seed.

Firing is counted, not assumed: tt_bio.triatt_sdpa.STATS is [served, declined] for the fused
triangle-attention route, read out of the folding process by scripts/lever_census.py. An arm
that serves zero has not exercised the candidate, whatever its geometry says.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from perf import clocksample  # noqa: E402

PY = "/home/ttuser/tt-bio-dev/env/bin/python"
OUT = WT / "perf/land_standing/out"
WORK = OUT / "of3_top_rung"
STEPS = int(os.environ.get("OF3_STEPS", 25))
SEED = int(os.environ.get("OF3_SEED", 0))
RUNGS = tuple(int(x) for x in os.environ.get("OF3_RUNGS", "1088,1024").split(","))
AB = "TT_BIO_TRIATT_SDPA_HIFI_AB"
ARMS = {"off": "-openfold3.trunk", "on": "openfold3.trunk"}


def gpu5_gate(cif):
    import importlib.util
    p = WT / "scripts/gpu_vs_tt/gpu5_accuracy_gate.py"
    spec = importlib.util.spec_from_file_location("gpu5", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gate(cif, None, None)


def leg(n, arm):
    label = f"of3_{n}_{arm}" + (f"_s{SEED}" if SEED else "")
    out_dir = WORK / label
    census = WORK / f"census_{label}.json"
    log = WORK / f"{label}.log"
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rm", "-rf", str(out_dir)], check=False)
    cmd = [PY, str(WT / "scripts/lever_census.py"), "--tt-bio", PY,
           "--label", label, "--out", str(census), "--pythonpath", str(WT), "--",
           "-m", "tt_bio.main", "predict", str(WT / f"perf/size512/fixtures/cdk2x2_{n}.yaml"),
           "--model", "openfold3", "--single_sequence", "--sampling_steps", str(STEPS),
           "--diffusion_samples", "1", "--seed", str(SEED), "--out_dir", str(out_dir)]
    env = {**os.environ, AB: ARMS[arm]}
    t0 = time.monotonic()
    with open(log, "w") as fp, clocksample.during() as clk:
        r = subprocess.run(cmd, cwd=str(WT), env=env, stdout=fp,
                           stderr=subprocess.STDOUT, timeout=3600)
    wall = time.monotonic() - t0
    row = {"rung": n, "arm": arm, "ab": ARMS[arm], "rc": r.returncode,
           "wall_s": round(wall, 3), "aiclk": clk.summary().get(0), "steps": STEPS,
           "seed": SEED}
    text = log.read_text()
    if r.returncode != 0:
        tail = [l for l in text.splitlines() if "Error" in l or "error" in l]
        row["fail_tail"] = tail[-4:] or text.splitlines()[-4:]
    cifs = sorted(out_dir.rglob("*.cif"))
    if cifs:
        row["cif"] = cifs[0].name
        row["cif_sha256"] = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16]
        try:
            g = gpu5_gate(cifs[0])
            row["geom"] = {k: g.get(k) for k in
                           ("n_ca", "clash_frac", "chain_break_frac", "ca_ca_median_A",
                            "radius_of_gyration_A", "plddt_mean")}
        except Exception as e:                                          # noqa: BLE001
            row["geom"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        c = json.loads(census.read_text())
        rows = {r2["flag"]: r2 for r2 in c["rows"]}
        # Every fused-route counter an openfold3 trunk call can land in, served/declined.
        row["firing"] = {f: [rows[f]["served"], rows[f]["declined"]]
                         for f in ("TRIATT_PERSISTENT_MASK", "SDPA_FUSED_LARGE_S",
                                   "TRIATT_HEAD_MAJOR_QKV", "SDPA_WIDE_K")
                         if f in rows}
    except Exception as e:                                              # noqa: BLE001
        row["firing"] = f"unread: {e}"
    print(json.dumps(row), flush=True)
    return row


def main():
    rows = []
    for n in RUNGS:
        for arm in ("off", "on"):
            rows.append(leg(n, arm))
            (OUT / ("of3_top_rung_ab%s.json" % (f"_s{SEED}" if SEED else ""))).write_text(json.dumps(
                {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
                 "loadavg": open("/proc/loadavg").read().split()[:3],
                 "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(WT),
                                          capture_output=True, text=True).stdout.strip(),
                 "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
