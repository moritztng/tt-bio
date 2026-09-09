"""Walk the RFD3 size ladder on a Wormhole Galaxy chip through the shipped design CLI.

One rung per process, always: the answer must not depend on what ran before it, and the L1
boundary this ladder is looking for is 0.39 % of a bank wide (state/ceiling-rfd3.md), so device
state carried between rungs would decide the result instead of the size.

The rung is run by invoking `tt_bio.main`'s own `design` command with the argv a user types --
not `RFD3Sampler.sample()` -- because a rung is only PASS if a real CIF lands on disk
(`verify-the-deployed-artifact-not-your-own-change`). It is invoked in-process rather than as a
subprocess for one reason: the L1-residency gate's grant/decline census lives in module state,
and a rung has to SAY whether the gate fired rather than leave an unchanged outcome to stand
for both a working decline and a dead gate (`negative-control-must-break-what-check-reads`).

    WH_TOTAL=768 WH_OUT_DIR=... WH_JSONL=... TT_VISIBLE_DEVICES=3 python3 perf/ceilrfd3/wh_ladder.py

WH_CROP names the target crop instead of the total (total becomes crop + WH_BINDER), and
WH_SPEC_ID names the output CIF, so two rungs that share a total but not a contig do not
overwrite each other.
"""
import json
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, os.getcwd())

BINDER = int(os.environ.get("WH_BINDER", "100"))
# Crop and binder are the two independent knobs; WH_TOTAL is the convenience that fixes the sum.
# Naming the crop directly is what lets a rung hold the target crop still while the binder moves,
# which is the only way "the break follows the size" and "the break follows the crop" come apart.
if "WH_CROP" in os.environ:
    CROP = int(os.environ["WH_CROP"])
    TOTAL = CROP + BINDER
else:
    TOTAL = int(os.environ["WH_TOTAL"])
    CROP = TOTAL - BINDER
STEPS = int(os.environ.get("WH_STEPS", "100"))
SEED = int(os.environ.get("WH_SEED", "42"))
TARGET = os.environ.get("WH_TARGET", "perf/ceilrfd3/targets/laczc_1008.cif")
CKPT = os.environ.get("WH_CKPT", "/home/cust-team/.boltz/rfd3/weights")
OUTD = pathlib.Path(os.environ["WH_OUT_DIR"])
JL = pathlib.Path(os.environ["WH_JSONL"])
HOST_THREADS = os.environ.get("WH_HOST_THREADS")

OUTD.mkdir(parents=True, exist_ok=True)
JL.parent.mkdir(parents=True, exist_ok=True)

spec_id = os.environ.get("WH_SPEC_ID") or "cap%d" % TOTAL
contig = "A1-%d,%d" % (CROP, BINDER)
specfile = OUTD / (spec_id + ".json")
specfile.write_text(json.dumps(
    {spec_id: {"input": TARGET, "contig": contig, "length": str(BINDER)}}, indent=2))
cif = OUTD / (spec_id + ".cif")
if cif.exists():
    cif.unlink()

argv = ["design", str(specfile), "--model", "rfd3", "--from_pdb",
        "--num_timesteps", str(STEPS), "--out_dir", str(OUTD),
        "--checkpoint", CKPT, "--seed", str(SEED)]
if HOST_THREADS:
    argv += ["--host_threads", HOST_THREADS]

rec = {"total_res": TOTAL, "target_res": CROP, "binder": BINDER, "contig": contig,
       "spec_id": spec_id, "steps": STEPS,
       "seed": SEED, "target": TARGET, "argv": " ".join(argv),
       "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "tag": os.environ.get("WH_TAG", ""), "ok": False, "atoms": None, "cif": None,
       "error": None, "pt_l1": None, "wall_s": None, "host_rss_peak_kB": None}

import tt_bio                                                        # noqa: E402
from tt_bio.main import cli                                          # noqa: E402
rec["tt_bio"] = tt_bio.__file__

t0 = time.time()
try:
    cli(argv, standalone_mode=False)
except BaseException as e:                                           # noqa: BLE001
    msg = str(e)
    rec["error"] = {"type": type(e).__name__, "msg": msg[:700],
                    "py": ["%s:%d %s" % (pathlib.Path(fs.filename).name, fs.lineno,
                                         (fs.line or "").strip())
                           for fs in traceback.extract_tb(e.__traceback__)][-14:]}
rec["wall_s"] = round(time.time() - t0, 1)

# PASS is a CIF on disk with atoms in it. Not "the call returned".
if cif.exists() and cif.stat().st_size > 0:
    n = sum(1 for ln in cif.read_text().splitlines()
            if ln.startswith("ATOM") or ln.startswith("HETATM"))
    rec["cif"] = str(cif)
    rec["atoms"] = n
    rec["ok"] = n > 0

try:
    from tt_bio.rfd3.model import PTL1STATS, PTL1DECLINES, PTL1REFUSED, ATOM_PAIR_BLOCK_STATS
    from tt_bio.tenstorrent import l1_resident_budget_bytes
    rec["pt_l1"] = {"budget_B": l1_resident_budget_bytes(),
                    "granted": PTL1STATS[0], "declined": PTL1STATS[1],
                    "declines": dict(PTL1DECLINES), "refused": sorted(PTL1REFUSED)}
    rec["atom_pair_rows"], rec["atom_pair_blocks"] = ATOM_PAIR_BLOCK_STATS
except Exception as e:                                               # noqa: BLE001
    rec["pt_l1"] = {"unavailable": type(e).__name__}

try:
    rec["host_rss_peak_kB"] = int([l.split()[1] for l in open("/proc/self/status")
                                   if l.startswith("VmHWM")][0])
except Exception:                                                    # noqa: BLE001
    pass

with JL.open("a") as fh:
    fh.write(json.dumps(rec) + "\n")
print("[wh] total=%d card=%s ok=%s atoms=%s wall=%ss %s"
      % (TOTAL, rec["card"], rec["ok"], rec["atoms"], rec["wall_s"],
         rec["error"]["type"] if rec["error"] else ""), flush=True)
print("[pt_l1] " + json.dumps(rec["pt_l1"]), flush=True)
