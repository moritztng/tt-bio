"""The timing plan: which model walks which rungs, and a free chip for each.

    python perf/mgx-speed/plan.py            # print the plan and the chips it would take
    python perf/mgx-speed/plan.py go [model ...]   # launch (all models, or the named ones)

The model list is read off tt_bio.main (PREDICT_MODELS, plus nesso1 from AFFINITY_MODELS), not
retyped. A chip is free when nobody holds its flock and its lease names no live pid; cards 1 and
24-27 (the co-tenant and the live app) are never taken.
"""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from tt_bio.main import AFFINITY_MODELS, PREDICT_MODELS  # noqa: E402

NEVER = {1, 24, 25, 26, 27}
FIT = (512, 640, 768, 896, 1024)
TOP = {"rf3": (1088, 1280, 1536)}          # rf3's ladder carries a 1088 rung
MODELS = list(PREDICT_MODELS) + [m for m in AFFINITY_MODELS if m == "nesso1"]
LEASES = Path(os.environ.get("TT_BIO_LEASE_DIR", Path.home() / "leases"))


def rungs(model):
    return FIT + TOP.get(model, (1280, 1536))


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def free(card):
    path = next(LEASES.glob(f"*-card{card}.json"), None)
    if path is None:
        return True
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        return False
    finally:
        os.close(fd)
    try:
        meta = json.loads(path.read_text() or "{}")
    except Exception:
        return False
    return bool(meta.get("released")) or not alive(meta.get("pid") or 0)


def main(argv):
    go = argv[1:2] == ["go"]
    want = argv[2:] or MODELS
    chips = [c for c in range(32) if c not in NEVER and free(c)]
    for m in want:
        card = chips.pop(0) if chips else None
        print(f"{m:14s} card {card}  rungs {','.join(map(str, rungs(m)))}")
        if go and card is not None:
            subprocess.Popen(["setsid", "nohup", "bash", str(ROOT / "perf/mgx-speed/launch.sh"), str(card), m,
                              ",".join(map(str, rungs(m)))], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)


if __name__ == "__main__":
    main(sys.argv)
