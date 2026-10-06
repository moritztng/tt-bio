"""CPU test of the engine's hang recovery: a chip that hangs is reset with its board mate, a chip
that keeps hanging rests and rejoins by itself, and its mate keeps folding. No device: a stand-in
chipworker hangs (tt-metal's chip_hung, exit 75) on every fold on chip 0 and folds fine on chip 1.
A second run makes chip 1 ignore SIGINT and SIGTERM, as a warming worker does: the board must
never be reset under it, chip 0 rests instead, and chip 1 keeps folding.

    python3 demo/booth/engine/tests/rest_test.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
FAKE = r'''
import json, os, signal, sys, time
chip = int(sys.argv[sys.argv.index("--chip") + 1])
if os.environ.get("FAKE_STUBBORN") and chip == 1:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
def emit(**e): print(json.dumps(e), flush=True)
emit(type="chip", chip=chip, state="ready")
for line in sys.stdin:
    j = json.loads(line)
    emit(type="fold_start", id=j["id"], chip=chip)
    time.sleep(0.3)
    if chip == 0:
        emit(type="fold_error", id=j["id"], chip=chip, reason="chip_hung")
        os._exit(75)
    emit(type="fold_done", id=j["id"], chip=chip, seconds=0.3)
    emit(type="chip", chip=chip, state="ready")
'''


def run(stubborn, seconds, extra):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "engine").mkdir()
    for f in ENGINE.glob("*.py"):  # the engine as deployed, with a stand-in chipworker
        shutil.copy(f, tmp / "engine")
    (tmp / "engine/chipworker.py").write_text(FAKE)
    (tmp / "hardware").symlink_to(ENGINE.parent / "hardware")
    (tmp / "attract.json").write_text(json.dumps([{"name": "Top7", "sequence": "ACDEFGHIK" * 10}]))
    reset = tmp / "reset.sh"
    reset.write_text(f"#!/bin/sh\necho \"$(date +%s) $1\" >> {tmp}/resets\nsleep 1\n")
    reset.chmod(0o755)
    port = 8699
    srv = subprocess.Popen([sys.executable, str(tmp / "engine/server.py"), "--chips", "0,1", "--port", str(port),
                            "--attract", str(tmp / "attract.json"), "--reset-cmd", str(reset), "--no-telemetry",
                            "--logdir", str(tmp / "logs"), "--record", "", "--replay", str(tmp),
                            "--reset-min-gap", "0", "--rest-after", "2", "--rest-s", "4", "--rest-window", "60",
                            *extra], stdout=open(tmp / "server.log", "w"), stderr=subprocess.STDOUT,
                           env=dict(os.environ, **({"FAKE_STUBBORN": "1"} if stubborn else {})))
    seen, t0 = [], time.time()
    try:
        while time.time() - t0 < seconds:
            time.sleep(0.25)
            try:
                st = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2))
            except OSError:
                continue
            row = {c["chip"]: (c["state"], c.get("back_at"), c["folds"]) for c in st["chips"]}
            if not seen or {k: v[0] for k, v in row.items()} != {k: v[0] for k, v in seen[-1][1].items()}:
                seen.append((round(time.time() - t0, 1), row))
    finally:
        srv.send_signal(2)
        srv.wait(30)
    for t, row in seen:
        print(t, row)
    resets = (tmp / "resets").read_text().split("\n") if (tmp / "resets").exists() else []
    rests = [t for t, row in seen if row[0][0] == "resting"]
    back = [t for i, (t, row) in enumerate(seen) if i and seen[i - 1][1][0][0] == "resting" and row[0][0] != "resting"]
    folds1 = seen[-1][1][1][2]
    n = len([r for r in resets if r])
    print(f"stubborn={stubborn} resets={n} rest_starts={len(rests)} rejoins={len(back)} chip1_folds={folds1}")
    assert rests, "chip 0 never rested"
    assert folds1 > 10, "chip 1 stopped folding"
    if stubborn:
        assert n == 0, "the board was reset while chip 1 was still working"
    else:
        assert any(row[0][1] for t, row in seen if row[0][0] == "resting"), "a resting chip carries back_at"
        assert len(back) >= 1, "chip 0 never rejoined after resting"


def main():
    run(False, 40, [])
    run(True, 25, ["--term-s", "0.5", "--warm-s", "1.5"])
    print("PASS")


if __name__ == "__main__":
    main()
