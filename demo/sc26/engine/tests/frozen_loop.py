"""The engine with its event loop blocked after --after seconds, as a hung engine would be.

    systemd-run --user -p WatchdogSec=10 -p NotifyAccess=main -p LimitCORE=0 -p Restart=always \\
        python3 demo/sc26/engine/tests/frozen_loop.py --after 20 -- --replay-only --port 8699

With the unit's WatchdogSec the loop stops saying WATCHDOG=1, systemd aborts the process and starts
it again (ops/units/sc26-engine.service). Without it the engine would answer nothing forever.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--after", type=float, default=20)
a, rest = ap.parse_known_args()
real = server.Service.main


async def main(self):
    asyncio.get_running_loop().call_later(a.after, lambda: (print("blocking the loop", flush=True), time.sleep(10**6)))
    await real(self)

server.Service.main = main
sys.argv = [sys.argv[0]] + [x for x in rest if x != "--"]
server.main()
