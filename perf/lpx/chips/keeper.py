"""Hold one .107 chip for an LPX row without keeping the device open.

The JapanFold agent's workers wait on tt-bio's per-card flock in the shared lease dir and take a
card the moment its holder lets go. A microbench row runs many short processes, so between two of
them the agent would win the chip back. This process holds the shared flock for the row instead;
the row's own processes open the device with a PRIVATE lease dir (TT_BIO_LEASE_DIR), so they do
not contend with it. The host-wide bring-up lock is in /tmp and still serializes every open.

Exits, handing the chip back to the agent, when ~/lpx/hold/<card>.until is gone or its epoch has
passed. A row extends its hold by writing a later epoch (capped at now + 12 h here).

    TT_BIO_LEASE_HOLDER=lpx-sdpa python keeper.py <card>
"""
import os
import sys
import time

from tt_bio.device_lease import DeviceLease

card = sys.argv[1]
until = os.path.expanduser(f"~/lpx/hold/{card}.until")
lease = DeviceLease(card, timeout=30).acquire()
print(f"{time.strftime('%FT%TZ', time.gmtime())} card {card} held by "
      f"{os.environ.get('TT_BIO_LEASE_HOLDER')} pid {os.getpid()}", flush=True)
try:
    while True:
        try:
            t = min(float(open(until).read().split()[0]), time.time() + 12 * 3600)
        except (OSError, ValueError, IndexError):
            break
        if time.time() >= t:
            break
        time.sleep(10)
finally:
    lease.release()
    print(f"{time.strftime('%FT%TZ', time.gmtime())} card {card} released", flush=True)
