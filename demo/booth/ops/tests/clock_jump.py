"""A wall-clock step (NTP at the booth) must not change any lane's state. Reads sysfs only and
opens no device, so it runs next to the live demo.

    python3 demo/booth/ops/tests/clock_jump.py [path/to/telemetry.py]

Before the fix a +1 h step made every lane read "resetting" until its next heartbeat.
"""
import importlib.util
import sys
import time
from pathlib import Path

src = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "hardware" / "telemetry.py")
spec = importlib.util.spec_from_file_location("tel", src)
tel = importlib.util.module_from_spec(spec)
sys.modules["tel"] = tel
spec.loader.exec_module(tel)

m = tel.Monitor(events=Path("/nonexistent/folds.jsonl"))
m.sample()
time.sleep(0.3)
m.sample()
cards = sorted(m.latest)
base = [m.state(c, time.time()) for c in cards]
print("before    ", base)
real = time.time
try:
    for jump in (+3600, -3600):
        tel.time.time = lambda j=jump: real() + j
        m.sample()
        got = [m.state(c, tel.time.time()) for c in cards]
        print(f"step {jump:+d}", got)
        if any(g == "resetting" != b for g, b in zip(got, base)):
            sys.exit("FAIL: a clock step made a beating chip read resetting")
finally:
    tel.time.time = real
print("PASS")
