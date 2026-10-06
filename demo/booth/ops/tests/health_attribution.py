"""health attributes injected restarts: 5 kiosk restarts, 4 right after chaos marks -> not NOT HEALTHY for restarts."""
import json, os, sys, tempfile, time, importlib.util
from pathlib import Path
d = Path(tempfile.mkdtemp())
spec = importlib.util.spec_from_file_location("health", sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "health.py")); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
now = time.time()
h.LOG, h.CHAOS = d / "watchdog.jsonl", d / "chaos.jsonl"
rows = [{"t": now - 30, "ev": "tick", "screen": "moving", "fps": 60, "disk_free_gb": 100, "mem_avail_gb": 50}]
marks = []
for k in range(5):
    t = now - 3000 + k * 600
    rows.insert(0, {"t": t + 20, "ev": "restart", "unit": "booth-kiosk", "why": "x"})
    if k: marks.append({"t": t, "event": "browser_crash"})
h.LOG.write_text("\n" + "\n".join(json.dumps(r) for r in sorted(rows, key=lambda r: r["t"])) + "\n")
h.CHAOS.write_text("\n" + "\n".join(json.dumps(m) for m in marks) + "\n")
from io import StringIO
import contextlib
buf = StringIO()
with contextlib.redirect_stdout(buf):
    h.main()
out = buf.getvalue(); print(out)
assert "4 or more" not in out and "4 of them after a test" in out, "FAIL"
h.CHAOS.write_text("")
buf = StringIO()
with contextlib.redirect_stdout(buf):
    h.main()
assert "4 or more" in buf.getvalue(), "FAIL control"
print("PASS")
