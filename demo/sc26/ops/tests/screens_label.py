"""screens.py labels a poster, a live page, a black frame and a frozen page, and fails on the last two.

    python3 demo/sc26/ops/tests/screens_label.py ~/sc26-logs/soak-1005T2100/shots/000-browser_crash-005s.ppm
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

ops = Path(__file__).resolve().parents[1]
page = Path(sys.argv[1]).expanduser()
d = Path(tempfile.mkdtemp())
(d / "shots").mkdir()
size = Image.open(page).size
ImageOps.fit(Image.open(ops / "session" / "poster.png").convert("RGB"), size).save(d / "shots" / "000-x-002s.ppm")
shutil.copy(page, d / "shots" / "000-x-005s.ppm")
shutil.copy(page, d / "shots" / "000-x-010s.ppm")      # identical page twice: frozen
Image.new("RGB", size, (0, 0, 0)).save(d / "shots" / "000-x-020s.ppm")
(d / "events.jsonl").write_text(json.dumps({"i": 0, "event": "x", "screen": [{"t": t} for t in (2, 5, 10, 20, 40)]}) + "\n")
r = subprocess.run([sys.executable, ops / "screens.py", d], capture_output=True, text=True)
print(r.stdout, end="")
want = "2s:poster 5s:page 10s:same 20s:flat 40s:none"
ok = want in r.stdout and r.returncode == 1
print("PASS" if ok else f"FAIL: wanted {want!r} and rc 1, rc {r.returncode}")
shutil.rmtree(d)
sys.exit(0 if ok else 1)
