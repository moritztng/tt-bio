"""Screenshot every mockup screen at 1920x1080 and 3840x2160 with headless Chrome."""
import subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
layers = sys.argv[2]  # dir holding the scene.py PNGs, linked in as L/
(here / "L").unlink(missing_ok=True); (here / "L").symlink_to(layers)
for sid in sys.argv[3].split(","):
    for scale, tag in ((1, "1080"), (2, "4k")):
        f = out / f"{sid}-{tag}.png"
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--allow-file-access-from-files", f"--force-device-scale-factor={scale}",
                        "--window-size=1920,1080", "--virtual-time-budget=3000",
                        f"--screenshot={f}", f"file://{here}/screens.html#{sid}"],
                       check=True, capture_output=True, timeout=120)
        print(f)
