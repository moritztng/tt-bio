"""The af2ig route on a card: the shipped CLI once, then the same fold timed in-process.

The CLI leg is the proof that `tt-bio predict <designed complex> --model af2ig` reaches the ttnn
trunk and writes af2ig's metrics, with nothing outside the request. The timed legs load the same
device model once and fold the example twice (first fold pays the kernel compile) and once with
the binder scrambled, sampling the card's AICLK from sysfs at 5 Hz DURING each fold. Nothing here
compares against CUDA; the agreement quoted is against the host-torch run of the same example
(perf/cmp_af2ig/cpu_route.json) on the same parameters.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:cmp-af2ig \
        PYTHONPATH=. python3 perf/cmp_af2ig/device_route.py --card 2 \
        --out perf/cmp_af2ig/device_route_qb2_card2.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpu_route import BINDER, EXAMPLE, SCRAMBLED, _digest  # noqa: E402


class Clock(threading.Thread):
    """AICLK and board power from sysfs at 5 Hz. Reads files only, never opens the device."""

    def __init__(self, card: str):
        super().__init__(daemon=True)
        root = Path("/sys/class/tenstorrent") / f"tenstorrent!{card}"
        self.clk = root / "tt_aiclk"
        self.pw = next(iter(root.glob("device/hwmon/hwmon*/power1_input")), None)
        self.stop, self.aiclk, self.power = threading.Event(), [], []
        self.start()

    def run(self):
        while not self.stop.wait(0.2):
            try:
                self.aiclk.append(int(self.clk.read_text().strip()))
                if self.pw is not None:
                    self.power.append(int(self.pw.read_text().strip()) / 1e6)
            except (OSError, ValueError):
                pass

    def take(self) -> dict:
        a, w = self.aiclk[:], self.power[:]
        self.aiclk.clear()
        self.power.clear()
        if not a:
            return {"aiclk_n": 0}
        return {"aiclk_mean": round(sum(a) / len(a), 1), "aiclk_min": min(a),
                "aiclk_max": max(a), "aiclk_frac_ge_1200": round(sum(x >= 1200 for x in a) / len(a), 3),
                "aiclk_n": len(a), "power_w_mean": round(sum(w) / len(w), 1) if w else None}


def cli_leg(clock: Clock) -> dict:
    """`tt-bio predict --model af2ig` exactly as a user runs it, in its own process."""
    out = Path(tempfile.mkdtemp(prefix="af2ig_cli_"))
    cmd = [sys.executable, "-c", "from tt_bio.main import cli; cli()", "predict", str(EXAMPLE),
           "--model", "af2ig", "--out_dir", str(out)]
    clock.take()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    wall = round(time.perf_counter() - t0, 3)
    leg = {"cmd": " ".join(["tt-bio"] + cmd[3:]), "returncode": proc.returncode,
           "process_wall_s": wall, **clock.take(), "out_dir": str(out),
           "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())}
    rows = sorted(out.rglob("*.json"))
    leg["result_json"] = {str(p.relative_to(out)): json.loads(p.read_text())
                          for p in rows if p.stat().st_size < 200_000}
    if proc.returncode:
        leg["stderr_tail"] = proc.stderr[-4000:]
    return leg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--params", default="~/.boltz/af2/params/params_model_1_ptm.npz")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    clock = Clock(args.card)
    report = {"host": socket.gethostname(), "card": args.card,
              "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
              "recycles": args.recycles, "passes_per_fold": args.recycles + 1}
    report["cli"] = cli_leg(clock)

    from tt_bio import af2ig
    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict

    t0 = time.perf_counter()
    model = load_af2_device_model(load_af2_state_dict(str(Path(args.params).expanduser())),
                                  template=True).eval()
    report["model_load_s"] = round(time.perf_counter() - t0, 3)
    spec = af2ig.read_af2ig_input(EXAMPLE)
    assert spec.binder_sequence == BINDER
    other = af2ig.AF2IGInput(structure=spec.structure, binder_sequence=SCRAMBLED,
                             target_chain=spec.target_chain, binder_chain=spec.binder_chain)
    legs = {}
    for name, s in (("first", spec), ("warm", spec), ("warm_again", spec), ("scrambled", other)):
        clock.take()
        t0 = time.perf_counter()
        pred = af2ig.fold(model, s, recycles=args.recycles)
        legs[name] = {"fold_s": round(time.perf_counter() - t0, 3), **clock.take(),
                      "tokens": pred.tokens, "coords_sha16": _digest(pred.coords),
                      "metrics": pred.metrics}
        print(name, json.dumps(legs[name]), flush=True)
    report["legs"] = legs
    report["deterministic_warm"] = (legs["warm"]["coords_sha16"] == legs["warm_again"]["coords_sha16"]
                                    and legs["warm"]["metrics"] == legs["warm_again"]["metrics"])
    report["sequence_reaches_the_model"] = legs["warm"]["coords_sha16"] != legs["scrambled"]["coords_sha16"]
    host = json.loads((Path(__file__).parent / "cpu_route.json").read_text())["legs"]["example"]["metrics"]
    report["vs_host_torch"] = {k: round(legs["warm"]["metrics"][k] - v, 4) for k, v in host.items()}
    text = json.dumps(report, indent=1) + "\n"
    Path(args.out).write_text(text)
    print(text, end="")
    return 0 if report["cli"]["returncode"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
