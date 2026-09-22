"""Does the lever move the other four models' shipped output, and by how much?

One fold per model per arm on the production CLI, same card, same seed. Reports the
structure-file sha256 for each arm (a digest that moves means the forward moved) and the
CA-RMSD between the arms in Angstrom, which is the number that decides whether it matters.

The NEGATIVE CONTROL is the point of the third arm: `ctl` folds with a deliberately
different seed. If `ctl`'s digest did not move either, the digest check is reading nothing
and the whole table is worthless. It must move.

Resumable: every completed arm writes done.json and is skipped on the next run, so a
relaunch continues instead of re-folding.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf import clocksample                                   # noqa: E402
from perf.of3t_softmax.fold_ab import run_arm, ca_coords, kabsch_rmsd   # noqa: E402

# model -> (CLI --model, input yaml, extra CLI-relevant notes)
MODELS = {
    "openfold3":   ("openfold3", "examples/ubq.yaml"),
    "boltz2":      ("boltz2", "examples/ubq.yaml"),
    "protenix-v2": ("protenix-v2", "examples/ubq.yaml"),
    "rf3":         ("rf3", "examples/ubq.yaml"),
    "opendde":     ("opendde", "examples/ubq.yaml"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="boltz2,protenix-v2,rf3")
    ap.add_argument("--workdir", default="/home/ttuser/of3t_softmax_work/models")
    ap.add_argument("--out", default="perf/of3t_softmax/models_digest_qb2c0.json")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctl-seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=3000)
    a = ap.parse_args()

    out_path = Path(a.out)
    out = json.loads(out_path.read_text()) if out_path.is_file() else {
        "host": "qb2", "card": 0, "board": "p300c", "samples": a.samples, "models": {}}

    with clocksample.during(period=5.0) as clk:
        for m in a.models.split(","):
            cli, inp = MODELS[m]
            wd = Path(a.workdir) / m
            rec = out["models"].setdefault(m, {})
            for arm, env_ab, seed in (("off", None, a.seed),
                                      ("on", "all", a.seed),
                                      ("ctl", None, a.ctl_seed)):
                r = run_arm(f"{m}-{arm}", cli, ROOT / inp, wd, env_ab, seed, a.samples,
                            timeout=a.timeout)
                rec[arm] = {k: r[k] for k in
                            ("rc", "wall_s", "sha256_structures", "cifs", "seed")}
                if r["rc"] != 0:
                    rec[arm]["stderr_tail"] = r.get("stderr_tail", "")[-1500:]
                print(m, arm, r["rc"], round(r["wall_s"], 1), r["sha256_structures"],
                      flush=True)
            for label, b in (("off_vs_on", "on"), ("off_vs_ctl", "ctl")):
                po = [Path(p) for p in sorted(rec["off"].get("cifs") or [])]
                pb = [Path(p) for p in sorted(rec[b].get("cifs") or [])]
                if po and pb and po[0].is_file() and pb[0].is_file():
                    rec[f"{label}_rank0_ca_rmsd_A"] = kabsch_rmsd(
                        ca_coords(po[0]), ca_coords(pb[0]))
                    rec[f"{label}_digest_moved"] = (
                        rec["off"]["sha256_structures"] != rec[b]["sha256_structures"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2))

    out["clock_aiclk_during"] = clk.summary()
    out["clock_line"] = clk.line(0)
    out_path.write_text(json.dumps(out, indent=2))
    print(clk.line(0))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
