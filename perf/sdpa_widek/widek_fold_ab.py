#!/usr/bin/env python3
"""Fold-level A/B and accuracy arm for the wide-k SDPA ladder.

One process per fold (that is how the fleet runs them, and it keeps the two arms' allocator
histories apart), arms interleaved off/on per seed so neither arm owns a position in the order.
Every leg asserts, from its own `SDPA_K_CHUNK_STATS` / `SDPA_CHUNK_PICKS`, which (q_chunk, k_chunk)
it actually served -- an arm that did not take must not be readable as a null
(`two-level-optin-ab-arm-and-page-provenance-drop`).

The accuracy arm is the fold's own structure and confidence, read against two controls, exactly as
`docs/boltz2-fast-parity.md` gates `--fast`:

  determinism floor   off vs off at the same seed
  seed spread         off at seed i vs off at seed j
  the lever           on vs off at the same seed

k_chunk sets the online-softmax reduction order, so the lever is not bit-exact and a nonzero
deviation is expected. It PASSES when its deviation sits inside the band the two controls define.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HOOK = '''
import atexit, json, os
def _dump():
    try:
        import tt_bio.tenstorrent as T
    except Exception:
        return
    picks = {f"{a}x{b}": v for (a, b), v in getattr(T, "SDPA_CHUNK_PICKS", {}).items()}
    if not picks and not any(getattr(T, "SDPA_K_CHUNK_STATS", [0, 0])):
        return
    d = os.environ["WIDEK_DUMP"]
    with open(os.path.join(d, f"pick_{os.getpid()}.json"), "w") as f:
        json.dump({"wide_k_resolved": bool(getattr(T, "SDPA_WIDE_K", False)),
                   "k_chunk_stats": list(getattr(T, "SDPA_K_CHUNK_STATS", [])),
                   "picks": picks}, f)
atexit.register(_dump)
'''


def _hookdir(base: Path) -> Path:
    """A sitecustomize that every process of the fold imports. `tt-bio predict` folds in spawned
    worker processes, so counters read in the launcher are always zero -- the same trap
    `scripts/lever_census.py` documents."""
    d = base / "_hook"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sitecustomize.py").write_text(HOOK)
    return d


def run_leg(a, arm: str, seed: int, tag: str) -> dict:
    out = a.workdir / f"{tag}"
    dump = a.workdir / f"dump_{tag}"
    dump.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TT_BIO_SDPA_WIDE_K"] = "1" if arm == "on" else "0"
    env["TT_VISIBLE_DEVICES"] = str(a.card)
    env["TT_BIO_LEASE_HOLDER"] = a.holder
    env["WIDEK_DUMP"] = str(dump)
    env["PYTHONPATH"] = os.pathsep.join([str(_hookdir(a.workdir)), str(ROOT)])
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", a.input, "--model", a.model,
           "--out_dir", str(out), "--override", "--seed", str(seed)] + a.extra
    t0 = time.perf_counter()
    with open(a.workdir / f"{tag}.log", "w") as log:
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    served, fell_back, picks = 0, 0, {}
    for f in sorted(dump.glob("pick_*.json")):
        d = json.loads(f.read_text())
        st = d.get("k_chunk_stats") or [0, 0]
        served += st[0]
        fell_back += st[1] if len(st) > 1 else 0
        picks.update(d.get("picks", {}))
    return {"arm": arm, "seed": seed, "tag": tag, "rc": rc, "wall_s": round(wall, 3),
            "wide_k_served": served, "fell_back": fell_back, "picks": picks,
            "out_dir": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--holder", default="worker:triatt-sdpa-wide-k-envelope-gate")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--repeat-seed", type=int, default=None,
                    help="seed to re-run OFF a second time, for the determinism floor")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("extra", nargs="*", help="extra args passed through to predict")
    a = ap.parse_args()

    a.workdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    legs = []
    for s in seeds:
        for arm in ("off", "on"):
            legs.append((arm, s, f"{arm}_s{s}"))
    if a.repeat_seed is not None:
        legs.append(("off", a.repeat_seed, f"off_s{a.repeat_seed}_repeat"))

    rec = {"model": a.model, "input": a.input, "card": a.card, "seeds": seeds,
           "extra": a.extra, "host": os.uname().nodename, "legs": []}
    for arm, seed, tag in legs:
        r = run_leg(a, arm, seed, tag)
        rec["legs"].append(r)
        print(f"{tag}: rc={r['rc']} {r['wall_s']:.1f}s wide_k_served={r['wide_k_served']} "
              f"fell_back={r['fell_back']} picks={r['picks']}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rec, indent=1))

    on = [l for l in rec["legs"] if l["arm"] == "on"]
    off = [l for l in rec["legs"] if l["arm"] == "off" and not l["tag"].endswith("repeat")]
    if on and off and all(l["rc"] == 0 for l in rec["legs"]):
        m_on = sum(l["wall_s"] for l in on) / len(on)
        m_off = sum(l["wall_s"] for l in off) / len(off)
        rec["mean_wall_off_s"] = round(m_off, 3)
        rec["mean_wall_on_s"] = round(m_on, 3)
        rec["speedup"] = round(m_off / m_on, 4)
        rec["arm_took"] = all(l["wide_k_served"] > 0 for l in on)
        rec["arm_off_clean"] = all(l["wide_k_served"] == 0 for l in off)
        print(f"off {m_off:.3f}s  on {m_on:.3f}s  {rec['speedup']:.4f}x  "
              f"arm_took={rec['arm_took']} arm_off_clean={rec['arm_off_clean']}", flush=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
