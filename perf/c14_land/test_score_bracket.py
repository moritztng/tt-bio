#!/usr/bin/env python3
"""Known-answer control for score_bracket.py. CPU only, no device, no network.

A scorer that never returns NULL for a null session cannot be trusted when it returns GO, so
every case below is paired with the case that must come out the other way.
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORER = HERE / "score_bracket.py"


def write_session(d: Path, blocks, size="512", base_s=16.0, delta=0.0, noise=0.0, seed=0,
                  drift=0.0, on_digest_differs=True):
    """Emit the driver's own on-disk layout: <size>_<arm>_<block>_<leg>.json."""
    rng = random.Random(seed)
    for b in range(blocks):
        level = base_s + drift * b          # a monotone neighbour, which the bracket must absorb
        for leg, arm in ((0, "base"), (1, "on"), (2, "base")):
            centre = level + (drift * leg / 3.0)
            if arm == "on":
                centre -= delta
            folds = [{"tag": str(i),
                      "fold_s": round(centre + rng.gauss(0, noise), 4),
                      "clock": {"aiclk_min": 1300, "aiclk_max": 1350, "aiclk_mean": 1340.0,
                                "aiclk_n": 60, "power_w_mean": 75.0},
                      "apb_served_declined": [5064, 0] if arm == "on" else [0, 5064],
                      "plddt": 0.84,
                      "cif_sha256": ("on" if (arm == "on" and on_digest_differs) else "base") * 4,
                      "loadavg1": 5.0} for i in range(5)]
            (d / f"{size}_{arm}_{b}_{leg}.json").write_text(
                json.dumps({"arm": arm, "size": size, "model": "boltz2", "folds": folds}))


def score(**kw):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        write_session(d, **kw)
        out = subprocess.run([sys.executable, str(SCORER), "--dir", str(d)],
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout)


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    return cond


def main():
    ok = True

    # 1. Known answer: a planted +0.30 s lever, well above the noise, must be recovered in
    #    value and called GO.
    r = score(blocks=12, delta=0.30, noise=0.03, seed=1)
    ok &= check("planted +0.30 s recovered", abs(r["mean_delta_s"] - 0.30) < 0.02,
                f"read {r['mean_delta_s']}")
    ok &= check("planted +0.30 s -> GO", r["verdict"] == "GO", f"p={r['perm_p_two_sided']}")

    # 2. NEGATIVE CONTROL, and it is the one that matters: the same session with NO lever.
    #    If this also returns GO the scorer is measuring nothing.
    r = score(blocks=12, delta=0.0, noise=0.15, seed=2)
    ok &= check("no lever -> not GO", r["verdict"] != "GO",
                f"verdict {r['verdict']}, mean {r['mean_delta_s']}")

    # 3. A real but small lever buried in noise 5x its size must NOT be called GO on this n.
    #    Under-powered is a NULL, never a win.
    r = score(blocks=12, delta=0.05, noise=0.25, seed=3)
    ok &= check("0.05 s under 0.25 s noise -> not GO", r["verdict"] != "GO",
                f"verdict {r['verdict']}")

    # 4. Sign: a lever that makes the fold SLOWER must read NEGATIVE, not NULL and not GO.
    r = score(blocks=12, delta=-0.30, noise=0.03, seed=4)
    ok &= check("planted -0.30 s -> NEGATIVE", r["verdict"] == "NEGATIVE",
                f"verdict {r['verdict']}, mean {r['mean_delta_s']}")

    # 5. The bracket's whole purpose: a monotone drifting neighbour with no lever must not
    #    manufacture one. Without bracketing, a base-then-on ordering under drift reads a win.
    r = score(blocks=12, delta=0.0, noise=0.02, drift=0.40, seed=5)
    ok &= check("pure drift, no lever -> not GO", r["verdict"] != "GO",
                f"verdict {r['verdict']}, mean {r['mean_delta_s']}")

    # 6. The permutation floor is real and must be reported, not silently passed.
    r = score(blocks=4, delta=0.30, noise=0.01, seed=6)
    ok &= check("n=4 cannot reach p<0.05", r["verdict"] != "GO",
                f"p_floor={r['perm_p_floor']}, p={r['perm_p_two_sided']}")

    # 7. Firing and digest bookkeeping, so a session that silently ran one arm twice is caught.
    r = score(blocks=6, delta=0.1, noise=0.02, seed=7)
    ok &= check("on arm fires 5064/0", r["firing_on"] == [[5064, 0]], str(r["firing_on"]))
    ok &= check("base arm declines 0/5064", r["firing_base"] == [[0, 5064]], str(r["firing_base"]))
    ok &= check("digest moves with the flag", r["digest_moves_with_flag"] is True)
    r = score(blocks=6, delta=0.1, noise=0.02, seed=8, on_digest_differs=False)
    ok &= check("identical digests detected as not moving",
                r["digest_moves_with_flag"] is False)

    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
