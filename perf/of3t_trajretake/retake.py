#!/usr/bin/env python3
"""Run `of3t-modeltraj`'s own 20-step model-in-the-loop trajectory on this tree, and stamp it.

This is a DRIVER, not a second instrument. Every number in the artifact is produced by
`perf/of3t_modeltraj/modeltraj.py` -- its partition, its reference side, its scorer, its growth
fit. What this adds is the two things the old artifact could not say about itself:

  * WHICH TREE. `perf/of3t_modeltraj/traj_shipped.json`'s `env` block records exactly one
    variable, `TT_VISIBLE_DEVICES`. No branch, no commit, no host. It was written 2026-09-21
    02:35:02 UTC on `a0851ef8a` and the `_PARAMS` re-key that repairs the defect it reports
    landed twenty minutes later at 02:55:54 UTC on `965c24f52` -- which is discoverable only
    from the git log, never from the artifact. An artifact that cannot name its tree cannot be
    re-checked when the tree moves, and that is how the defect survived 22 hours.
  * THE CLOCK, DURING. Sampled off `tt-smi -s` while the child runs, per device, with a median.
    This row makes no perf claim, so the clock is here for attribution and not for a ratio; it
    is recorded because D155's rule -- a digest claim names its hardware -- has no reason to
    stop at digests.

Both are written INTO the artifact the harness produced, so there is one file per arm and it
carries its own provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARNESS = os.path.join(WT, "perf", "of3t_modeltraj", "modeltraj.py")
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
TT_SMI = os.path.expanduser("~/.local/bin/tt-smi")
REFDEPS = "/home/ttuser/of3t_refprec/deps"

# The artifact this row re-takes, cited by path and content so the comparison is anchored.
CITED = {
    "perf/of3t_modeltraj/traj_shipped.json": "the arm this row re-takes",
    "perf/of3t_modeltraj/traj_repin.json": "the D126 repair carried default-off by of3t-modeltraj",
    "perf/of3t_modeltraj/traj_renorm.json": "of3t-modeltraj's D56-on arm",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def git(*a):
    return subprocess.run(("git",) + a, cwd=WT, capture_output=True, text=True).stdout.strip()


def smi():
    """One `tt-smi -s` snapshot, or None. AICLK arrives as a RIGHT-ALIGNED STRING (' 800'),
    so it is int()'d rather than compared, which is the shape of a lesson this fleet has
    already paid for."""
    try:
        r = subprocess.run([TT_SMI, "-s"], capture_output=True, text=True, timeout=120)
        d = json.loads(r.stdout)
    except Exception:
        return None
    out = {"t": time.time(), "aiclk": {}, "pid_on_device": {}}
    for i, c in enumerate(d.get("device_info", [])):
        try:
            out["aiclk"][str(i)] = int(str(c["telemetry"]["aiclk"]).strip())
        except Exception:
            pass
    for p in d.get("processes", []):
        out["pid_on_device"].setdefault(str(p.get("pid")), []).append(p.get("device"))
    return out


class Sampler(threading.Thread):
    def __init__(self, every):
        super().__init__(daemon=True)
        self.every, self.stop, self.rows = every, threading.Event(), []

    def run(self):
        while not self.stop.is_set():
            s = smi()
            if s is not None:
                self.rows.append(s)
            self.stop.wait(self.every)


def clauses(art):
    """The three things the brief asks this artifact to answer, computed from the harness's own
    log and per-step rows rather than read off prose.

    `resolves` and `grad_norm` come from OUR step log and the reference arms (`aa`,
    `theirs-f32-bar`) do not have one -- they are upstream against upstream and there is no
    tape. Those arms report the clause as not applicable rather than as a pass, because a
    field that reads as a pass on an arm it cannot describe is this row's own subject.
    """
    log = art.get("our_step_log") or []
    per = art.get("per_step") or []
    ours = [r for r in log if "tape_resolves_after_step" in r]
    walked = max((r.get("of_walked") or 0) for r in ours) if ours else 0
    res = [r.get("tape_resolves_after_step") for r in ours]
    gn = [r.get("grad_norm") for r in ours if r.get("grad_norm") is not None]
    live = [(r["k"], r["d_theirs_norm"], r["d_ours_norm"]) for r in per
            if (r.get("d_theirs_norm") or 0.0) > 0.0]
    stationary = [k for k, _t, o in live if not (o > 0.0)]
    NA = "not applicable: this arm has no tape of ours (upstream against upstream)"
    out = {
        "moves": {
            "rungs_with_theirs_moving": len(live),
            "rungs_ours_stationary": stationary,
            "clause_met": bool(live) and not stationary,
            "note": "k=1 is excluded by construction: lr(1) is exactly 0 under the AF3 warmup, "
                    "so d_1 is 0 on BOTH sides and PROTOCOL S7a does not read that as a pass"},
    }
    if not ours:
        out["resolves"] = {"applicable": False, "note": NA}
        out["grad_norm"] = {"applicable": False, "note": NA}
        return out
    out["resolves"] = {
        "applicable": True, "steps": len(res), "of_walked": walked,
        "min": min(res), "max": max(res),
        "nonzero_at_every_step": all((x or 0) > 0 for x in res),
        "full_at_every_step": walked > 0 and all(x == walked for x in res),
        "per_step": res}
    out["grad_norm"] = {
        "applicable": True, "min": (min(gn) if gn else None),
        "max": (max(gn) if gn else None),
        "nonzero_at_every_step": len(gn) == len(res) and all(x > 0.0 for x in gn),
        "k1": (gn[0] if gn else None), "k_last": (gn[-1] if gn else None)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--renorm", choices=["1", "0", "unset"], default="1",
                    help="TT_BIO_SOFTMAX_BW_RENORM: asked on, asked OFF, or absent from the "
                         "environment. Absent is NOT off on this tree -- "
                         "`autograd.py:87` is env_flag(..., True) since D56 landed at 1aa7070f5, "
                         "so removing the variable leaves the renorm LIVE. Distinguishing the "
                         "two is the point of the choice.")
    ap.add_argument("--card", default="1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()

    out = os.path.join("perf", "of3t_trajretake", f"traj_retake_{a.arm}{a.tag}.json")
    env = dict(os.environ)
    env["TT_VISIBLE_DEVICES"] = a.card
    env["TT_BIO_LEASE_CARDS"] = a.card
    env["TT_BIO_LEASE_HOLDER"] = "worker:of3t-trajretake"
    if a.renorm == "unset":
        env.pop("TT_BIO_SOFTMAX_BW_RENORM", None)
    else:
        env["TT_BIO_SOFTMAX_BW_RENORM"] = a.renorm
    # `deps`, NOT `pylibs`. Upstream 0.4.3 needs `ml_collections` and both directories carry it,
    # but `pylibs` also carries an `openfold3` 0.5.0 -- put that on the path and the reference
    # silently becomes the wrong version, which is D149 exactly. `deps` has the dependencies and
    # no openfold3, and the harness's own `sys.path.insert(0, OF3PKG)` stays ahead of it either
    # way. `reference_resolution` in the artifact is what proves which tree actually answered.
    env["PYTHONPATH"] = os.pathsep.join(
        x for x in (REFDEPS, env.get("PYTHONPATH")) if x)
    # COUNT THE REACHES, do not infer them. The D56-on and D56-off arms came back bit-identical
    # at every rung, and "bit-identical" has two readings: the renorm is inert at this boundary,
    # or the helper is never called here at all. `autograd.py:94` dumps
    # SOFTMAX_BW_RENORM_STATS per pid when this directory is set, so the artifact carries the
    # call count and the distinction is measured rather than argued.
    stats = os.path.join("/tmp/of3t/of3t-trajretake/renorm_stats",
                         "%s%s_%s" % (a.arm, a.tag, a.renorm))
    subprocess.run(["rm", "-rf", stats])
    os.makedirs(stats, exist_ok=True)
    env["TT_BIO_RENORM_STATS_DIR"] = stats
    cmd = [PY, HARNESS, "--arm", a.arm, "--steps", str(a.steps), "--out", out]

    # Read the tree BEFORE the child runs. The child WRITES into the tree, so a status taken
    # afterwards lists the artifact in its own dirty list, and a stamp that cannot tell "the tree
    # that produced this" from "the tree after this was produced" is the weakness this row is
    # here to close.
    tree = {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "commit_date": git("log", "-1", "--format=%ad", "--date=iso"),
        "identical_to_origin_wk_of3t": git("rev-parse", "origin/wk/of3t"),
        "worktree": WT,
        "dirty_paths_before_run": sorted(
            l.split(None, 1)[1] for l in git("status", "--porcelain").splitlines()
            if len(l.split(None, 1)) > 1),
    }
    samp = Sampler(a.every)
    t0 = time.time()
    samp.start()
    print("driver: %s\n        renorm asked=%s card=%s" % (" ".join(cmd), a.renorm, a.card),
          flush=True)
    proc = subprocess.Popen(cmd, cwd=WT, env=env)
    rc = proc.wait()
    t1 = time.time()
    samp.stop.set()
    samp.join(timeout=130)
    print("driver: child rc=%d in %.1f s, %d clock samples" % (rc, t1 - t0, len(samp.rows)),
          flush=True)
    if rc != 0:
        return rc

    path = os.path.join(WT, out)
    art = json.load(open(path))

    # which device index the child actually sat on, read off tt-smi's own process table rather
    # than off the env var we set -- a UMD logical id and a /dev/tenstorrent node are not the
    # same number and this fleet has already filed that confusion.
    devs = sorted({d for s in samp.rows for d in s["pid_on_device"].get(str(proc.pid), [])})
    per_dev = {}
    for s in samp.rows:
        for k, v in s["aiclk"].items():
            per_dev.setdefault(k, []).append(v)
    mine = str(devs[0]) if devs else a.card
    my = per_dev.get(mine, [])

    art["tree"] = {
        **tree,
        "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "card_requested": a.card,
        "card_observed_by_smi": devs,
        "python": PY,
        "harness": "perf/of3t_modeltraj/modeltraj.py",
        "harness_sha256": sha256(HARNESS),
        "reference_deps_on_pythonpath": REFDEPS,
        "autograd_rekey_at": "tt_bio/autograd.py value setter, _PARAMS re-keyed on handle replace",
    }
    art["aiclk_during"] = {
        "sampled_by": "tt-smi -s, every %.1f s, DURING the child" % a.every,
        "window_utc": [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t1))],
        "samples": len(my), "device": mine,
        "median_mhz": (statistics.median(my) if my else None),
        "min_mhz": (min(my) if my else None), "max_mhz": (max(my) if my else None),
        "all_devices_median_mhz": {k: statistics.median(v) for k, v in per_dev.items()},
        "series_mhz": my,
        "note": "recorded for attribution. This row's claims are trajectory and parity, not "
                "throughput, so no number here is divided by a clock.",
    }
    live = (art.get("flag_reach") or {}).get("_SOFTMAX_BW_RENORM")
    art["d56"] = {
        "softmax_bw_renorm_asked": (None if a.renorm == "unset" else a.renorm == "1"),
        "softmax_bw_renorm_env": ("absent" if a.renorm == "unset" else a.renorm),
        "package_default": "tt_bio/autograd.py:87 env_flag(\"TT_BIO_SOFTMAX_BW_RENORM\", True) "
                           "since 1aa7070f5 -- absent means LIVE, not off",
        "softmax_bw_renorm_live": (None if live is None else bool(live)),
        "live_read_from": "the LOADED tt_bio.taped_ttnn module's _SOFTMAX_BW_RENORM, not the "
                          "environment -- what was asked for and what the import holds are two "
                          "different facts",
        "agree": (None if live is None or a.renorm == "unset"
                  else bool(live) == (a.renorm == "1")),
    }
    got = {}
    for fn in sorted(os.listdir(stats)) if os.path.isdir(stats) else []:
        try:
            got[fn] = json.load(open(os.path.join(stats, fn)))
        except Exception:
            pass
    art["d56"]["renorm_stats_per_pid"] = got
    art["d56"]["softmax_bw_inner_calls"] = sum((v.get("applied", 0) + v.get("declined", 0))
                                               for v in got.values())
    art["d56"]["applied"] = sum(v.get("applied", 0) for v in got.values())
    art["d56"]["declined"] = sum(v.get("declined", 0) for v in got.values())
    art["clauses"] = clauses(art)
    art["cites"] = {p: {"why": w, "sha256": sha256(os.path.join(WT, p))}
                    for p, w in CITED.items()}
    art["retake_of"] = ("perf/of3t_modeltraj/traj_shipped.json, written 2026-09-21 02:35:02 UTC "
                        "on a0851ef8a, twenty minutes before the _PARAMS re-key landed at "
                        "02:55:54 UTC on 965c24f52. This row re-takes it; it does not edit it.")
    json.dump(art, open(path, "w"), indent=1)

    c = art["clauses"]
    print("\n=== %s ===" % out)
    print(" resolves  %s" % json.dumps(c["resolves"], default=str)[:300])
    print(" grad_norm %s" % json.dumps(c["grad_norm"], default=str)[:300])
    print(" moves     met=%s over %d rungs, stationary at %s"
          % (c["moves"]["clause_met"], c["moves"]["rungs_with_theirs_moving"],
             c["moves"]["rungs_ours_stationary"] or "none"))
    print(" growth    %s" % json.dumps(art["growth_k2_20"]))
    print(" d56       %s" % json.dumps(art["d56"], default=str)[:200])
    print(" aiclk     dev %s median %s MHz over %d samples (min %s, max %s)"
          % (art["aiclk_during"]["device"], art["aiclk_during"]["median_mhz"],
             art["aiclk_during"]["samples"], art["aiclk_during"]["min_mhz"],
             art["aiclk_during"]["max_mhz"]))
    print(" tree      %s @ %s on %s card %s"
          % (art["tree"]["branch"], art["tree"]["commit"][:9], art["tree"]["host"],
             art["tree"]["card_observed_by_smi"] or art["tree"]["card_requested"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
