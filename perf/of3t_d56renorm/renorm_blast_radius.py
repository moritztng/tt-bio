#!/usr/bin/env python3
"""D56: what `TT_BIO_SOFTMAX_BW_RENORM=1` does to a shipped fold. Nothing, and provably so.

Three arms per model, one card, one seed, one fixture:

  off      the flag off
  on       the flag on -- the configuration the shipped default becomes
  f64      `TT_BIO_HOST_F64_SOFTMAX_AB=all`, and this arm is not about the renorm at all

`on` must be BYTE-IDENTICAL to `off`. That is the whole claim, and on its own it is worth
nothing: a digest that could not see the softmax would report byte-identity whatever the flag
did. So `f64` is the sensitivity control. It changes the softmax FORWARD at every site, the
digest has to move, and only once it has does the identity above mean the flag was inert
rather than the instrument blind.

The counters close the other half. `renorm_stats` must read 0/0 in every inference arm -- not
just `applied` 0, because `declined` counts the branch being EVALUATED, so 0/0 says the
backward closure never ran at all rather than that it ran and chose the other side. That zero
is made informative by `renorm_tape_control.py`, which drives the same counters non-zero.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

PY = "/home/ttuser/tt-bio-dev/env/bin/python"
FLAGS = ("TT_BIO_SOFTMAX_BW_RENORM", "TT_BIO_HOST_F64_SOFTMAX_AB")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fold(tree, model, fixture, out, env_extra, card, holder):
    statsdir = out.parent / (out.name + ".stats")
    env = dict(os.environ)
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": holder, "PYTHONPATH": str(tree),
                "TT_BIO_RENORM_STATS_DIR": str(statsdir)})
    for k in FLAGS:
        env.pop(k, None)
    env.update(env_extra)
    cmd = [PY, "-m", "tt_bio.main", "predict", str(fixture), "--model", model,
           "--single_sequence", "--sampling_steps", "6", "--diffusion_samples", "1",
           "--seed", "0", "--out_dir", str(out)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True)
    # Summed over every process that imported the module, parent and spawned workers alike.
    procs = [json.load(open(f)) for f in sorted(statsdir.glob("*.json"))] \
        if statsdir.is_dir() else []
    stats = {"applied": sum(p["applied"] for p in procs),
             "declined": sum(p["declined"] for p in procs)}
    cifs = sorted(out.rglob("*.cif"))
    rep = {"rc": r.returncode, "wall_seconds": round(time.time() - t0, 1),
           "cifs": {c.name: digest(c) for c in cifs},
           "renorm_stats": stats, "processes_reporting": len(procs),
           "module_read": any(p["flag"] for p in procs) if procs else None,
           "env_flag": env.get("TT_BIO_SOFTMAX_BW_RENORM"),
           "f64_flag": env.get("TT_BIO_HOST_F64_SOFTMAX_AB")}
    if r.returncode:
        rep["tail"] = (r.stdout + r.stderr)[-3000:]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="openfold3,protenix-v2,opendde")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--card", default="3")
    ap.add_argument("--holder", default="worker:of3t-d56-renorm")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    wd = pathlib.Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    # `default` sets nothing at all, so it is the configuration a user gets. It is the arm the
    # ship decision is actually about; `on` is the same thing forced, and the two being equal
    # is the check that the default is what the module reads.
    arms = {"off": {"TT_BIO_SOFTMAX_BW_RENORM": "0"},
            "on": {"TT_BIO_SOFTMAX_BW_RENORM": "1"},
            "default": {},
            "f64": {"TT_BIO_HOST_F64_SOFTMAX_AB": "all"}}

    res, fail = {}, []
    for model in a.models.split(","):
        res[model] = {}
        for arm, env in arms.items():
            out = wd / ("out_%s_%s" % (model, arm))
            r = fold(pathlib.Path(a.tree), model, pathlib.Path(a.fixture), out, env,
                     a.card, a.holder)
            res[model][arm] = r
            print("[%s/%s] rc=%s %ss stats=%s cifs=%s" %
                  (model, arm, r.get("rc"), r.get("wall_seconds"),
                   r.get("renorm_stats"), list(r.get("cifs", {}).values())), flush=True)
            if r.get("rc"):
                print(r.get("tail", ""), flush=True)
                fail.append("%s/%s exited %s" % (model, arm, r.get("rc")))

        m = res[model]
        if not m["off"].get("cifs"):
            fail.append("%s: the off arm produced no cif, so nothing was compared" % model)
        elif m["on"]["cifs"] != m["off"]["cifs"]:
            fail.append("%s: the renorm flag MOVED the fold" % model)
        if m["f64"].get("cifs") == m["off"].get("cifs"):
            fail.append("%s: the f64 control did not move the digest, so the byte-identity "
                        "above is uninformative" % model)
        for arm in arms:
            st = m[arm].get("renorm_stats")
            if st is None:
                fail.append("%s/%s: no counter was read" % (model, arm))
            elif st.get("applied") or st.get("declined"):
                fail.append("%s/%s: a fold reached the renorm branch -- %s" % (model, arm, st))
        if m["on"].get("module_read") is not True:
            fail.append("%s: no process in the on arm read the flag as true -- the arm ran "
                        "the shipped configuration under another name" % model)
        if m["default"]["cifs"] != m["off"]["cifs"]:
            fail.append("%s: the SHIPPED default moves the fold against the flag forced off"
                        % model)
        if m["default"].get("module_read") is not True:
            fail.append("%s: the default arm did not read the flag as true -- the shipped "
                        "default is not the one this run is scoring" % model)
        if not m["off"].get("processes_reporting"):
            fail.append("%s: no process reported a counter at all, so the zeros above are "
                        "the dump not firing rather than the branch not running" % model)

    rep = {"fixture": a.fixture, "tree": a.tree, "card": a.card, "arms": res,
           "failures": fail, "verdict": "PASS" if not fail else "FAIL"}
    print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=1), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("wrote " + a.out)
    return 0 if not fail else 3


if __name__ == "__main__":
    sys.exit(main())
