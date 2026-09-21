#!/usr/bin/env python3
"""D56: what `TT_BIO_SOFTMAX_BW_RENORM=1` does to a shipped fold. Nothing, and provably so.

Three arms per model, one card, one seed, one fixture:

  off      the flag off
  on       the flag on -- the configuration the shipped default becomes
  accsm    `TT_BIO_ACCURATE_SOFTMAX_AB=all`, and this arm is not about the renorm at all
  f64      `TT_BIO_HOST_F64_SOFTMAX_AB=all`, expected INERT here, see below

`on` must be BYTE-IDENTICAL to `off`. That is the whole claim, and on its own it is worth
nothing: a digest that could not see the softmax would report byte-identity whatever the flag
did. So `accsm` is the sensitivity control. It swaps in the 5-op accurate softmax at every
site, which changes the FORWARD, the digest has to move, and only once it has does the
identity above mean the flag was inert rather than the instrument blind.

`f64` used to be that control and is kept as a RECORD rather than dropped. On the tree this
row started from it moved the OpenFold3 digest to 768b47cf...; on the d116 tree it does not
move it at all, because `host_f64_softmax` now raises unless a tape is installed. That is
D137 -- the host float64 softmax reaching an inference tensor -- closed, and the arm is the
reading that says so. It is asserted INERT, not ignored: if it starts moving a fold again,
D137 has reopened.

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
PROBE_HOME = "/home/ttuser/d56probe"
FLAGS = ("TT_BIO_SOFTMAX_BW_RENORM", "TT_BIO_HOST_F64_SOFTMAX_AB",
         "TT_BIO_ACCURATE_SOFTMAX_AB")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fold(tree, model, fixture, out, env_extra, card, holder):
    statsdir = out.parent / (out.name + ".stats")
    probedir = out.parent / (out.name + ".probe")
    # `sitecustomize` is imported by EVERY interpreter, including the workers `predict`
    # spawns, which is the only hook that reaches them: an atexit registered inside
    # `tt_bio.autograd` cannot fire in a process that never imports `tt_bio.autograd`, and on
    # an inference fold that is every process. PROBE_HOME is outside the repo on purpose.
    env = dict(os.environ)
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": holder,
                "PYTHONPATH": PROBE_HOME + ":" + str(tree),
                "TT_BIO_D56_PROBE_DIR": str(probedir),
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
    wit = [json.load(open(f)) for f in sorted(probedir.glob("*.json"))] \
        if probedir.is_dir() else []
    imported = [w for w in wit if w.get("imported")]
    cifs = sorted(out.rglob("*.cif"))
    rep = {"rc": r.returncode, "wall_seconds": round(time.time() - t0, 1),
           "cifs": {c.name: digest(c) for c in cifs},
           "renorm_stats": stats, "processes_reporting": len(procs),
           "module_read": any(p["flag"] for p in procs) if procs else None,
           "processes_witnessed": len(wit),
           "processes_that_imported_autograd": len(imported),
           "autograd_importers": [w["argv"] for w in imported][:4],
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
    ap.add_argument("--arms", default="", help="comma list, default all")
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
            "accsm": {"TT_BIO_ACCURATE_SOFTMAX_AB": "all"},
            "f64": {"TT_BIO_HOST_F64_SOFTMAX_AB": "all"}}
    if a.arms:
        keep = set(a.arms.split(","))
        arms = {k: v for k, v in arms.items() if k in keep}

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
        if "off" not in m:
            continue
        if not m["off"].get("cifs"):
            fail.append("%s: the off arm produced no cif, so nothing was compared" % model)
        elif "on" in m and m["on"]["cifs"] != m["off"]["cifs"]:
            fail.append("%s: the renorm flag MOVED the fold" % model)
        if "accsm" in m and m["accsm"].get("cifs") == m["off"].get("cifs"):
            fail.append("%s: the accurate-softmax control did not move the digest, so the "
                        "byte-identity above is uninformative" % model)
        if "f64" in m and m["f64"].get("cifs") != m["off"].get("cifs"):
            fail.append("%s: the host float64 softmax MOVED an inference fold -- D137 has "
                        "reopened, that path is supposed to refuse without a tape" % model)
        for arm in arms:
            a_ = m[arm]
            if not a_.get("processes_witnessed"):
                fail.append("%s/%s: not one process wrote a witness, so everything below is "
                            "the probe not loading rather than a reading" % (model, arm))
            # The real claim, and it is stronger than a zero counter: no process in a fold
            # even IMPORTS the module that holds the flag and the branch.
            if a_.get("processes_that_imported_autograd"):
                fail.append("%s/%s: %d fold process(es) imported tt_bio.autograd -- %s"
                            % (model, arm, a_["processes_that_imported_autograd"],
                               a_.get("autograd_importers")))
            st = a_.get("renorm_stats") or {}
            if st.get("applied") or st.get("declined"):
                fail.append("%s/%s: a fold reached the renorm branch -- %s" % (model, arm, st))
        if "default" in m and m["default"]["cifs"] != m["off"]["cifs"]:
            fail.append("%s: the SHIPPED default moves the fold against the flag forced off"
                        % model)

    rep = {"fixture": a.fixture, "tree": a.tree, "card": a.card, "arms": res,
           "failures": fail, "verdict": "PASS" if not fail else "FAIL"}
    print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=1), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("wrote " + a.out)
    return 0 if not fail else 3


if __name__ == "__main__":
    sys.exit(main())
