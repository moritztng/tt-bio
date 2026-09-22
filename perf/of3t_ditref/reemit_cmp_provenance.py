#!/usr/bin/env python3
"""AMENDMENT 1: give of3t-ditcot's twelve CMP artifacts the provenance they should have carried.

`assert_digest_claims_name_their_card.py` refuses a `bit_identical` claim that does not name its
host, because card 0 is a different card on pc, qb1 and qb2 and pc's card 0 must not host
hash-equality gating at all. The twelve carry comparison fields only.

This is a re-emit, not a re-measurement, and it PROVES that rather than asserting it: every file
is recomputed from the very `.pt` dumps its own `a`/`b` fields name, and the recomputed comparison
must equal the stored one field for field, floats included, before anything is written. If one
number moves, nothing is written and the script fails -- because a moved number would mean this
is not a re-emit.

The host and card are read off the run scripts that produced the dumps, not off another machine's
launcher log: `perf/of3t_ditcot/run_kcfg.sh` and `run_kcfg_trunk.sh` both pin
`TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0` literally, and `opiso.py` calls
`tt_bio.tenstorrent.get_device()` under the same row's card-0 lease grant, which tt-bio refuses to
widen and refuses to leave unpinned. The host is this one, qb2, which is where the dumps still sit.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

import torch

DITCOT = pathlib.Path("perf/of3t_ditcot")

HOST = "qb2"
CARD = 0
CARD_EVIDENCE = {
    "run_kcfg.sh": "TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0, literal in the script",
    "run_kcfg_trunk.sh": "TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0, literal in the script",
    "opiso.py": "tt_bio.tenstorrent.get_device() under of3t-ditcot's card-0 lease grant; tt-bio "
                "refuses a card outside the grant and refuses an unpinned open",
}
AICLK_NOTE = ("of3t-ditcot captured no AICLK on these arms, so none is recorded rather than "
              "reconstructed. These twelve are bit-equality comparisons of dumps already on "
              "disk, not timings: no clock-dependent number is claimed here. The arms' own "
              "A/A (CMP_AA, CMP_t1_AA, CMP_t3_AA, CMP_trunk_AA) is the hardware-health "
              "control -- a degraded card breaks bit-identity, it does not manufacture it.")

# exactly the fields cmp_arms.py writes; every one must survive the recompute unchanged
CMP_FIELDS = ("a", "b", "compared", "bit_identical", "moved", "bit_identical_frac", "max_rel",
              "top_moved")


def load_grads(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d.get("grads", d) if isinstance(d, dict) else d


def recompute(a_path, b_path):
    """cmp_arms.py's body, unchanged, so the recompute is the same function."""
    A, B = load_grads(a_path), load_grads(b_path)
    keys = sorted(set(A) & set(B))
    same = 0
    moved = []
    for k in keys:
        x, y = A[k], B[k]
        if not (torch.is_tensor(x) and torch.is_tensor(y)) or x.shape != y.shape:
            continue
        if torch.equal(x, y):
            same += 1
            continue
        xf, yf = x.double().reshape(-1), y.double().reshape(-1)
        nx = float(xf.norm())
        moved.append({"tensor": k, "rel": float((yf - xf).norm() / nx) if nx else float("inf"),
                      "max_abs": float((yf - xf).abs().max())})
    moved.sort(key=lambda r: -r["rel"])
    return {"a": a_path, "b": b_path, "compared": len(keys), "bit_identical": same,
            "moved": len(moved), "bit_identical_frac": same / len(keys) if keys else None,
            "max_rel": moved[0]["rel"] if moved else 0.0,
            "top_moved": moved[:15]}


def canon(v):
    """Exact comparison, floats included: json with repr-level float fidelity."""
    return json.dumps(v, sort_keys=True, default=repr)


def aiclk_now():
    try:
        out = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"], capture_output=True,
                             text=True, timeout=60).stdout
        return out
    except Exception:
        return None


def main():
    files = sorted(glob.glob(str(DITCOT / "CMP_*.json")))
    if len(files) != 12:
        print("STOP expected 12 CMP artifacts, found %d" % len(files))
        return 2

    load0 = os.getloadavg()
    t0 = time.time()
    checked, drift = [], []
    for f in files:
        old = json.loads(pathlib.Path(f).read_text())
        for pth in (old["a"], old["b"]):
            if not os.path.exists(pth):
                print("STOP %s: input %s is gone, so a re-emit cannot be proven a re-emit"
                      % (os.path.basename(f), pth))
                return 2
        new = recompute(old["a"], old["b"])
        bad = [k for k in CMP_FIELDS if canon(old.get(k)) != canon(new.get(k))]
        if bad:
            drift.append((f, bad, {k: (old.get(k), new.get(k)) for k in bad
                                   if k != "top_moved"}))
        checked.append((f, old, new))
        print("  %-28s compared=%-5d bit_identical=%-5d moved=%-5d  %s"
              % (os.path.basename(f), new["compared"], new["bit_identical"], new["moved"],
                 "DRIFT " + ",".join(bad) if bad else "unchanged"))

    if drift:
        print("\nSTOP %d artifact(s) did not reproduce. A moved number means this is not a "
              "re-emit; nothing was written." % len(drift))
        for f, bad, vals in drift:
            print("  %s %s %s" % (f, bad, vals))
        return 1

    load1 = os.getloadavg()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for f, old, _new in checked:
        out = dict(old)
        out["host"] = HOST
        out["card"] = CARD
        out["aiclk_mhz_during"] = None
        out["reemit"] = {
            "what": "provenance added; comparison fields recomputed from the same dumps and "
                    "unchanged field for field",
            "why": "D155 / AMENDMENT 1: a bit_identical claim that does not name its host cannot "
                   "be attributed to healthy hardware",
            "by": "perf/of3t_ditref/reemit_cmp_provenance.py",
            "at": stamp,
            "reemit_ran_on": socket.gethostname(),
            "fields_reproduced": list(CMP_FIELDS),
            "numbers_unchanged": True,
            "card_evidence": CARD_EVIDENCE,
            "aiclk_note": AICLK_NOTE,
            "inputs": {k: {"path": old[k], "bytes": os.path.getsize(old[k]),
                           "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime(os.path.getmtime(old[k])))}
                       for k in ("a", "b")},
            "loadavg_start": list(load0),
            "loadavg_end": list(load1),
            "recompute_seconds": round(time.time() - t0, 1),
        }
        pathlib.Path(f).write_text(json.dumps(out, indent=1))
    print("\nok    12 of 12 reproduced exactly and were re-emitted with host=%s card=%s "
          "(%.0f s, loadavg %.2f -> %.2f)" % (HOST, CARD, time.time() - t0, load0[0], load1[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
