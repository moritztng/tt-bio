#!/usr/bin/env python3
"""Compare two `tapeprof.py --grad-dump` files BIT for BIT over the whole declared set.

A memory fix is a gradient change until proven otherwise, and the way a released node goes wrong
is that a gradient goes SILENTLY ABSENT -- so this reports the None count on both sides as well
as the digests, because two arms that both dropped the same weight would otherwise "agree".

Four of OpenFold3's declared weights are named by `walk_weights` with a Python `id()` in the
path (`diffusion.dm._wc.<id>`), which differs per process. They are compared as a MULTISET of
digests instead of by name, and counted separately so the concession is visible.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ID_NAMED = re.compile(r"\._wc\.\d+$")


def split(path: str):
    d = json.loads(Path(path).read_text())
    stable, unstable = {}, []
    for k, v in d["grads"].items():
        (unstable.append(v) if ID_NAMED.search(k) else stable.__setitem__(k, v))
    return d, stable, unstable


def sig(v):
    if v is None:
        return None
    return (v.get("sha256"), tuple(v.get("shape", ())), v.get("dtype"), v.get("bytes"),
            v.get("error"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("fixed")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dc, c, cu = split(a.control)
    df, f, fu = split(a.fixed)
    res = {"control": a.control, "fixed": a.fixed,
           "control_arm": dc.get("arm"), "fixed_arm": df.get("arm"),
           "tokens": dc.get("tokens"), "declared": [dc["declared"], df["declared"]],
           "with_grad": [dc["with_grad"], df["with_grad"]],
           "key_sets_equal": set(c) == set(f),
           "name_keyed": len(c), "id_named": len(cu)}
    if not res["key_sets_equal"]:
        res["only_control"], res["only_fixed"] = sorted(set(c) - set(f))[:20], sorted(set(f) - set(c))[:20]
    common = sorted(set(c) & set(f))
    res["identical"] = sum(1 for k in common if sig(c[k]) == sig(f[k]))
    res["differing"] = [k for k in common if sig(c[k]) != sig(f[k])][:40]
    res["none_both"] = sum(1 for k in common if c[k] is None and f[k] is None)
    res["none_one_sided"] = [k for k in common if (c[k] is None) != (f[k] is None)][:40]
    res["digest_errors_control"] = sum(1 for v in c.values() if isinstance(v, dict) and "error" in v)
    res["digest_errors_fixed"] = sum(1 for v in f.values() if isinstance(v, dict) and "error" in v)
    res["bytes_hashed_mib"] = round(
        sum(v["bytes"] for v in c.values() if isinstance(v, dict) and "bytes" in v) / 2 ** 20, 1)
    res["id_named_multiset_equal"] = (collections.Counter(map(sig, cu))
                                      == collections.Counter(map(sig, fu)))
    res["BIT_IDENTICAL"] = bool(
        res["key_sets_equal"] and not res["differing"] and not res["none_one_sided"]
        and res["id_named_multiset_equal"]
        and res["digest_errors_control"] == res["digest_errors_fixed"] == 0
        and res["with_grad"][0] == res["with_grad"][1])
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    return 0 if res["BIT_IDENTICAL"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
