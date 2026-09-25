#!/usr/bin/env python3
"""Every parameter gradient of one OF3 trunk backward, digested, so two trees can be compared.

The op-level A/Bs (`qkv_heads_ab.py`) show each rewritten vjp is bit-identical to a float64
host reference. This is the same question asked of the whole backward: run it once per tree and
compare per-parameter sha256. `--taped-from REV` loads `tt_bio/taped_ttnn.py` as it was at REV
in place of the working tree's, which is the only file the fix changes, so pre and post run
from one checkout with everything else held fixed.

    grad_digest.py --tokens 256 --out out/grads_256_post.json
    grad_digest.py --tokens 256 --taped-from 9de66106e --out out/grads_256_pre.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))


def _load_taped_from(rev):
    src = subprocess.check_output(["git", "-C", str(REPO), "show", f"{rev}:tt_bio/taped_ttnn.py"])
    path = Path(tempfile.mkdtemp()) / "taped_ttnn.py"
    path.write_bytes(src)
    import tt_bio
    spec = importlib.util.spec_from_file_location("tt_bio.taped_ttnn", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tt_bio.taped_ttnn"] = mod
    spec.loader.exec_module(mod)
    tt_bio.taped_ttnn = mod
    return hashlib.sha256(src).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--dead-values", choices=("on", "off"), default="on")
    ap.add_argument("--taped-from", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    if a.taped_from:
        out["env"]["taped_from"] = a.taped_from
        out["env"]["taped_sha256"] = _load_taped_from(a.taped_from)
    else:
        out["env"]["taped_sha256"] = hashlib.sha256(
            (REPO / "tt_bio" / "taped_ttnn.py").read_bytes()).hexdigest()

    from perf.of3t_perf import step as S
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn as TT
    from tt_bio.tenstorrent import get_device
    assert TT.__file__ == sys.modules["tt_bio.taped_ttnn"].__file__
    out["env"]["taped_file"] = TT.__file__
    ag.DROP_DEAD_VALUES = a.dead_values == "on"

    try:
        held, _meta = S.capture(a.tokens, out)
        trunk = held["trunk"][0]
        dev = get_device()
        params = S.declare_weights(trunk, out)
        snap_args, snap_kwargs = held["trunk_snap"]
        args_ = S._rehydrate(snap_args, dev)
        kwargs_ = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items() if k != "progress_fn"}
        trunk.num_cycles = 1
        with ag.tape():
            _s, z = trunk(*args_, **kwargs_)
        zr = z.value
        seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                               layout=ttnn.TILE_LAYOUT, device=dev, dtype=zr.dtype)
        with TT.recompute_scope():
            ag.backward([z], [seed])
        ttnn.synchronize_device(dev)
        grads, whole = {}, hashlib.sha256()
        for name in sorted(params):
            g = getattr(params[name], "grad", None)
            if g is None:
                continue
            g = ttnn.to_torch(getattr(g, "value", g)).float().contiguous()
            h = hashlib.sha256(g.numpy().tobytes()).hexdigest()
            whole.update(name.encode() + h.encode())
            grads[name] = {"sha256": h[:16], "l2": float(g.double().norm()),
                           "shape": list(g.shape)}
        out["grads"] = grads
        out["params_with_grad"] = len(grads)
        out["digest"] = whole.hexdigest()[:16]
        ag.release_pins()
    except Exception:                                                      # noqa: BLE001
        out["error"] = traceback.format_exc()[-4000:]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("tokens %d  taped %s  grads %s  digest %s" % (
        a.tokens, out["env"]["taped_sha256"][:12], out.get("params_with_grad"),
        out.get("digest")), flush=True)
    if out.get("error"):
        print("ERROR:", out["error"][-1500:], flush=True)
    return 0 if not out.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
