#!/usr/bin/env python3
"""Which of the campaign's bfp8 gates does region T actually hand a narrowed operand to?

The ledger lists eight of our own bf16 refusals on the pair hot path and assigns them to three
rows. `TT_BIO_TRIATT_B8` is the only region that exists in code, so before any of those rows is
given a verdict it has to be said, from execution rather than from reading, WHICH of the gates
region T already drives and with what operand dtypes. Two of them (`_pair_proj_program_config`,
`_qkv_l1_config`) `return None` silently, so no served/declined counter can see them fire.

Every entry point is wrapped here, in the harness, not in the model: this row edits no production
code. The wrapper records (call site, operand dtypes, destination dtype, served/declined) and
calls straight through, so the fold it censuses is the fold the accuracy arms measured.

    operand_census.py --out <json> --size 298 --arms base,T
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402

ARMS = {"base": {}, "T": {"_TRIATT_B8": True}, "Tbias": {"_TRIATT_BIAS_B8": True}}


def dt(x):
    try:
        return str(x.dtype).rsplit(".", 1)[-1]
    except AttributeError:
        return str(x).rsplit(".", 1)[-1] if x is not None else "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", default="298")
    ap.add_argument("--arms", default="base,T")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    arms = args.arms.split(",")

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_qkv as QKV
    import tt_bio.mm_dualnoc as DN
    import tt_bio.mm_generic as G
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)

    tally: dict = collections.defaultdict(collections.Counter)

    def wrap_qkv(mod, name, ndt):
        """`ndt` names the positional args that are operands / the destination format."""
        fn = getattr(mod, name)

        def w(*a, **kw):
            r = fn(*a, **kw)
            ops = tuple(dt(a[i]) if i < len(a) else "?" for i in ndt)
            served = r is not None and r != (None, None) and r != (None, None, None)
            tally[f"{mod.__name__}.{name}"][f"{'/'.join(ops)} -> {'served' if served else 'DECLINED'}"] += 1
            return r

        setattr(mod, name, w)

    # x, w, dtype(destination) positions per signature, read off the definitions at 42eebf800
    for name, pos in (("qkv_heads", (0, 1, 5)), ("qkvg_heads", (0, 1, 5)),
                      ("qkvgb_heads", (0, 1, 5)), ("gate_proj", (0, 1, 5))):
        if hasattr(QKV, name):
            wrap_qkv(QKV, name, pos)
    if hasattr(DN, "in_proj"):
        wrap_qkv(DN, "in_proj", (0, 1, 3))

    sdpa = T._tri_att_sdpa_inner

    def sdpa_w(q, k, v, bias, *a, **kw):
        tally["tenstorrent._tri_att_sdpa_inner"][
            f"q={dt(q)} k={dt(k)} v={dt(v)} bias={dt(bias)}"] += 1
        return sdpa(q, k, v, bias, *a, **kw)

    T._tri_att_sdpa_inner = sdpa_w

    # The two silent gates. Both `return None` onto a default path, so a count of Nones IS the
    # signal; there is no counter in the tree that shows them.
    for name in ("_pair_proj_program_config", "_qkv_l1_config", "_pair_proj_minimal_matmul"):
        if not hasattr(T, name):
            continue
        fn = getattr(T, name)

        def mk(fn, name):
            def w(*a, **kw):
                r = fn(*a, **kw)
                tally[f"tenstorrent.{name}"]["None (silent fallback)" if r is None else "config"] += 1
                return r
            return w
        setattr(T, name, mk(fn, name))

    fdo = G.fast_dtypes_ok

    def fdo_w(*dtypes):
        r = fdo(*dtypes)
        tally["mm_generic.fast_dtypes_ok"][f"{'/'.join(dt(d) for d in dtypes)} -> {r}"] += 1
        return r

    G.fast_dtypes_ok = fdo_w
    QKV.G.fast_dtypes_ok = fdo_w
    DN.G.fast_dtypes_ok = fdo_w

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="b8cens-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    name = f"cdk2x2_{args.size}"
    AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("bfp8-census", cfg)

    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "size": args.size, "seed": args.seed,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "arms": {}}
    defaults = {n: getattr(T, n) for a in arms for n in ARMS[a]}

    for arm in arms:
        for n, v in defaults.items():
            setattr(T, n, ARMS[arm].get(n, v))
        tally.clear()
        cfg["seed"] = args.seed
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        state.predict_one(AB.FIX / f"{name}.yaml", cfg)
        ttnn.synchronize_device(dev)
        out["arms"][arm] = {k: dict(v) for k, v in sorted(tally.items())}
        print(f"\n===== arm {arm} =====")
        for site, counts in sorted(out["arms"][arm].items()):
            print(f"  {site}")
            for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"      {n:6d}  {k}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
