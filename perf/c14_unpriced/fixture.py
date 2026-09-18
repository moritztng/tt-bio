#!/usr/bin/env python3
"""What the fold ACTUALLY issues, counted inside a live 512 aa fold, beside its own wall time.

`c14-matmul-ceiling` found two defects in the census this row reconciles against, and both are
properties of the census's CAPTURE rather than of the fold:

  1. the capture was taken at MSA depth 1024; the 512 aa cell pads 35 rows to a ladder entry
  2. a row-blocked op's census key records the BLOCK, not the op -- pair keys say b=16 where
     main issues b=47 at identical FLOPs

Neither is answerable from the capture, because the capture is the thing under suspicion. Both
are answerable by counting the fold's own calls, which needs no Tracy build and no profiler:
every top-level `ttnn.*` the fold calls is wrapped here and its operand and result shapes are
recorded. The overhead is host-side python, so the COUNTED fold is not a wall-clock read and
the TIMED fold runs with the wrappers off, in the same process and the same session.

Output is an executed launch-key census in the same key format as `perf/roof_launch`, so it
joins to `keys_512.json` directly and the two can be differenced key by key.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "perf/c10_bare_baseline"), str(ROOT),
                str(ROOT / "scripts/gpu_vs_tt"), str(ROOT / "perf/other512")]

import control                                                                # noqa: E402
from force_aiclk import FORCE_AICLK, smc                                       # noqa: E402

TARGET_MHZ = 1350
# the ops the launch-key census keys on, so the executed census is comparable key for key
WRAP = ["linear", "matmul", "multiply_", "add_", "multiply", "add", "layer_norm", "reshape",
        "to_memory_config", "slice", "permute", "transpose", "to_layout", "concat", "pad",
        "softmax", "chunk", "cos", "generic_op", "deallocate", "squeeze", "unsqueeze"]
WRAP_SUB = [("experimental", "nlp_create_qkv_heads"), ("experimental", "nlp_concat_heads"),
            ("transformer", "scaled_dot_product_attention")]


def _shape(x):
    try:
        return tuple(int(d) for d in x.shape)
    except Exception:                                                          # noqa: BLE001
        return None


def install(ttnn, counts, enabled):
    """Wrap every census op so a live fold reports its own (name, in shapes, out shape).

    `enabled` is a one-element list read on every call, so the timed fold can switch the
    recording off without unwrapping and changing the code path between the two folds.
    """
    originals = []

    def wrap(holder, name, label):
        fn = getattr(holder, name)

        def counted(*args, **kw):
            if not enabled[0]:
                return fn(*args, **kw)
            ins = tuple(s for s in (_shape(a) for a in args) if s)
            out = fn(*args, **kw)
            o = _shape(out)
            if o is None and isinstance(out, (list, tuple)):
                o = _shape(out[0]) if out else None
            key = "%s|out=%s|in=%s" % (
                label,
                "x".join(str(d) for d in o) if o else "",
                ",".join("x".join(str(d) for d in s) for s in ins))
            counts[key] += 1
            return out
        setattr(holder, name, counted)
        originals.append((holder, name, fn))

    for n in WRAP:
        if hasattr(ttnn, n):
            wrap(ttnn, n, "ttnn." + n)
    for sub, n in WRAP_SUB:
        h = getattr(ttnn, sub, None)
        if h is not None and hasattr(h, n):
            wrap(h, n, "ttnn.%s.%s" % (sub, n))
    return originals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=False)

    R = {"pid": os.getpid(), "host": socket.gethostname(), "node": a.node, "size": a.size,
         "started_utc_ns": time.time_ns(),
         "git_rev": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            text=True).strip(),
         "protocol": ("one process, one device context, one session: fold 1 TIMED with the "
                      "wrappers inert, fold 2 COUNTED with them recording. The counted fold is "
                      "NOT a wall-clock read."),
         "folds": [], "errors": [], "completed": False}
    save = (lambda: control.write_json(out / "fixture.json", R))
    save()
    if R["host"] != "tt-quietbox2":
        raise RuntimeError("wrong host")

    import ttnn                                                               # noqa: PLC0415
    import tt_bio.tenstorrent as T                                            # noqa: PLC0415
    import tt_baseline as B                                                   # noqa: PLC0415
    from fold_ab_multi import patch_boltz2_cfg                                # noqa: PLC0415
    R["ttnn_path"], R["tt_bio_path"] = ttnn.__file__, T.__file__

    B.RECYCLING_STEPS = 3
    B.SAMPLING_STEPS = 200
    B.DIFFUSION_SAMPLES = 1
    B.SEED = 0
    patch_boltz2_cfg()
    fixture = ROOT / ("perf/size512/fixtures/cdk2x2_%d" % a.size)
    target, msa = fixture.with_suffix(".yaml"), fixture.with_suffix(".a3m")
    _f, meta, state = B.build_fold("boltz2", out / "msa", target, msa, instrument=False,
                                   hoist=False, fast=False, trace=False, recycling_steps=3)
    dev = T.get_device()

    # THE FIXTURE FACT the census's capture got wrong, read off the fold's own metadata
    from tt_bio.tenstorrent import msa_pad_amount                             # noqa: PLC0415
    n_msa = meta["n_msa"]
    R["msa"] = {"rows_in_a3m": n_msa, "pad_amount": int(msa_pad_amount(n_msa)),
                "executed_depth": int(n_msa + msa_pad_amount(n_msa)),
                "census_capture_depth": 1024,
                "capture_signature": "MSALayer|1x512x512x128,1x1024x512x64"}
    R["msa"]["capture_over_executed_x"] = (R["msa"]["census_capture_depth"]
                                           / R["msa"]["executed_depth"])
    R["model_predict_args"] = dict(state.model.predict_args)
    save()

    counts = defaultdict(int)
    enabled = [False]
    install(ttnn, counts, enabled)

    fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    sampler = None
    try:
        R["force_response"] = list(smc(fd, FORCE_AICLK, TARGET_MHZ))
        if R["force_response"][0] != 0:
            raise RuntimeError("FORCE_AICLK failed: %s" % R["force_response"])
        time.sleep(0.15)
        sampler = subprocess.Popen(
            [sys.executable, str(ROOT / "perf/c10_bare_baseline/control.py"),
             str(out / "clock.jsonl"), str(os.getpid())],
            stdin=subprocess.PIPE, stdout=(out / "sampler.log").open("w"),
            stderr=subprocess.STDOUT)
        time.sleep(0.4)

        for label, record in (("warm", False), ("timed", False), ("counted", True)):
            enabled[0] = record
            counts.clear()
            ttnn.synchronize_device(dev)
            s_ns = time.monotonic_ns()
            state.predict_one(target, meta["job_cfg"])
            ttnn.synchronize_device(dev)
            e_ns = time.monotonic_ns()
            row = {"label": label, "recording": record,
                   "start_monotonic_ns": s_ns, "end_monotonic_ns": e_ns,
                   "elapsed_s": (e_ns - s_ns) / 1e9,
                   "n_calls": sum(counts.values()) if record else None,
                   "n_keys": len(counts) if record else None}
            if record:
                (out / "executed_keys.json").write_text(
                    json.dumps(dict(sorted(counts.items(), key=lambda kv: -kv[1])), indent=1))
            R["folds"].append(row)
            save()
            print(json.dumps({k: row[k] for k in ("label", "elapsed_s", "n_calls", "n_keys")}),
                  flush=True)
        R["completed"] = True
    finally:
        enabled[0] = False
        try:
            R["release_response"] = list(smc(fd, FORCE_AICLK, 0))
        except Exception as e:                                                 # noqa: BLE001
            R["errors"].append("clock release failed: %r" % (e,))
        os.close(fd)
        if sampler is not None:
            try:
                sampler.stdin.write(b"stop\n")
                sampler.stdin.flush()
                sampler.stdin.close()
                sampler.wait(timeout=10)
            except Exception:                                                  # noqa: BLE001
                sampler.kill()
        R["ended_utc_ns"] = time.time_ns()
        save()

    samples = [json.loads(ln) for ln in (out / "clock.jsonl").read_text().splitlines() if ln]
    mhz = [s["MHz"] for s in samples if "MHz" in s]
    R["clock_samples"] = len(samples)
    R["clock_min_MHz"], R["clock_max_MHz"] = (min(mhz), max(mhz)) if mhz else (None, None)
    for row in R["folds"]:
        row["clock"] = control.coverage(samples, row)
    save()
    print("clock %s-%s MHz over %d samples"
          % (R["clock_min_MHz"], R["clock_max_MHz"], len(samples)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
