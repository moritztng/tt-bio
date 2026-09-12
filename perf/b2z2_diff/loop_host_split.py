#!/usr/bin/env python3
"""Where does the Boltz-2 sampler loop spend the time the device is not working?

`bioir-dispatch` measured the sampler stage at ~9.2 s/fold against a 32.53 ms/step pure replay
floor, and showed ttnn trace closes only 0.15 s of the gap. So the gap is not op dispatch. This
harness brackets the named host regions of one traced sampler loop and prices each one, then
prices the instrument itself with an uninstrumented traced fold in the same process.

Regions (all per fold, 200 steps):
  pnf          AtomDiffusion.preconditioned_network_forward, the whole denoiser call
  stage_in       host tilize of r/times + copy_host_to_device_tensor
  replay         ttnn.execute_trace (non-blocking: the device wait lands in `out`)
  out            ttnn.to_torch of the trace output buffer, i.e. the device wait + untilize
  augment        compute_random_augmentation
  align          weighted_rigid_align (alignment_reverse_diff)
  residual       sample wall minus the above, i.e. the loop body torch math

`out` is device-inclusive by construction; everything else is host-serial time the device spends
idle. The claim this harness can support is the SPLIT, not an absolute fold number.

The thread arms exist because pass 1 found the split does not add up to the work: the same
`weighted_rigid_align` costs 0.312 ms/call on this host in isolation and 5.44 ms inside the fold.
The suspect is torch oversubscribing its pools (8 intra-op, 16 interop by default here) next to
tt-metal's spinning dispatch and completion threads. `--arms` takes `label:threads` pairs; threads
0 means leave the pools alone. A thread count changes the summation order of a multithreaded
reduction, so the arm is NOT bit-exact by construction: every fold records its CIF sha256 and the
arm is scored on whether the digest holds.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 30))

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))   # patch_boltz2_cfg

W = defaultdict(lambda: {"n": 0, "s": 0.0})
ON = {"v": False}


def sha_dir(d):
    return {q.name: hashlib.sha256(q.read_bytes()).hexdigest()[:16]
            for q in sorted(Path(d).glob("*")) if q.is_file()}


def timed(key, fn):
    def wrapper(*a, **kw):
        if not ON["v"]:
            return fn(*a, **kw)
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            w = W[key]
            w["n"] += 1
            w["s"] += time.perf_counter() - t0
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, default=ROOT / "perf" / "b2z2_diff" / "loop_host_split.json")
    ap.add_argument("--arms", default="base:0,t1:1,base:0,t1:1,base:0,t1:1",
                    help="label:threads pairs run in order; threads 0 leaves torch alone")
    ap.add_argument("--instrument-arms", default="base:0,t1:1",
                    help="run each of these once more with the region brackets on")
    ap.add_argument("--traced", type=int, default=1, help="1 = diffusion trace on, 0 = eager")
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.boltz2 as B2
    import tt_baseline as B
    from fold_ab_multi import patch_boltz2_cfg   # build_fold carries no Boltz-2 hyperparameters

    # patch_boltz2_cfg reads B.RECYCLING_STEPS, which tt_baseline does not define (it resolves
    # recycles per model instead). 3 is the cell protocol and what _resolve_recycling_steps
    # returns for boltz2, so the injected conf_kwargs and the fold agree.
    B.RECYCLING_STEPS = 3
    patch_boltz2_cfg()

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold("boltz2", ROOT / f".msa_b2z2_{a.size}", tgt, a3m)
    assert meta["recycling_steps"] == 3, meta["recycling_steps"]
    assert T.trace_region_size() > 0, "device opened without a trace region"
    struct_dir = Path(meta["struct_dir"])

    sms = [m for m in state.model.modules() if hasattr(m, "_diffusion_trace")]
    assert len(sms) == 1, f"expected one AtomDiffusion, found {len(sms)}"
    sm = sms[0]
    adapter = sm.score_model

    sm.sample = timed("sample", sm.sample)
    sm.preconditioned_network_forward = timed("pnf", sm.preconditioned_network_forward)
    adapter.forward_traced = timed("forward_traced", adapter.forward_traced)
    adapter._host_tt = timed("stage_in", adapter._host_tt)
    B2.compute_random_augmentation = timed("augment", B2.compute_random_augmentation)
    B2.weighted_rigid_align = timed("align", B2.weighted_rigid_align)
    ttnn.copy_host_to_device_tensor = timed("stage_in", ttnn.copy_host_to_device_tensor)
    ttnn.execute_trace = timed("replay", ttnn.execute_trace)
    ttnn.to_torch = timed("out", ttnn.to_torch)

    default_threads = (torch.get_num_threads(), torch.get_num_interop_threads())
    res = {"size": a.size, "trace_region": T.trace_region_size(),
           "grid": list(T.COMPUTE_GRID_MAIN), "traced": bool(a.traced),
           "default_threads": list(default_threads), "runs": []}

    def run(label, threads, instrument):
        W.clear()
        ON["v"] = instrument
        sm._diffusion_trace = bool(a.traced)
        # interop threads cannot be changed after the pool starts, so only intra-op moves; that is
        # the pool the small tensor ops in the loop actually use.
        torch.set_num_threads(threads or default_threads[0])
        for q in struct_dir.glob("*"):
            q.unlink()
        fold_s, m = one_fold()
        ON["v"] = False
        rec = {"label": label, "threads": torch.get_num_threads(), "instrumented": instrument,
               "fold_s": round(fold_s, 3), "metrics": {k: v for k, v in m.items()
                                                       if isinstance(v, (int, float, str, bool))},
               "cif": sha_dir(struct_dir),
               "wall": {k: {"n": v["n"], "ms": round(v["s"] * 1e3, 2)}
                        for k, v in sorted(W.items(), key=lambda kv: -kv[1]["s"])},
               "loadavg": open("/proc/loadavg").read().split()[:3]}
        if instrument:
            g = lambda k: W[k]["s"] * 1e3
            rec["split_ms"] = {
                "sample": round(g("sample"), 1),
                "pnf": round(g("pnf"), 1),
                "device_wait_out": round(g("out"), 1),
                "pnf_minus_device": round(g("pnf") - g("out"), 1),
                "stage_in": round(g("stage_in"), 1),
                "replay_issue": round(g("replay"), 1),
                "augment": round(g("augment"), 1),
                "align": round(g("align"), 1),
                "loop_residual": round(g("sample") - g("pnf") - g("augment") - g("align"), 1),
                "host_serial": round(g("sample") - g("out"), 1),
            }
        res["runs"].append(rec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print("%-22s threads=%d fold %.2fs %s %s"
              % (label, rec["threads"], fold_s, json.dumps(rec.get("split_ms", {})),
                 json.dumps(rec["cif"])), flush=True)

    def parse(spec):
        out = []
        for item in spec.split(","):
            if not item.strip():
                continue
            lab, _, th = item.partition(":")
            out.append((lab, int(th or 0)))
        return out

    timed_arms = parse(a.arms)
    run("cold", timed_arms[0][1], False)
    for i, (lab, th) in enumerate(timed_arms):
        run(f"{lab}_{i}", th, False)
    for lab, th in parse(a.instrument_arms):
        run(f"{lab}_instrumented", th, True)

    # Summary: per-label median fold, the paired ratio, and whether the CIF moved.
    import statistics as st
    labels = sorted({lab for lab, _ in timed_arms})
    summary = {}
    for lab in labels:
        folds = [r["fold_s"] for r in res["runs"]
                 if r["label"].startswith(lab + "_") and not r["instrumented"]]
        if folds:
            summary[lab] = {"n": len(folds), "median_fold_s": round(st.median(folds), 3),
                            "folds": folds}
    if len(labels) == 2 and all(l in summary for l in labels):
        a_, b_ = labels
        summary["ratio_%s_over_%s" % (a_, b_)] = round(
            summary[a_]["median_fold_s"] / summary[b_]["median_fold_s"], 4)
    cifs = {json.dumps(r["cif"], sort_keys=True) for r in res["runs"] if r["cif"]}
    summary["cif_identical_across_arms"] = (len(cifs) == 1)
    summary["cif_digests"] = sorted(cifs)
    res["summary"] = summary
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
